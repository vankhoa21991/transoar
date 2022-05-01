"""Focused decoder class."""

import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import trunc_normal_


class FocusedDecoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6, 
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        bbox_props=None,
        config=None
    ):
        super().__init__()
        self.bbox_props = bbox_props
        self.config = config

        self.shapes = {
            'P0': [160, 160, 256],
            'P1': [80, 80, 128],
            'P2': [40, 40, 64],
            'P3': [20, 20, 32],
            'P4': [10, 10, 16],
            'P5': [5, 5, 8]
        }

        self.d_model = d_model
        self.nhead = nhead

        attn_mask = self.generate_attn_mask()
        decoder_layer = FocusedDecoderLayer(d_model, dim_feedforward, dropout, activation, nhead, attn_mask)
        self.decoder = FocusedDecoderModel(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self._reset_parameters()

    def generate_attn_mask(self, padding=0):
        assert self.config['num_queries'] == self.config['queries_per_organ'] * self.config['num_feature_levels'] * self.config['num_organs']
        input_shape = [self.shapes[self.config['input_level']]]

        # Init full attn mask
        num_patches_per_lvl = torch.tensor(input_shape).prod(axis=1)
        attn_mask = torch.ones((self.config['num_queries'], num_patches_per_lvl.sum()), dtype=torch.bool)

        # Get per class volume to attend to at different feature map lvls
        attn_volumes = defaultdict(list)
        for class_, props in self.bbox_props.items():
            attn_volume_normalized = torch.tensor(props['attn_area'])   # x1, y1, z1, x2, y2, z2

            # Show information about class boxes
            # from transoar.utils.bboxes import box_cxcyczwhd_to_xyzxyz
            # median_box = box_cxcyczwhd_to_xyzxyz(torch.tensor(props['median']))
            # max_box = box_cxcyczwhd_to_xyzxyz(torch.tensor(props['max']))
            # print(
            #     class_,
            #     # (attn_volume_normalized[3:] -  attn_volume_normalized[:3]).tolist(),
            #     (median_box[3:] -  median_box[:3]).tolist(),
            #     (max_box[3:] -  max_box[:3]).tolist(),
            # )

            for fmap_shape in input_shape:
                attn_volume = torch.tensor(
                    [
                        torch.floor(attn_volume_normalized[0] * fmap_shape[0]) - padding,   # x1
                        torch.floor(attn_volume_normalized[1] * fmap_shape[1]) - padding,   # y1
                        torch.floor(attn_volume_normalized[2] * fmap_shape[2]) - padding,   # z1
                        torch.ceil(attn_volume_normalized[3] * fmap_shape[0]) + padding,    # x2
                        torch.ceil(attn_volume_normalized[4] * fmap_shape[1]) + padding,    # y2
                        torch.ceil(attn_volume_normalized[5] * fmap_shape[2]) + padding     # z2
                    ]
                )
                attn_volumes[int(class_)].append(attn_volume.to(dtype=torch.int))

        # Set attn mask to mask out region which is not in desired attn volume
        query_classes = torch.repeat_interleave(torch.arange(1, self.config['num_organs'] + 1), self.config['queries_per_organ'] * self.config['num_feature_levels'])
        query_fmap_lvls = torch.repeat_interleave(torch.arange(self.config['num_feature_levels']), self.config['queries_per_organ']).repeat(self.config['num_organs']) 
        for query_attn_volume, query_class, query_fmap_lvl in zip(attn_mask, query_classes, query_fmap_lvls):
            # Retrieve class attn volume of current query
            dummy_fmap = torch.zeros(input_shape[query_fmap_lvl.item()])
            class_attn_volume = attn_volumes[query_class.item()][query_fmap_lvl.item()]

            # Restrict attn to region of interest
            dummy_fmap[class_attn_volume[0]:class_attn_volume[3], class_attn_volume[1]:class_attn_volume[4], class_attn_volume[2]:class_attn_volume[5]] = 1
            dummy_fmap_flattened = dummy_fmap.flatten().nonzero()

            if query_fmap_lvl.item() > 0:
                dummy_fmap_flattened += num_patches_per_lvl[query_fmap_lvl.item() - 1]

            query_attn_volume[dummy_fmap_flattened] = False

        return attn_mask #torch.zeros_like(attn_mask, dtype=torch.bool)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, srcs, query_embed, pos_embeds):
        assert query_embed is not None

        # prepare input for decoder
        src_flatten = srcs.flatten(2).transpose(1, 2)                                # [Batch, Patches, HiddenDim] 
        pos_embed_flatten = pos_embeds.flatten(2).transpose(1, 2)                    # [Batch, Patches, HiddenDim] 
        
        bs, _, c = src_flatten.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)                        # Tgt in contrast to detr not zeros, but learnable
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        # decoder
        hs = self.decoder(tgt, src_flatten, pos_embed_flatten, query_embed)

        return hs

class FocusedDecoderModel(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, src, src_pos, query_pos=None):
        output = tgt

        intermediate = []
        for layer in self.layers:
            output = layer(output, query_pos, src_pos, src)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class FocusedDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1, 
        activation="relu",
        n_heads=8,
        attn_mask=None
    ):
        super().__init__()
        self.attn_mask = attn_mask

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, src_pos, src):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(src, src_pos)
        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), src.transpose(0, 1), attn_mask=self.attn_mask.to(device=tgt.device))[0].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# class FocusedAttnDecoder(nn.Module):
#     def __init__(
#         self,
#         d_model=256,
#         nhead=8,
#         num_decoder_layers=6, 
#         dim_feedforward=1024,
#         dropout=0.1,
#         activation="relu",
#         return_intermediate_dec=False,
#         bbox_props=None,
#         config=None,
#         padding=0
#     ):
#         super().__init__()
#         self._bbox_props = bbox_props
#         self._config = config

#         self._d_model = d_model
#         self._nhead = nhead
#         self._return_intermediate = return_intermediate_dec

#         # Get attn volume mask for all classes in the dataset
#         self._attn_volume_masks = defaultdict(list)
#         for class_, props in self._bbox_props.items():
#             attn_volume_normalized = torch.tensor(props['attn_area'])   # x1, y1, z1, x2, y2, z2

#             for fmap_shape in self._config['input_shapes']:
#                 attn_volume = torch.tensor(
#                     [
#                         torch.floor(attn_volume_normalized[0] * fmap_shape[0]) - padding,   # x1
#                         torch.floor(attn_volume_normalized[1] * fmap_shape[1]) - padding,   # y1
#                         torch.floor(attn_volume_normalized[2] * fmap_shape[2]) - padding,   # z1
#                         torch.ceil(attn_volume_normalized[3] * fmap_shape[0]) + padding,    # x2
#                         torch.ceil(attn_volume_normalized[4] * fmap_shape[1]) + padding,    # y2
#                         torch.ceil(attn_volume_normalized[5] * fmap_shape[2]) + padding     # z2
#                     ]
#                 )
#                 self._attn_volume_masks[int(class_)].append(attn_volume.to(dtype=torch.int))

#         # Generate layers
#         layer = FocusedAttnDecoderLayer(d_model, dim_feedforward, dropout, activation, nhead, self._attn_volume_masks)
#         self.layers = _get_clones(layer, num_decoder_layers)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, srcs, query_embed, pos_embeds=None):
#         assert query_embed is not None

#         # Prepare tgt and query embedding 
#         bs, c, *_ = srcs.shape
#         query_embed, tgt = torch.split(query_embed, c, dim=1)                        # Tgt in contrast to detr not zeros, but learnable
#         query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
#         tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

#         # Decoder layers
#         output = tgt
#         intermediate = []
#         for layer in self.layers:
#             output = layer(output, query_embed, pos_embeds, srcs)

#             if self._return_intermediate:
#                 intermediate.append(output)

#         if self._return_intermediate:
#             return torch.stack(intermediate)

#         return output


# class FocusedAttnDecoderLayer(nn.Module):
#     def __init__(
#         self,
#         d_model=256,
#         d_ffn=1024,
#         dropout=0.1, 
#         activation="relu",
#         n_heads=8,
#         attn_volume_masks=None
#     ):
#         super().__init__()
#         self._attn_volume_masks = attn_volume_masks

#         # focused cross attention
#         self.cross_attn = nn.ModuleDict()
#         for key, value in attn_volume_masks.items():
#             attn_window = value[0][3:] - value[0][:3]
#             self.cross_attn[str(key)] = FocusedAttn(d_model, attn_window, 6, True, None, 0, 0)

#         # self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # self attention
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout3 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout4 = nn.Dropout(dropout)
#         self.norm3 = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, tgt):
#         tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout4(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt

#     def forward(self, tgt, query_pos, src_pos, srcs):
#         # Self attention
#         q = k = self.with_pos_embed(tgt, query_pos)
#         tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)

#         # Extract attn volumes
#         srcs = self.with_pos_embed(srcs, src_pos)
#         tgt = self.with_pos_embed(tgt, query_pos)
#         attn_volumes = {}
#         for key, value in self._attn_volume_masks.items():
#             x1, y1, z1, x2, y2, z2 = value[0]
#             attn_volumes[key] = srcs[:, :, x1:x2, y1:y2, z1:z2].flatten(2).transpose(1, 2)

#         # Focused cross attention
#         tgt = tgt.reshape(2, 20, 27, -1).transpose(0, 1)
#         for class_, attn_volume in attn_volumes.items():
#             tgt[class_ - 1] = self.cross_attn[str(class_)](attn_volume, tgt[class_ - 1])
#         tgt = tgt.transpose(0, 1).flatten(1, 2)

#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)

#         # FFN
#         tgt = self.forward_ffn(tgt)

#         return tgt


# class FocusedAttn(nn.Module):
#     def __init__(
#         self,
#         dim,
#         window_size,
#         num_heads,
#         qkv_bias,
#         qk_scale,
#         attn_drop,
#         proj_drop
#     ):
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wd, Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # learnable position bias
#         self.position_bias = nn.Parameter(
#             torch.zeros(6, 27, self.window_size.prod())
#         )

#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.position_bias, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, src_k, src_q):
#         B, N_k, C = src_k.shape
#         B, N_q, C = src_q.shape

#         # Project and split heads
#         k = self.k_proj(src_k).reshape(B, N_k, self.num_heads, C // self.num_heads)
#         v = self.v_proj(src_k).reshape(B, N_k, self.num_heads, C // self.num_heads)
#         q = self.k_proj(src_q).reshape(B, N_q, self.num_heads, C // self.num_heads)

#         q = q * self.scale
#         attn = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1))
#         attn = attn + self.position_bias

#         attn = self.softmax(attn)
#         attn = self.attn_drop(attn)

#         x = (attn @ v.permute(0, 2, 1, 3)).transpose(1, 2).reshape(B, N_q, C)

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x