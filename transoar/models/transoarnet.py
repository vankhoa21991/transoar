"""Main model of the transoar project."""

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from transoar.models.build import build_backbone, build_neck, build_pos_enc

class TransoarNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['neck']['hidden_dim']
        num_queries = config['neck']['num_queries']
        num_channels = config['backbone']['num_channels']

        # Use auxiliary decoding losses if required
        self._aux_loss = config['neck']['aux_loss']

        # Skip connection from backbone outputs to heads
        self._skip_con = config['neck']['skip_con']
        if self._skip_con:
            self._skip_proj = nn.Linear(
                config['backbone']['num_feature_patches'], 
                config['neck']['num_queries']
            )

        # Get anchors        
        self.anchors = self._generate_anchors(config['neck'], config['bbox_properties']).cuda()

        # Get backbone
        self._backbone = build_backbone(config['backbone'])

        # Get neck
        self._neck = build_neck(config['neck'], config['bbox_properties'])

        # Get heads
        self._cls_head = nn.Linear(hidden_dim, 2)
        self._bbox_reg_head = MLP(hidden_dim, hidden_dim, 6, 3)

        self._query_embed = nn.Embedding(num_queries, hidden_dim)
        self._input_proj = nn.Conv3d(num_channels, hidden_dim, kernel_size=1)

        # Get positional encoding
        self._pos_enc = build_pos_enc(config['neck'])

    def _reset_parameter(self):
        nn.init.constant_(self._bbox_reg_head.layers[-1].weight.data, 0)
        nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data, 0)

        for proj in self._input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, x, mask):
        out_backbone = self._backbone(x)

        srcs = self._input_proj(out_backbone[0][0])
        masks = out_backbone[0][1]
        pos = self._pos_enc(masks)

        out_neck = self._neck(             # [Batch, Queries, HiddenDim]         
            srcs,
            masks,
            self._query_embed.weight,
            pos
        )

        if self._skip_con:
            if isinstance(srcs, torch.Tensor):
                out_backbone_proj = srcs.flatten(2)
            else:
                out_backbone_proj = torch.cat([src.flatten(2) for src in srcs], dim=-1)
            out_backbone_skip_proj = self._skip_proj(out_backbone_proj).permute(0, 2, 1)
            out_neck = out_neck + out_backbone_skip_proj

        pred_logits = self._cls_head(out_neck)
        pred_boxes = self._bbox_reg_head(out_neck).tanh() * 0.2

        out = {
            'pred_logits': pred_logits[-1], # Take output of last layer
            'pred_boxes': pred_boxes[-1] + self.anchors
        }

        if self._aux_loss:
            out['aux_outputs'] = self._set_aux_loss(pred_logits, pred_boxes)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, pred_logits, pred_boxes):
        # Hack to support dictionary with non-homogeneous values
        return [{'pred_logits': a, 'pred_boxes': b + self.anchors}
                for a, b in zip(pred_logits[:-1], pred_boxes[:-1])]

    def _generate_anchors(self, model_config, bbox_props):
        median_bboxes = defaultdict(list)
        for class_, class_bbox_props in bbox_props.items():
            median_bboxes[int(class_)] = class_bbox_props['median']

        anchors = torch.zeros((model_config['num_queries'], 6))
        query_classes = torch.repeat_interleave(
            torch.arange(1, model_config['num_organs'] + 1), model_config['queries_per_organ'] * model_config['num_feature_levels']
        )

        anchor_offset = model_config['anchor_offsets']
        possible_offsets = torch.tensor([0, anchor_offset, -anchor_offset])
        offsets =  torch.cartesian_prod(
            possible_offsets, possible_offsets, possible_offsets
        ).repeat(model_config['num_organs'] * model_config['num_feature_levels'], 1)

        for idx, (query_class, offset) in enumerate(zip(query_classes, offsets)):
            query_median_box = torch.tensor(median_bboxes[query_class.item()])
            query_median_box[:3] += offset 
            anchors[idx] = query_median_box

        return anchors


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)
