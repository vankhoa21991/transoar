/*!
**************************************************************************************************
* Deformable 3D DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved. TODO
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/fundamentalvision/Deformable-DETR
**************************************************************************************************
*/

#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    AT_ERROR("Not implement on cpu");
}

std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    AT_ERROR("Not implement on cpu");
}
