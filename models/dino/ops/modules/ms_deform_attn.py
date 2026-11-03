# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4,
                 use_sadra=False, sadra_max_scale=2.5,
                 sadra_scale_threshold=0.1,
                 sadra_use_boundary_band=True,
                 sadra_boundary_band_width=0.2,
                 sadra_use_scale_expansion=True,
                 sadra_use_region_prior=True,
                 sadra_context_points=None):
        r"""
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.use_sadra = use_sadra

        if use_sadra:
            if n_points < 2:
                raise ValueError('SADRA requires at least two sampling points per attention head')
            if sadra_max_scale < 1.0:
                raise ValueError('sadra_max_scale must be greater than or equal to 1.0')
            if not 0.0 < sadra_scale_threshold <= 1.0:
                raise ValueError('sadra_scale_threshold must be in (0, 1]')
            if (sadra_use_boundary_band and
                    not 0.0 <= sadra_boundary_band_width < 1.0):
                raise ValueError(
                    'sadra_boundary_band_width must be in [0, 1)')

            self.sadra_max_scale = float(sadra_max_scale)
            self.sadra_scale_threshold = float(sadra_scale_threshold)
            self.sadra_use_boundary_band = bool(sadra_use_boundary_band)
            self.sadra_boundary_band_width = float(
                sadra_boundary_band_width)
            self.sadra_use_scale_expansion = bool(sadra_use_scale_expansion)
            self.sadra_use_region_prior = bool(sadra_use_region_prior)
            # Preserve the default half split; ablate K without changing P.
            if sadra_context_points is None:
                sadra_context_points = n_points // 2
            if (not isinstance(sadra_context_points, int) or
                    isinstance(sadra_context_points, bool) or
                    not 1 <= sadra_context_points < n_points):
                raise ValueError('sadra_context_points must be an integer satisfying 1 <= K < n_points')
            self.sadra_context_points = sadra_context_points

            # One scalar per decoder layer replaces the query-conditioned MLP.
            # Its sigmoid is the total attention prior assigned to context
            # points.  Initializing it from the point ratio gives every point
            # the same prior at the start of training, so enabling SADRA does
            # not initially bias one region merely because it has fewer points.
            base_context_ratio = self.sadra_context_points / self.n_points
            base_context_logit = math.log(
                base_context_ratio / (1.0 - base_context_ratio))
            self.sadra_context_logit = nn.Parameter(
                torch.tensor(base_context_logit, dtype=torch.float32),
                requires_grad=self.sadra_use_region_prior)
            # Keep the state_dict key when disabling the prior, but freeze it
            # so DDP does not see an unused trainable parameter.

            # When a learned offset is exactly zero its ray direction is
            # undefined. Use the corresponding head's standard radial
            # initialization as a deterministic, parameter-free fallback.
            fallback_thetas = (
                torch.arange(n_heads, dtype=torch.float32)
                * (2.0 * math.pi / n_heads))
            fallback_directions = torch.stack(
                [fallback_thetas.cos(), fallback_thetas.sin()], dim=-1)
            fallback_directions = (
                fallback_directions
                / fallback_directions.abs().amax(dim=-1, keepdim=True))
            self.register_buffer(
                'sadra_fallback_directions',
                fallback_directions.view(1, 1, n_heads, 1, 1, 2),
                persistent=False)
        else:
            self.sadra_max_scale = 1.0
            self.sadra_scale_threshold = 1.0
            self.sadra_use_boundary_band = False
            self.sadra_boundary_band_width = 0.0
            self.sadra_use_scale_expansion = False
            self.sadra_use_region_prior = False
            self.sadra_context_points = 0
            self.register_parameter('sadra_context_logit', None)
            self.register_buffer(
                'sadra_fallback_directions', torch.empty(0),
                persistent=False)

        # Populated on the most recent forward pass for optional diagnostics.
        self.last_sampling_locations = None
        self.last_baseline_sampling_locations = None
        self.capture_sampling = False

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        # sadra_context_logit is initialized in __init__ because its value is
        # determined by the fixed local/context point ratio.

    def _project_to_boundary_band(self, context_offsets):
        """Project offsets to a differentiable band around a reference box.

        DINO converts an offset ``p`` into a location with

            location = center + (p / P) * (box_wh / 2),

        where ``P`` is the number of points per head and level. Therefore the
        reference-box boundary is exactly ``||p / P||_inf = 1``. The radial
        map below keeps the learned direction but compresses the radius into
        ``[1 - delta, 1 + delta]``. Points can consequently lie just inside or
        just outside the current reference box instead of being forced onto an
        exact, non-differentiable border.
        """
        normalized_offsets = context_offsets / float(self.n_points)
        radius = normalized_offsets.abs().amax(dim=-1, keepdim=True)
        eps = max(torch.finfo(context_offsets.dtype).eps, 1e-6)
        direction = normalized_offsets / radius.clamp_min(eps)

        fallback = self.sadra_fallback_directions.to(
            device=context_offsets.device, dtype=context_offsets.dtype)
        direction = torch.where(radius > eps, direction, fallback)

        boundary_radius = 1.0 + self.sadra_boundary_band_width * torch.tanh(
            radius - 1.0)
        return direction * boundary_radius * float(self.n_points)

    def _apply_sadra(self, reference_points, sampling_offsets,
                     attention_logits):
        """Apply scale-adaptive dual-region sampling to decoder queries.

        The first sampling points remain object-centric. The last
        sadra_context_points are expanded according to the reference-box area.
        By default, the expanded context points are then softly constrained to
        a narrow band straddling the reference-box boundary. A single learnable
        scalar controls the relative attention mass assigned to the two regions
        in each layer.
        """
        if reference_points.shape[-1] != 4:
            raise ValueError(
                'SADRA requires 4-D decoder reference boxes (cx, cy, w, h), '
                f'but received last dimension {reference_points.shape[-1]}')

        eps = torch.finfo(attention_logits.dtype).eps
        box_wh_per_level = reference_points[..., 2:].clamp(min=eps, max=1.0)
        object_scale = torch.sqrt(
            box_wh_per_level[..., 0] * box_wh_per_level[..., 1])
        # A geometrically interpretable inverse-size mapping.  Objects whose
        # normalized linear scale is above the threshold keep the baseline
        # offsets; smaller objects receive proportionally wider context until
        # sadra_max_scale is reached.
        context_scale = (
            self.sadra_scale_threshold / object_scale
        ).clamp(min=1.0, max=self.sadra_max_scale)

        num_object_points = self.n_points - self.sadra_context_points
        local_offsets = sampling_offsets[..., :num_object_points, :]
        context_offsets = sampling_offsets[..., num_object_points:, :]
        if self.sadra_use_scale_expansion:
            context_offsets = context_offsets * context_scale[
                :, :, None, :, None, None]
        if self.sadra_use_boundary_band:
            context_offsets = self._project_to_boundary_band(context_offsets)
        sampling_offsets = torch.cat(
            [local_offsets, context_offsets], dim=-2)

        if not self.sadra_use_region_prior:
            # No region term, even if a loaded checkpoint contains a trained
            # context logit. Ordinary learned attention remains unchanged.
            return sampling_offsets, attention_logits

        context_mask = sampling_offsets.new_zeros(self.n_points)
        context_mask[num_object_points:] = 1.0

        context_gate = torch.sigmoid(
            self.sadra_context_logit.to(dtype=attention_logits.dtype))

        context_mask_logits = context_mask.view(1, 1, 1, 1, self.n_points)
        object_points = self.n_points - self.sadra_context_points
        object_weight = (1.0 - context_gate) / object_points
        context_weight = context_gate / self.sadra_context_points
        region_weight = ((1.0 - context_mask_logits) * object_weight
                         + context_mask_logits * context_weight)
        attention_logits = attention_logits + torch.log(region_weight.clamp_min(eps))
        return sampling_offsets, attention_logits

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        r"""
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_logits = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)
        baseline_offsets = sampling_offsets
        if self.use_sadra:
            sampling_offsets, attention_logits = self._apply_sadra(
                reference_points, sampling_offsets, attention_logits)
        attention_weights = F.softmax(
            attention_logits.flatten(-2), -1).view(
                N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            if self.capture_sampling:
                baseline_locations = reference_points[:, :, None, :, None, :] \
                                     + baseline_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            if self.capture_sampling:
                baseline_locations = reference_points[:, :, None, :, None, :2] \
                                     + baseline_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        if self.capture_sampling:
            self.last_baseline_sampling_locations = baseline_locations.detach()
            self.last_sampling_locations = sampling_locations.detach() if self.use_sadra else None

        if value.device.type == 'cpu':
            # The upstream reference implementation permits CPU regression tests.
            # CUDA training still uses the original compiled operator below.
            output = ms_deform_attn_core_pytorch(
                value, input_spatial_shapes, sampling_locations, attention_weights)
            return self.output_proj(output)

        # for amp
        if value.dtype == torch.float16:
            # for mixed precision
            output = MSDeformAttnFunction.apply(
            value.to(torch.float32), input_spatial_shapes, input_level_start_index, sampling_locations.to(torch.float32), attention_weights, self.im2col_step)
            output = output.to(torch.float16)
            output = self.output_proj(output)
            return output


        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
