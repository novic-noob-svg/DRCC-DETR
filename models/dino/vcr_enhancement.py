"""Parameter-free VCR feature controls; the projection and loss stay shared."""

import torch
import torch.nn.functional as F


VCR_ENHANCEMENT_MODES = ('laplacian', 'identity', 'constant')


def apply_vcr_enhancement(features, laplacian_kernel, mode='laplacian'):
    """Apply the original gate or one of its two single-factor controls."""
    if mode == 'identity':
        return features
    if mode == 'constant':
        # Replace sigmoid(Laplacian(Fp)) by a fixed 0.5, not a learned gate.
        return features + features * 0.5
    if mode != 'laplacian':
        raise ValueError('Unknown VCR enhancement mode: {}'.format(mode))
    edge_response = F.conv2d(
        features, laplacian_kernel, padding=1, groups=features.shape[1])
    return features + features * torch.sigmoid(edge_response)
