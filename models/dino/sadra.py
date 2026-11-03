"""Scale-Adaptive Dual-Region Attention used by DRCC-DETR.

The implementation is integrated into the multi-scale deformable-attention
operator so the existing decoder and CUDA aggregation retain their interfaces.
"""
from .ops.modules.ms_deform_attn import MSDeformAttn

# Explicit paper-facing name without adding a wrapper or changing tensor keys.
ScaleAdaptiveDualRegionAttention = MSDeformAttn

__all__ = ['ScaleAdaptiveDualRegionAttention']
