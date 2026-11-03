import torch
import torch.nn as nn
import torch.nn.functional as F


class VCRHead(nn.Module):
    """
    Lightweight Visual Contrastive Regularization scoring module.

    The two raw probability maps are kept for visualization and checkpoint
    compatibility.  During training, VCRLoss applies GT-derived
    positive/negative supports to `vcr_logits` and performs a masked spatial
    softmax, so these raw maps are not treated as unconstrained foreground and
    background pseudo labels.

    Input:
        feat: [B, C, H, W]
    Output:
        dict with
        - vcr_pos_map: [B, H, W]
        - vcr_neg_map: [B, H, W]
        - vcr_logits:  [B, 2, H, W]
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.pw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(32, hidden_dim)
        self.act = nn.GELU()

        squeeze_dim = max(hidden_dim // 4, 32)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, squeeze_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(squeeze_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.map_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)

    def forward(self, feat):
        local_feat = self.dw_conv(feat)
        local_feat = self.pw_conv(local_feat)
        gate = self.channel_gate(feat)
        fused = self.norm(feat + local_feat * gate)
        fused = self.act(fused)

        logits = self.map_head(fused)
        probs = F.softmax(logits, dim=1)
        return {
            'vcr_pos_map': probs[:, 0],
            'vcr_neg_map': probs[:, 1],
            'vcr_logits': logits,
        }
