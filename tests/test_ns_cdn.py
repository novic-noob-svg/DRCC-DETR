#!/usr/bin/env python3
"""Focused CPU tests for Neighbor-Safe Contrastive Denoising."""

import os
import sys

import torch
from torch import nn


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.dino.dn_components import (  # noqa: E402
    _compute_cdn_collision_mask,
    prepare_for_cdn,
)


def test_collision_definition():
    gt_boxes = torch.tensor(
        [
            [0.25, 0.50, 0.20, 0.20],
            [0.65, 0.50, 0.20, 0.20],
        ],
        dtype=torch.float32,
    )
    # Query 0 originates from GT 0 but exactly covers GT 1: collision.
    # Query 1 originates from GT 1 and covers GT 1: no collision.
    candidates_xyxy = torch.tensor(
        [
            [0.55, 0.40, 0.75, 0.60],
            [0.55, 0.40, 0.75, 0.60],
        ],
        dtype=torch.float32,
    )
    collision, source_iou, alien_iou = _compute_cdn_collision_mask(
        noised_boxes_xyxy=candidates_xyxy,
        known_bid=torch.tensor([0, 0]),
        source_indices=torch.tensor([0, 1]),
        all_gt_boxes=gt_boxes,
        known_num=[2],
        min_alien_iou=0.1,
        ownership_margin=0.0,
    )
    assert collision.tolist() == [True, False]
    assert source_iou[0].item() < alien_iou[0].item()
    assert source_iou[1].item() > alien_iou[1].item()


def test_prepare_for_cdn_keeps_layout_and_reduces_collisions():
    torch.manual_seed(7)
    label_encoder = nn.Embedding(92, 256)
    targets = [
        {
            'labels': torch.tensor([1, 1], dtype=torch.long),
            'boxes': torch.tensor(
                [
                    [0.45, 0.50, 0.20, 0.20],
                    [0.55, 0.50, 0.20, 0.20],
                ],
                dtype=torch.float32,
            ),
        }
    ]
    query_label, query_bbox, attention_mask, dn_meta = prepare_for_cdn(
        dn_args=(targets, 100, 0.5, 0.6),
        training=True,
        num_queries=900,
        num_classes=91,
        hidden_dim=256,
        label_enc=label_encoder,
        use_ns_cdn=True,
        dn_collision_trials=3,
        dn_collision_iou_threshold=0.1,
        dn_collision_ownership_margin=0.0,
    )

    assert dn_meta['num_dn_group'] == 50
    assert dn_meta['pad_size'] == 200
    assert query_label.shape == (1, 200, 256)
    assert query_bbox.shape == (1, 200, 4)
    assert attention_mask.shape == (1100, 1100)
    assert 'ns_cdn_stats' in dn_meta

    stats = dn_meta['ns_cdn_stats']
    assert stats['pos_collision_after'] <= stats['pos_collision_before']
    assert stats['neg_collision_after'] <= stats['neg_collision_before']
    assert 1.0 <= stats['avg_trials'].item() <= 3.0


def test_ns_cdn_can_be_disabled():
    torch.manual_seed(11)
    label_encoder = nn.Embedding(92, 32)
    targets = [
        {
            'labels': torch.tensor([1], dtype=torch.long),
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        }
    ]
    _, _, _, dn_meta = prepare_for_cdn(
        dn_args=(targets, 100, 0.5, 0.6),
        training=True,
        num_queries=20,
        num_classes=91,
        hidden_dim=32,
        label_enc=label_encoder,
        use_ns_cdn=False,
    )
    assert 'ns_cdn_stats' not in dn_meta


def test_one_trial_preserves_standard_dino_noise():
    """K=1 must be diagnostic-only and reproduce the original CDN sample."""
    label_encoder = nn.Embedding(92, 32)
    targets = [
        {
            'labels': torch.tensor([1, 2], dtype=torch.long),
            'boxes': torch.tensor(
                [[0.45, 0.5, 0.2, 0.2], [0.55, 0.5, 0.2, 0.2]],
                dtype=torch.float32,
            ),
        }
    ]
    common_args = dict(
        dn_args=(targets, 100, 0.5, 0.6),
        training=True,
        num_queries=20,
        num_classes=91,
        hidden_dim=32,
        label_enc=label_encoder,
    )

    torch.manual_seed(23)
    standard = prepare_for_cdn(**common_args, use_ns_cdn=False)
    torch.manual_seed(23)
    one_trial = prepare_for_cdn(
        **common_args,
        use_ns_cdn=True,
        dn_collision_trials=1,
        dn_collision_iou_threshold=0.1,
        dn_collision_ownership_margin=0.0,
    )

    for standard_tensor, one_trial_tensor in zip(standard[:3], one_trial[:3]):
        assert torch.equal(standard_tensor, one_trial_tensor)
    assert one_trial[3]['ns_cdn_stats']['avg_trials'].item() == 1.0


if __name__ == '__main__':
    test_collision_definition()
    test_prepare_for_cdn_keeps_layout_and_reduces_collisions()
    test_ns_cdn_can_be_disabled()
    test_one_trial_preserves_standard_dino_noise()
    print('NS-CDN tests passed.')
