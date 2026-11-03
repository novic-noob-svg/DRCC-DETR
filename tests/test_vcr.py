"""CPU smoke tests for the GT-constrained VCR auxiliary loss.

Run from the repository root:
    python tools/test_vcr_gt_constrained.py
"""

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vcr_loss_module = _load_module(
    'vcr_loss_standalone', 'models/dino/vcr_loss.py')
vcr_head_module = _load_module(
    'vcr_head_standalone', 'models/dino/vcr_head.py')
VCRLoss = vcr_loss_module.VCRLoss
VCRHead = vcr_head_module.VCRHead


def test_supports_and_gradients():
    torch.manual_seed(7)
    batch_size, channels, height, width = 2, 32, 8, 10
    feat = torch.randn(
        batch_size, channels, height, width, requires_grad=True)
    head = VCRHead(channels)
    maps = head(feat)

    # Image 0 has padding at the bottom and right; image 1 uses the full map.
    valid = torch.zeros(batch_size, height, width, dtype=torch.bool)
    valid[0, :6, :8] = True
    valid[1] = True

    targets = [
        {
            'labels': torch.tensor([1, 2]),
            # The second tiny box must still occupy at least one feature cell.
            'boxes': torch.tensor([
                [0.35, 0.40, 0.20, 0.25],
                [0.90, 0.90, 0.005, 0.005],
            ]),
        },
        {
            'labels': torch.tensor([3]),
            'boxes': torch.tensor([[0.50, 0.50, 0.20, 0.20]]),
        },
    ]

    # Two DN groups, with max_gt=2: each group contains two positive slots
    # followed by two negative slots.
    num_groups = 2
    pad_size = num_groups * 2 * 2
    query_embed = torch.randn(
        batch_size, pad_size, channels, requires_grad=True)
    outputs = {
        'query_embed': query_embed,
        'high_res_feat_proj': feat,
        'vcr_valid_mask': valid,
        **maps,
    }
    dn_meta = {'num_dn_group': num_groups, 'pad_size': pad_size}
    criterion = VCRLoss(temperature=0.07)

    pos_support, neg_support = criterion._build_gt_supports(
        targets[0]['boxes'], valid[0])
    assert pos_support is not None and neg_support is not None
    assert not bool((pos_support & neg_support).any())
    assert torch.equal(pos_support | neg_support, valid[0])

    pos_map, neg_map = criterion._get_maps(
        outputs, targets[0], 0, height, width, feat.device)
    assert pos_map is not None and neg_map is not None
    assert torch.isclose(pos_map.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(neg_map.sum(), torch.tensor(1.0), atol=1e-5)
    assert not bool(pos_map[~valid[0]].any())
    assert not bool(neg_map[~valid[0]].any())
    assert not bool(((pos_map > 0) & (neg_map > 0)).any())

    loss = criterion(outputs, targets, dn_meta)['loss_vcr']
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert query_embed.grad is not None
    assert torch.isfinite(query_embed.grad).all()
    assert feat.grad is not None
    assert torch.isfinite(feat.grad).all()
    assert head.map_head.weight.grad is not None


def test_no_valid_background_is_safe():
    height, width = 4, 5
    criterion = VCRLoss()
    outputs = {
        'vcr_logits': torch.zeros(1, 2, height, width),
        'vcr_valid_mask': torch.ones(1, height, width, dtype=torch.bool),
    }
    target = {
        'labels': torch.tensor([1]),
        'boxes': torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
    }
    pos_map, neg_map = criterion._get_maps(
        outputs, target, 0, height, width, torch.device('cpu'))
    assert pos_map is None and neg_map is None


def test_unconstrained_maps_use_all_valid_cells():
    height, width = 4, 5
    valid = torch.ones(1, height, width, dtype=torch.bool)
    valid[:, -1, :] = False
    logits = torch.randn(1, 2, height, width)
    outputs = {
        'vcr_logits': logits,
        'vcr_valid_mask': valid,
    }
    target = {
        'labels': torch.tensor([1]),
        'boxes': torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
    }
    criterion = VCRLoss(gt_constrained=False)
    pos_map, neg_map = criterion._get_maps(
        outputs, target, 0, height, width, torch.device('cpu'))

    assert pos_map is not None and neg_map is not None
    assert torch.isclose(pos_map.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(neg_map.sum(), torch.tensor(1.0), atol=1e-5)
    assert not bool(pos_map[~valid[0]].any())
    assert not bool(neg_map[~valid[0]].any())
    assert bool(((pos_map > 0) & (neg_map > 0)).any())


def test_negative_dn_query_loss_can_be_disabled():
    torch.manual_seed(11)
    channels, height, width = 8, 4, 5
    query_embed = torch.randn(1, 2, channels, requires_grad=True)
    feat = torch.randn(1, channels, height, width)
    outputs = {
        'query_embed': query_embed,
        'high_res_feat_proj': feat,
        'vcr_logits': torch.randn(1, 2, height, width),
        'vcr_valid_mask': torch.ones(1, height, width, dtype=torch.bool),
    }
    targets = [{
        'labels': torch.tensor([1]),
        'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
    }]
    dn_meta = {'num_dn_group': 1, 'pad_size': 2}
    criterion = VCRLoss(use_negative_queries=False)

    pos_map, neg_map = criterion._get_maps(
        outputs, targets[0], 0, height, width, feat.device)
    assert pos_map is not None and neg_map is not None
    feat_flat = feat.detach().flatten(2).transpose(1, 2)[0]
    pos_token = criterion._pool_token(feat_flat, pos_map)
    neg_token = criterion._pool_token(feat_flat, neg_map)
    expected = criterion._nce(query_embed[0, :1], pos_token, neg_token)

    loss = criterion(outputs, targets, dn_meta)['loss_vcr']
    assert torch.allclose(loss, expected)
    loss.backward()
    assert query_embed.grad is not None
    assert torch.count_nonzero(query_embed.grad[0, 0]) > 0
    assert torch.count_nonzero(query_embed.grad[0, 1]) == 0


if __name__ == '__main__':
    test_supports_and_gradients()
    test_no_valid_background_is_safe()
    test_unconstrained_maps_use_all_valid_cells()
    test_negative_dn_query_loss_can_be_disabled()
    print('GT-constrained VCR tests passed.')
