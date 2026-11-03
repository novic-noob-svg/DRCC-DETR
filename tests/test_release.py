"""Focused regression tests for checkpoint/config migration and attention."""
import argparse
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.checkpoint import load_model_state, migrate_state_dict
from util.release_config import normalize_config, resolve_training_config
from main import get_args_parser, build_model_main
from models.dino.ops.modules.ms_deform_attn import MSDeformAttn


class ReleaseTests(unittest.TestCase):
    def test_legacy_config_names_and_disabled_ns_default(self):
        cfg = normalize_config({'use_vca': True, 'use_vca_dn': True,
                                'vca_dn_loss_coef': 0.5})
        self.assertTrue(cfg['use_vcr'])
        self.assertEqual(cfg['vcr_loss_coef'], 0.5)
        self.assertFalse(cfg['use_ns_cdn'])
        self.assertEqual(normalize_config({'modelname': 'dcc_detr'})['modelname'], 'drcc_detr')
        self.assertEqual(normalize_config({'modelname': 'drcc_detr'})['modelname'], 'drcc_detr')
        self.assertEqual(normalize_config({'modelname': 'dino'})['modelname'], 'dino')
        self.assertNotIn('use_ns_cdn', normalize_config({'num_select': 100}, defaults=False))
        with self.assertRaises(ValueError):
            normalize_config({'use_vca': False, 'use_vcr': True})

    def test_final_config_and_explicit_cli_override(self):
        cfg = str(ROOT / 'config/DRCC_DETR/aitodv2_r50_5scale.py')
        args = get_args_parser().parse_args(['-c', cfg, '--no_use_vcr',
                                            '--sadra_max_scale', '3.0'])
        args, _ = resolve_training_config(args)
        self.assertFalse(args.use_vcr)
        self.assertTrue(args.use_sadra)
        self.assertEqual(args.sadra_max_scale, 3.0)
        self.assertTrue(args.sadra_use_boundary_band)
        self.assertFalse(args.use_ns_cdn)
        self.assertFalse(args.embed_init_tgt)
        self.assertEqual(args.num_classes, 9)
        self.assertEqual(args.num_feature_levels, 5)
        self.assertEqual(args.modelname, 'drcc_detr')
        self.assertEqual(args.lr_drop_list, [13, 23])

    def test_legacy_weights_load_and_mismatch_is_rejected(self):
        model = torch.nn.Module()
        model.vcr_feat_proj = torch.nn.Linear(4, 3)
        old = {'module.' + key.replace('vcr_feat_proj', 'vca_feat_proj'): value.clone()
               for key, value in model.state_dict().items()}
        result = load_model_state(model, old, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        with self.assertRaises(RuntimeError):
            load_model_state(model, {}, strict=True)
        with self.assertRaises(ValueError):
            migrate_state_dict({'vca_feat_proj.weight': torch.ones(1),
                                'vcr_feat_proj.weight': torch.ones(1)})

    def test_attention_cpu_forward_backward_and_optional_capture(self):
        torch.manual_seed(6)
        module = MSDeformAttn(d_model=32, n_levels=2, n_heads=4, n_points=4,
                             use_sadra=True)
        query = torch.randn(2, 3, 32, requires_grad=True)
        value = torch.randn(2, 20, 32, requires_grad=True)
        refs = torch.rand(2, 3, 2, 4).clamp_min(0.02)
        shapes = torch.tensor([[4, 4], [2, 2]])
        starts = torch.tensor([0, 16])
        output = module(query, refs, value, shapes, starts)
        self.assertEqual(output.shape, (2, 3, 32))
        self.assertIsNone(module.last_sampling_locations)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertIsNotNone(module.sadra_context_logit.grad)
        module.capture_sampling = True
        captured = module(query, refs, value, shapes, starts)
        torch.testing.assert_close(output, captured, rtol=0, atol=0)
        self.assertEqual(module.last_sampling_locations.shape, (2, 3, 4, 2, 4, 2))

    def test_allocation_and_prior_ablations(self):
        refs = torch.full((1, 2, 1, 4), 0.01)
        offsets = torch.ones(1, 2, 4, 1, 4, 2)
        logits = torch.zeros(1, 2, 4, 1, 4)
        for k in (1, 2, 3):
            module = MSDeformAttn(32, 1, 4, 4, use_sadra=True,
                sadra_use_boundary_band=False, sadra_context_points=k)
            adjusted, priors = module._apply_sadra(refs, offsets, logits)
            torch.testing.assert_close(adjusted[..., :4-k, :], offsets[..., :4-k, :])
            torch.testing.assert_close(adjusted[..., 4-k:, :], offsets[..., 4-k:, :] * 2.5)
            weights = priors.softmax(-1)
            torch.testing.assert_close(weights, torch.full_like(weights, 0.25))
            self.assertEqual(sum(p.numel() for n, p in module.named_parameters()
                                 if n.startswith('sadra_')), 1)

    def test_small_end_to_end_training_graph(self):
        args = get_args_parser().parse_args(['-c', str(ROOT / 'config/DRCC_DETR/aitodv2_r50_5scale.py'), '--device', 'cpu'])
        args, _ = resolve_training_config(args)
        args.backbone_pretrained = False
        # DINO's reference-box sine embedding is defined for hidden_dim=256.
        args.hidden_dim = 256
        args.nheads = 8
        args.dim_feedforward = 64
        args.enc_layers = args.dec_layers = 1
        args.num_feature_levels = 4
        args.return_interm_indices = [1, 2, 3]
        args.num_queries = args.num_select = 5
        args.dn_number = 2
        model, criterion, _ = build_model_main(args)
        model.train()
        images = [torch.randn(3, 64, 64), torch.randn(3, 64, 64)]
        targets = [{'boxes': torch.tensor([[0.45, 0.45, 0.2, 0.2]]),
                    'labels': torch.tensor([1])} for _ in images]
        output = model(images, targets)
        losses = criterion(output, targets)
        self.assertIn('loss_vcr', losses)
        total = sum(value * criterion.weight_dict[key] for key, value in losses.items()
                    if key in criterion.weight_dict)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(any(p.grad is not None for p in model.vcr_head.parameters()))
        self.assertIsNotNone(model.transformer.decoder.layers[0].cross_attn.sadra_context_logit.grad)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
