"""CPU tests of best-weight retention, real training-loop branches and CSV output."""

import argparse
import ast
import csv
import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from ablation_results import write_summary


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_best_weights_test', ROOT / 'util/best_weights.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BestWeightsSaver = module.BestWeightsSaver


def metrics(ap):
    return [ap, 0.6, 0.4, 0.2, 0.3, -1.0] + [0.5] * 6


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


class BestWeightsTests(unittest.TestCase):
    def test_only_improvements_replace_one_weights_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(2, 1)
            saver = BestWeightsSaver(root)
            self.assertTrue(saver.update(model, 0, metrics(0), argparse.Namespace()))
            self.assertTrue((root / 'checkpoint_best.pth').is_file())
            with torch.no_grad():
                model.weight.fill_(7)
            self.assertTrue(saver.update(model, 1, metrics(0.25), argparse.Namespace()))
            saved_bytes = (root / 'checkpoint_best.pth').read_bytes()
            with torch.no_grad():
                model.weight.fill_(99)
            for ap in (0.24, 0.25, -1.0, float('nan'), float('inf')):
                self.assertFalse(saver.update(model, 2, metrics(ap), argparse.Namespace()))
            self.assertEqual((root / 'checkpoint_best.pth').read_bytes(), saved_bytes)
            checkpoint = load_checkpoint(root / 'checkpoint_best.pth')
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertNotIn('optimizer', checkpoint)
            self.assertNotIn('lr_scheduler', checkpoint)
            torch.testing.assert_close(checkpoint['model']['weight'], torch.full_like(model.weight, 7))
            self.assertEqual([path.name for path in root.glob('*.pth')], ['checkpoint_best.pth'])
            meta = json.loads((root / 'best_metrics.json').read_text())
            self.assertEqual(meta['AP'], 25.0)
            self.assertEqual(meta['epoch_number'], 2)
            self.assertIsNone(meta['APl'])

    def test_regular_and_ema_compete_for_one_best_file(self):
        with tempfile.TemporaryDirectory() as directory:
            saver = BestWeightsSaver(directory)
            regular, ema = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
            args = argparse.Namespace()
            saver.update(regular, 0, metrics(0.3), args)
            saver.update(ema, 0, metrics(0.4), args, 'ema')
            checkpoint = load_checkpoint(Path(directory) / 'checkpoint_best.pth')
            self.assertEqual(checkpoint['model_type'], 'ema')
            saver.update(regular, 1, metrics(0.5), args)
            self.assertFalse(saver.update(ema, 1, metrics(0.45), args, 'ema'))
            checkpoint = load_checkpoint(Path(directory) / 'checkpoint_best.pth')
            self.assertEqual(checkpoint['model_type'], 'regular')
            self.assertEqual(checkpoint['best_ap'], 0.5)
            self.assertEqual(len(list(Path(directory).glob('*.pth'))), 1)

    def test_failed_save_preserves_previous_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(2, 1)
            saver = BestWeightsSaver(root)
            saver.update(model, 0, metrics(0.1), argparse.Namespace())
            before = (root / 'checkpoint_best.pth').read_bytes()
            before_metrics = (root / 'best_metrics.json').read_bytes()
            def failed_write(weights, path):
                Path(path).write_bytes(b'partial write')
                raise OSError('simulated full disk')
            with patch.object(module.torch, 'save', side_effect=failed_write):
                with self.assertRaises(OSError):
                    saver.update(model, 1, metrics(0.2), argparse.Namespace())
            self.assertEqual((root / 'checkpoint_best.pth').read_bytes(), before)
            self.assertEqual((root / 'best_metrics.json').read_bytes(), before_metrics)
            self.assertEqual(saver.best_ap, 0.1)
            self.assertEqual(list(root.glob('*.tmp')), [])

    def test_csv_uses_best_epoch_and_leaves_unrun_metrics_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments = [{'name': 'sadra_no_scale', 'reference': 'sadra_reference'},
                           {'name': 'sadra_k1', 'reference': 'sadra_reference'}]
            run_dir = root / 'sadra_no_scale/seed_42'
            BestWeightsSaver(run_dir).update(torch.nn.Linear(2, 1), 3, metrics(0.37), argparse.Namespace())
            (run_dir / 'ablation_run.json').write_text('{"status": "completed"}')
            path = write_summary(root, experiments, [42])
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['AP'], '37.0')
            self.assertEqual(rows[0]['best_epoch'], '4')
            self.assertEqual(rows[0]['APl'], '')
            self.assertEqual(rows[1]['status'], 'pending')
            self.assertEqual(rows[1]['AP'], '')

    def test_actual_main_loop_skips_all_epoch_and_eval_files(self):
        # Execute the real training-loop AST with a tiny model and simulated
        # train/evaluate callbacks. This checks save-policy integration without
        # needing DINO's dataset, torchvision or compiled CUDA extension.
        tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        start = next(index for index, node in enumerate(main.body)
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == 'best_map_holder')
        loop_index = next(index for index in range(start, len(main.body))
                          if isinstance(main.body[index], ast.For))
        nodes = main.body[start:loop_index + 1]
        class StripHelperImport(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.module == 'util.best_weights':
                    return ast.copy_location(ast.Pass(), node)
                return node
        code = ast.fix_missing_locations(StripHelperImport().visit(ast.Module(body=nodes, type_ignores=[])))
        utilities = ast.parse((ROOT / 'util/utils.py').read_text(encoding='utf-8'))
        best_classes = [node for node in utilities.body if isinstance(node, ast.ClassDef)
                        and node.name in ('BestMetricSingle', 'BestMetricHolder')]
        for best_only in (True, False):
            with self.subTest(best_only=best_only), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = argparse.Namespace(output_dir=directory, save_best_only=best_only,
                    use_ema=False, start_epoch=0, epochs=3, distributed=False,
                    clip_max_norm=0.1, save_log=False, onecyclelr=False,
                    lr_drop=11, save_checkpoint_interval=1)
                model = torch.nn.Linear(2, 1)
                evaluation_scores = iter([0.1, 0.3, 0.2])
                evaluator = SimpleNamespace(coco_eval={'bbox': SimpleNamespace(eval={'tiny_eval': 1})})
                def evaluate(*unused, **kwargs):
                    return {'coco_eval_bbox': metrics(next(evaluation_scores))}, evaluator
                def save(payload, path):
                    torch.save(payload, path)
                scheduler = SimpleNamespace(step=lambda: None, state_dict=lambda: {})
                namespace = {'args': args, 'output_dir': root, 'time': time,
                    'datetime': datetime, 'json': json, 'torch': torch,
                    'BestWeightsSaver': BestWeightsSaver, 'model_without_ddp': model,
                    'model': model, 'ema_m': None, 'n_parameters': 3,
                    'criterion': None, 'data_loader_train': None, 'data_loader_val': None,
                    'postprocessors': None, 'base_ds': None, 'device': 'cpu',
                    'wo_class_error': False, 'lr_scheduler': scheduler,
                    'optimizer': SimpleNamespace(state_dict=lambda: {}),
                    'train_one_epoch': lambda *a, **kw: {}, 'evaluate': evaluate,
                    'utils': SimpleNamespace(is_main_process=lambda: True, save_on_master=save)}
                exec(compile(ast.Module(body=best_classes, type_ignores=[]), 'metric_classes', 'exec'), namespace)
                exec(compile(code, str(ROOT / 'main.py'), 'exec'), namespace)
                if best_only:
                    self.assertEqual(sorted(path.name for path in root.rglob('*.pth')),
                                     ['checkpoint_best.pth'])
                    self.assertEqual(load_checkpoint(root / 'checkpoint_best.pth')['epoch'], 1)
                    self.assertFalse((root / 'eval').exists())
                else:
                    self.assertTrue((root / 'checkpoint.pth').is_file())
                    self.assertTrue((root / 'checkpoint0002.pth').is_file())
                    self.assertTrue((root / 'checkpoint_best_regular.pth').is_file())
                    self.assertTrue((root / 'eval/latest.pth').is_file())
                self.assertEqual(len((root / 'log.txt').read_text().splitlines()), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
