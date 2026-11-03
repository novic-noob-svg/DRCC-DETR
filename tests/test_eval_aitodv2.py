"""Run: python -m unittest discover -s tools -p test_eval_aitodv2.py -v"""
import unittest
import copy
import tempfile
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_aitodv2 as evaluation


class MetricProtocolTests(unittest.TestCase):
    def test_numpy_legacy_float_works_without_global_monkeypatch(self):
        import numpy as np
        before = dict(np.__dict__)
        proxy = evaluation.NumpyCompatibility(np)
        result = proxy.cumsum([[True, False, True]], axis=1).astype(dtype=proxy.float)
        self.assertEqual(result.tolist(), [[1.0, 1.0, 2.0]])
        self.assertEqual(set(before), set(np.__dict__))
        for name in evaluation.NumpyCompatibility.aliases:
            self.assertIs(np.__dict__.get(name), before.get(name))

    def test_evaluation_cache_roundtrip_preserves_match_records_and_params(self):
        original = SimpleNamespace(params={'maxDets': [1, 100, 1500]},
                                   _paramsEval={'imgIds': [10]},
                                   evalImgs=[{'dtMatches': [[7, 0]], 'dtScores': [.9, .2]}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evaluation_cache.pkl'
            evaluation.save_evaluation_cache(original, {'prediction_count': 2}, path)
            restored, metadata = evaluation.restore_evaluation_cache(path, lambda **kw: SimpleNamespace())
        self.assertEqual(restored.evalImgs, original.evalImgs)
        self.assertEqual(restored.params, original.params)
        self.assertEqual(restored._paramsEval, original._paramsEval)
        self.assertEqual(metadata['prediction_count'], 2)

    def test_cache_mode_requires_neither_annotations_nor_weights(self):
        cli = evaluation.parser().parse_args(['--eval-cache', 'cache.pkl', '--output-dir', 'new'])
        self.assertIsNone(cli.annotations)
        self.assertIsNone(cli.checkpoint)

    def annotation_data(self, box, area=0):
        return {'images': [{'id': 10}], 'categories': [{'id': 1}],
                'annotations': [{'id': 7, 'image_id': 10, 'category_id': 1,
                                 'bbox': box, 'area': area, 'iscrowd': 0}]}

    def test_zero_extent_is_preserved_and_diagnosed(self):
        data = self.annotation_data([1, 2, 0, 4])
        original = copy.deepcopy(data)
        diagnostics = []
        self.assertEqual(evaluation.validate_annotations(data, diagnostics), ({10}, {1}))
        self.assertEqual(data, original)
        self.assertEqual(diagnostics[0]['annotation_id'], 7)
        self.assertEqual(diagnostics[0]['action'], 'preserved_for_official_evaluator')

    def test_negative_extent_is_reported_without_changing_gt(self):
        data = self.annotation_data([1, 2, -1, 4], -4)
        original = copy.deepcopy(data)
        diagnostics = []
        evaluation.validate_annotations(data, diagnostics)
        self.assertEqual(data, original)
        self.assertEqual(len(diagnostics), 1)

    def test_malformed_annotation_has_actionable_id(self):
        for box in ([1, 2, 3], [1, 2, float('nan'), 4], [1, 2, '3', 4]):
            with self.assertRaisesRegex(ValueError, 'annotation_id=7'):
                evaluation.validate_annotations(self.annotation_data(box))

    def test_test_renamed_val2017_uses_exact_coco_paths(self):
        cli = evaluation.parser().parse_args(['--checkpoint', 'best.pth', '--coco-path', '/data/aitod', '--output-dir', 'new'])
        evaluation.resolve_data_paths(cli)
        self.assertEqual(cli.annotations, Path('/data/aitod/annotations/instances_val2017.json'))
        self.assertEqual(cli.image_root, Path('/data/aitod/val2017'))
        self.assertEqual(cli.split_name, 'test')

    def test_explicit_annotation_overrides_coco_default(self):
        cli = evaluation.parser().parse_args(['--predictions', 'pred.json', '--coco-path', '/data/aitod', '--annotations', 'specific.json', '--output-dir', 'new'])
        evaluation.resolve_data_paths(cli)
        self.assertEqual(cli.annotations, Path('specific.json'))

    def test_ap25_is_not_mistaken_for_ap50(self):
        metrics = evaluation.extract_metrics([.1, .99, .2, .3, .4, .5, .6, .7])
        self.assertEqual(metrics, dict(zip(evaluation.METRIC_NAMES, [10, 20, 30, 40, 50, 60, 70])))

    def test_unavailable_metric_is_not_negative_percent(self):
        self.assertIsNone(evaluation.extract_metrics([0, 0, 0, 0, -1, 0, 0, 0])['APvt'])

    def test_wrong_prediction_ids_rejected(self):
        row = {'image_id': 1, 'category_id': 9, 'bbox': [0, 0, 4, 4], 'score': .9}
        with self.assertRaises(ValueError):
            evaluation.validate_predictions([row], {1}, {2})
        row['category_id'] = 2
        evaluation.validate_predictions([row], {1}, {2})

    def test_nonfinite_prediction_rejected(self):
        row = {'image_id': 1, 'category_id': 2, 'bbox': [0, 0, 4, 4], 'score': float('nan')}
        with self.assertRaises(ValueError):
            evaluation.validate_predictions([row], {1}, {2})

    def test_official_api_perfect_detections_across_four_size_groups(self):
        try:
            COCO, COCOeval = evaluation.load_official_api()
        except RuntimeError:
            self.skipTest('Official AI-TOD API is not installed in this environment')
        gt = COCO()
        gt.dataset = {'images': [{'id': 1, 'width': 300, 'height': 300}],
                      'categories': [{'id': 1, 'name': 'object'}],
                      'annotations': [{'id': i+1, 'image_id': 1, 'category_id': 1,
                                       'bbox': [i*65, 0, side, side], 'area': side**2,
                                       'iscrowd': 0} for i, side in enumerate((4, 12, 24, 48))]}
        gt.createIndex()
        predictions = [{'image_id': 1, 'category_id': 1, 'bbox': a['bbox'], 'score': .99}
                       for a in gt.dataset['annotations']]
        evaluator = COCOeval(gt, gt.loadRes(predictions), 'bbox')
        evaluator.params.maxDets = evaluation.MAX_DETS.copy()
        evaluator.params.areaRng = evaluation.AREA_RANGES
        evaluator.params.areaRngLbl = evaluation.AREA_LABELS
        evaluator.evaluate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evaluation_cache.pkl'
            evaluation.save_evaluation_cache(evaluator, {}, path)
            restored, _ = evaluation.restore_evaluation_cache(path, COCOeval)
            restored.accumulate()
            restored.summarize()
            for value in evaluation.extract_metrics(restored.stats).values():
                self.assertAlmostEqual(value, 100.0)
        evaluator.accumulate()
        evaluator.summarize()
        for value in evaluation.extract_metrics(evaluator.stats).values():
            self.assertAlmostEqual(value, 100.0)


if __name__ == '__main__':
    unittest.main()
