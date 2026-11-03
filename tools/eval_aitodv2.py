#!/usr/bin/env python3
"""Evaluate existing DINO/VCR/SADRA weights with the official AI-TOD API.

Heavy model dependencies are imported only in checkpoint mode. --predictions
can re-evaluate a complete COCO detection JSON without Torch or CUDA.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.release_config import normalize_config

METRIC_NAMES = ('AP', 'AP50', 'AP75', 'APvt', 'APt', 'APs', 'APm')
METRIC_INDICES = (0, 2, 3, 4, 5, 6, 7)  # Official API inserts AP25 at index 1.
AREA_RANGES = [[0, 1e10], [0, 64], [64, 256], [256, 1024], [1024, 1e10]]
AREA_LABELS = ['all', 'verytiny', 'tiny', 'small', 'medium']
MAX_DETS = [1, 100, 1500]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--checkpoint', type=Path)
    mode.add_argument('--predictions', type=Path, help='Complete COCO detection JSON')
    mode.add_argument('--eval-cache', type=Path, help='Resume accumulation from this script\'s own evaluation_cache.pkl')
    p.add_argument('--no-eval-cache', action='store_true', help='Disable saving per-image evaluation cache')
    p.add_argument('--coco-path', '--coco_path', type=Path,
                   help='COCO-style root: val2017/ and annotations/instances_val2017.json')
    p.add_argument('--annotations', type=Path)
    p.add_argument('--image-root', type=Path, help='Root relative to annotation file_name')
    p.add_argument('--output-dir', type=Path, required=True, help='New/empty output directory')
    p.add_argument('--split-name', default='test', help='Metadata only; does not select images')
    p.add_argument('--run-config', type=Path)
    p.add_argument('--config-file', '-c', type=Path)
    p.add_argument('--weights-key', default='model', choices=['model', 'ema_model'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-select', type=int, default=1500)
    p.add_argument('--options', nargs='+', default=[], metavar='KEY=VALUE')
    p.add_argument('--category-map', type=Path,
                   help='JSON mapping model label to annotation category_id; default identity')
    for name in ('vcr', 'sadra'):
        group = p.add_mutually_exclusive_group()
        group.add_argument('--use-' + name, dest='use_' + name, action='store_true')
        group.add_argument('--no-use-' + name, dest='use_' + name, action='store_false')
    p.set_defaults(use_vcr=None, use_sadra=None)
    return p


def validate_annotations(data, diagnostics=None):
    if not data.get('images') or not data.get('annotations') or not data.get('categories'):
        raise ValueError('Evaluation needs images, ground-truth annotations and categories.')
    image_ids = {r['id'] for r in data['images']}
    if len(image_ids) != len(data['images']):
        raise ValueError('Duplicate image IDs')
    cats = {r['id'] for r in data['categories']}
    ann_ids = set()
    for a in data['annotations']:
        if a['id'] in ann_ids:
            raise ValueError('Duplicate annotation IDs')
        ann_ids.add(a['id'])
        if a['image_id'] not in image_ids or a['category_id'] not in cats:
            raise ValueError('Annotation refers to an unknown image/category')
        if 'area' not in a or 'iscrowd' not in a:
            raise ValueError('Use official annotations containing area and iscrowd.')
        box = a.get('bbox')
        detail = 'annotation_id=%s, image_id=%s, bbox=%r' % (a['id'], a['image_id'], box)
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError('Malformed ground-truth bbox: ' + detail)
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) for x in box):
            raise ValueError('Non-numeric/nonfinite ground-truth bbox: ' + detail)
        if not isinstance(a['area'], (int, float)) or not math.isfinite(a['area']):
            raise ValueError('Nonfinite/non-numeric ground-truth area: ' + detail)
        if min(box[2:]) <= 0 or a['area'] <= 0:
            # Official bbox evaluation passes annotation boxes and area through
            # without removing degenerate GT. Preserve the benchmark input;
            # independently filtering GT would change the measured protocol.
            issue = {'annotation_id': a['id'], 'image_id': a['image_id'],
                     'category_id': a['category_id'], 'bbox': list(box),
                     'area': a['area'], 'iscrowd': a['iscrowd'],
                     'reason': 'nonpositive_bbox_extent_or_area',
                     'action': 'preserved_for_official_evaluator'}
            if diagnostics is not None:
                diagnostics.append(issue)
    return image_ids, cats


def validate_predictions(rows, image_ids, cats):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Predictions must be a nonempty list of COCO detections.')
    for row in rows:
        if row['image_id'] not in image_ids or row['category_id'] not in cats:
            raise ValueError('Prediction image_id/category_id does not match annotations.')
        box = row['bbox']
        if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
            raise ValueError('Nonfinite/invalid prediction bbox')
        if min(box[2:]) < 0 or not math.isfinite(float(row['score'])):
            raise ValueError('Invalid prediction size/score')


def extract_metrics(stats):
    if len(stats) < 8:
        raise ValueError('Expected official AI-TOD stats, not the standard COCO layout.')
    # -1 means no evaluable GT, not a negative AP percentage.
    return {name: (float(stats[i]) * 100 if float(stats[i]) >= 0 else None)
            for name, i in zip(METRIC_NAMES, METRIC_INDICES)}


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


class NumpyCompatibility:
    """Compatibility local to aitodpycocotools; never modify global NumPy."""
    aliases = {'float': float, 'int': int, 'bool': bool, 'complex': complex,
               'str': str, 'object': object}

    def __init__(self, numpy_module):
        self.module = numpy_module

    def __getattr__(self, name):
        if name in self.aliases:
            return self.aliases[name]
        return getattr(self.module, name)


def load_official_api():
    try:
        from aitodpycocotools.coco import COCO
        import aitodpycocotools.cocoeval as api
    except ImportError as exc:
        raise RuntimeError('Install official API: pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"') from exc
    if not isinstance(api.np, NumpyCompatibility):
        api.np = NumpyCompatibility(api.np)
    return COCO, api.COCOeval


def save_evaluation_cache(evaluator, metadata, path):
    # Do not pickle the full COCO GT/results or enormous IoU dictionaries.
    # accumulate() needs the per-image matches and their parameter snapshots.
    payload = {'format_version': 1, 'metadata': metadata,
               'params': evaluator.params, 'params_eval': evaluator._paramsEval,
               'eval_imgs': evaluator.evalImgs}
    partial = path.with_suffix('.partial.pkl')
    with partial.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    partial.replace(path)


def restore_evaluation_cache(path, evaluator_class):
    # Only load a cache generated by this script on your own machine.
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported evaluation cache format')
    evaluator = evaluator_class(iouType='bbox')
    evaluator.params = payload['params']
    evaluator._paramsEval = payload['params_eval']
    evaluator.evalImgs = payload['eval_imgs']
    return evaluator, payload['metadata']


def write_metrics(evaluator, metadata, output_dir):
    metrics = extract_metrics(evaluator.stats)
    metadata.update({'metric_unit': 'percent', 'metrics': metrics,
                     'raw_stats': evaluator.stats.tolist()})
    (output_dir / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with (output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_NAMES)
        writer.writeheader()
        writer.writerow(metrics)
    latex = ' & '.join('--' if metrics[n] is None else '%.2f' % metrics[n] for n in METRIC_NAMES) + r' \\'
    (output_dir / 'latex_row.txt').write_text(latex + '\n', encoding='utf-8')
    print('\nAP / AP50 / AP75 / APvt / APt / APs / APm (percent):\n' + latex)
    print('Saved results to ' + str(output_dir.resolve()))


def resolve_data_paths(cli):
    if cli.coco_path:
        cli.annotations = cli.annotations or cli.coco_path / 'annotations' / 'instances_val2017.json'
        cli.image_root = cli.image_root or cli.coco_path / 'val2017'
    if not cli.annotations:
        raise ValueError('Pass --coco-path or --annotations.')
    if cli.checkpoint and not cli.image_root:
        raise ValueError('Pass --coco-path or --image-root with --checkpoint.')
    return cli


def infer(cli, annotations, cats):
    import torch
    from PIL import Image
    from main import build_model_main, get_args_parser, load_model_state
    from util.slconfig import SLConfig, DictAction
    from util.misc import nested_tensor_from_tensor_list
    from datasets.coco import make_coco_transforms
    import logging

    checkpoint_path = cli.checkpoint.resolve()
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    defaults = vars(get_args_parser().parse_args(['-c', str(cli.config_file or 'unused')]))
    run_path = cli.run_config or checkpoint_path.parent / 'config_args_all.json'
    if cli.run_config or run_path.is_file():
        source = load_json(run_path)
        config_source = str(run_path.resolve())
    elif 'args' in checkpoint:
        saved = checkpoint['args']
        source = saved if isinstance(saved, dict) else vars(saved)
        config_source = 'checkpoint.args'
    elif cli.config_file:
        source = SLConfig.fromfile(str(cli.config_file))._cfg_dict.to_dict()
        config_source = str(cli.config_file.resolve())
    else:
        raise ValueError('Pass --run-config/--config-file, or keep config_args_all.json beside weights.')
    defaults.update(normalize_config(source))
    args = argparse.Namespace(**defaults)
    if cli.options:
        DictAction(['--options'], 'overrides')(None, args, cli.options)
        for key, value in args.overrides.items():
            setattr(args, key, value)
    for name in ('vcr', 'sadra'):
        value = getattr(cli, 'use_' + name)
        if value is not None:
            setattr(args, 'use_' + name, value)
        elif getattr(args, 'use_' + name, None) is None:
            raise ValueError('Config does not specify use_%s; pass --use-%s/--no-use-%s.' % (name, name, name))
    args.device = cli.device
    args.distributed = False
    args.rank = args.local_rank = args.gpu = 0
    args.world_size = 1
    args.eval = True
    args.num_select = cli.num_select
    args.backbone_pretrained = False  # All inference weights come from checkpoint.
    if args.masks:
        raise ValueError('This entry point evaluates bbox detection only.')
    device = torch.device(cli.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the original server environment or --device cpu.')
    if cli.weights_key not in checkpoint:
        raise ValueError('Missing checkpoint weights key: ' + cli.weights_key)
    model, _, postprocessors = build_model_main(args)
    result = load_model_state(model, checkpoint[cli.weights_key], logging.getLogger('aitodv2'), strict=True)
    if result.missing_keys:
        raise ValueError('Checkpoint/config mismatch; refusing random missing weights: ' + str(result.missing_keys))
    model.to(device).eval()
    transform = make_coco_transforms('val', fix_size=args.fix_size, args=args)
    category_map = ({int(k): int(v) for k, v in load_json(cli.category_map).items()}
                    if cli.category_map else {int(c): int(c) for c in cats})
    if not set(category_map.values()) <= cats:
        raise ValueError('Category map contains unknown annotation IDs.')
    if not cats <= set(category_map.values()):
        raise ValueError('Category map omits ground-truth categories.')
    resolved = vars(args).copy()
    (cli.output_dir / 'resolved_config.json').write_text(json.dumps(resolved, indent=2, default=str), encoding='utf-8')
    output = cli.output_dir / 'predictions.json'
    temp = cli.output_dir / 'predictions.partial.json'
    skipped = 0
    count = 0
    with temp.open('w', encoding='utf-8') as handle, torch.inference_mode():
        handle.write('[')
        for i, row in enumerate(annotations['images'], 1):
            with Image.open(cli.image_root / row['file_name']) as raw:
                image = raw.convert('RGB')
            if image.size != (row['width'], row['height']):
                raise ValueError('Image dimensions disagree with annotation: ' + row['file_name'])
            tensor, _ = transform(image, None)
            outputs = model(nested_tensor_from_tensor_list([tensor.to(device)]))
            num_classes = outputs['pred_logits'].shape[2]
            if min(category_map) < 0 or max(category_map) >= num_classes:
                raise ValueError('Category map labels exceed model class channels; check category IDs and training configuration.')
            # Exclude padding/non-benchmark channels before top-k so they cannot
            # consume the 1,500 valid candidate budget. Category identity is never
            # guessed or renumbered: use an explicit map for remapped training labels.
            invalid = [c for c in range(num_classes) if c not in category_map]
            if invalid:
                outputs['pred_logits'][..., invalid] = float('-inf')
            capacity = outputs['pred_logits'].shape[1] * outputs['pred_logits'].shape[2]
            if cli.num_select > capacity:
                raise ValueError('--num-select exceeds query/class candidate count; do not change num_queries.')
            sizes = torch.tensor([[row['height'], row['width']]], device=device)
            prediction = postprocessors['bbox'](outputs, sizes)[0]
            for box, score, label in zip(prediction['boxes'].cpu().tolist(),
                                         prediction['scores'].cpu().tolist(),
                                         prediction['labels'].cpu().tolist()):
                if label not in category_map:
                    skipped += 1  # Padding/background labels have no benchmark category.
                    continue
                if not math.isfinite(score) or score <= 0:
                    continue  # Underflow/fully masked candidates cannot match.
                x1, y1, x2, y2 = box
                detection = {'image_id': row['id'], 'category_id': category_map[label],
                             'bbox': [x1, y1, x2-x1, y2-y1], 'score': score}
                if count:
                    handle.write(',\n')
                json.dump(detection, handle)
                count += 1
            if i == 1 or i % 100 == 0 or i == len(annotations['images']):
                print('[inference] %d/%d images, %d detections' % (i, len(annotations['images']), count), flush=True)
        handle.write(']\n')
    temp.replace(output)
    return output, {'config_source': config_source, 'checkpoint': str(checkpoint_path),
                    'weights_key': cli.weights_key, 'num_select': cli.num_select,
                    'unmapped_predictions_skipped': skipped, 'category_map': category_map,
                    'use_sadra': args.use_sadra, 'use_vcr': args.use_vcr,
                    'sadra_use_boundary_band': args.sadra_use_boundary_band}


def main(cli):
    if cli.eval_cache:
        _, evaluator_class = load_official_api()
        if cli.output_dir.exists() and any(cli.output_dir.iterdir()):
            raise ValueError('Choose a new/empty output directory.')
        cli.output_dir.mkdir(parents=True, exist_ok=True)
        evaluator, metadata = restore_evaluation_cache(cli.eval_cache, evaluator_class)
        metadata['resumed_from_cache'] = str(cli.eval_cache.resolve())
        print('[cache] Loaded per-image matches; skipping model inference and evaluate().', flush=True)
        evaluator.accumulate()
        evaluator.summarize()
        write_metrics(evaluator, metadata, cli.output_dir)
        return
    resolve_data_paths(cli)
    if cli.num_select < 1:
        raise ValueError('--num-select must be positive')
    annotations = load_json(cli.annotations)
    diagnostics = []
    image_ids, cats = validate_annotations(annotations, diagnostics)
    if diagnostics:
        print('[annotations] %d nonpositive bbox/area records; preserved unchanged for the official evaluator. First records: %s'
              % (len(diagnostics), json.dumps(diagnostics[:3])), flush=True)
    COCO, COCOeval = load_official_api()
    if cli.output_dir.exists() and any(cli.output_dir.iterdir()):
        raise ValueError('Choose a new/empty output directory; existing results are not overwritten.')
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    (cli.output_dir / 'annotation_diagnostics.json').write_text(
        json.dumps({'annotation_count': len(annotations['annotations']),
                    'suspicious_count': len(diagnostics),
                    'removed_count': 0, 'records': diagnostics}, indent=2), encoding='utf-8')
    if cli.checkpoint:
        prediction_path, metadata = infer(cli, annotations, cats)
    else:
        prediction_path = cli.predictions.resolve()
        metadata = {'mode': 'offline_predictions', 'num_select': None}
    predictions = load_json(prediction_path)
    validate_predictions(predictions, image_ids, cats)
    gt = COCO(str(cli.annotations))
    dt = gt.loadRes(predictions)
    evaluator = COCOeval(gt, dt, 'bbox')
    # Set and record protocol explicitly rather than depending on package defaults.
    evaluator.params.maxDets = MAX_DETS.copy()
    evaluator.params.areaRng = [r.copy() for r in AREA_RANGES]
    evaluator.params.areaRngLbl = AREA_LABELS.copy()
    evaluator.params.imgIds = sorted(image_ids)  # Include images with no predictions.
    metadata.update({'split_name': cli.split_name, 'annotations': str(cli.annotations.resolve()),
                     'predictions': str(prediction_path.resolve()), 'image_count': len(image_ids),
                     'prediction_count': len(predictions), 'maxDets': MAX_DETS,
                     'suspicious_annotation_count': len(diagnostics),
                     'annotation_policy': 'preserved_unchanged',
                     'areaRng': AREA_RANGES, 'areaRngLbl': AREA_LABELS,
                     'iouThrs': evaluator.params.iouThrs.tolist(),
                     'evaluator_file': sys.modules[COCOeval.__module__].__file__,
                     'numpy_compatibility': 'local_builtin_aliases'})
    evaluator.evaluate()
    if not cli.no_eval_cache:
        cache_path = cli.output_dir / 'evaluation_cache.pkl'
        print('[cache] Saving per-image matches before accumulation...', flush=True)
        try:
            save_evaluation_cache(evaluator, metadata, cache_path)
            print('[cache] Saved ' + str(cache_path), flush=True)
        except OSError as exc:
            print('[cache] Save failed (%s); continuing accumulation without a complete cache.' % exc, flush=True)
    evaluator.accumulate()
    evaluator.summarize()
    write_metrics(evaluator, metadata, cli.output_dir)


if __name__ == '__main__':
    main(parser().parse_args())
