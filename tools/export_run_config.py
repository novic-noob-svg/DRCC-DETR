#!/usr/bin/env python3
"""Export a portable Python config from an actual saved config_args_all.json."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.release_config import normalize_config

RUNTIME = {'config_file', 'options', 'dataset_file', 'coco_path', 'coco_panoptic_path',
    'output_dir', 'note', 'device', 'seed', 'resume', 'pretrain_model_path',
    'finetune_ignore', 'start_epoch', 'eval', 'num_workers', 'test', 'debug',
    'find_unused_params', 'save_results', 'save_log', 'world_size', 'dist_url',
    'rank', 'local_rank', 'distributed', 'gpu', 'dist_backend', '_explicit_args'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refuse to overwrite ' + str(args.output))
    source = normalize_config(json.loads(args.input.read_text(encoding='utf-8-sig')))
    values = {key: value for key, value in source.items() if key not in RUNTIME}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('"""Exported from a saved training configuration."""\n' +
        '\n'.join(key + ' = ' + repr(value) for key, value in values.items()) + '\n', encoding='utf-8')
    print('Saved:', args.output)
