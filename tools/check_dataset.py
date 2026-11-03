#!/usr/bin/env python3
"""Validate COCO-format dataset paths and unchanged category IDs before training."""
import argparse
import json
import math
from pathlib import Path


def check(root, split, num_classes=None):
    path = root / 'annotations' / ('instances_' + split + '2017.json')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    images = {r['id']: r for r in data['images']}
    categories = {r['id'] for r in data['categories']}
    if len(images) != len(data['images']) or len(categories) != len(data['categories']):
        raise ValueError('Duplicate image/category IDs: ' + str(path))
    if not categories or min(categories) < 0:
        raise ValueError('Invalid category IDs')
    if num_classes is not None and max(categories) >= num_classes:
        raise ValueError('num_classes must exceed the largest annotation category_id')
    missing = []
    image_root = (root / (split + '2017')).resolve()
    for row in images.values():
        image = (image_root / row['file_name']).resolve()
        if not image.is_relative_to(image_root):
            raise ValueError('Image file_name escapes the dataset root: ' + row['file_name'])
        if not image.is_file():
            missing.append(row['file_name'])
    ids = set()
    degenerate = 0
    for row in data['annotations']:
        if row['id'] in ids or row['image_id'] not in images or row['category_id'] not in categories:
            raise ValueError('Duplicate annotation ID or unknown image/category reference')
        ids.add(row['id'])
        box = row['bbox']
        if len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
            raise ValueError('Invalid bbox at annotation ' + str(row['id']))
        degenerate += min(box[2:]) <= 0
    result = {'split': split, 'images': len(images), 'annotations': len(ids),
              'category_ids': sorted(categories), 'required_num_classes': max(categories) + 1,
              'missing_images': len(missing), 'missing_examples': missing[:10],
              'nonpositive_extent_boxes_preserved': degenerate}
    print(json.dumps(result, indent=2))
    if missing:
        raise FileNotFoundError('Missing dataset images; see the examples above.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-path', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val'], default=['train', 'val'])
    parser.add_argument('--num-classes', type=int)
    args = parser.parse_args()
    results = [check(args.coco_path, split, args.num_classes) for split in args.splits]
    if any(row['category_ids'] != results[0]['category_ids'] for row in results):
        raise ValueError('Training/validation category IDs differ.')
