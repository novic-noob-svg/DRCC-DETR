#!/usr/bin/env python3
"""Plot query/prototype similarities for DINO versus DINO+VCR.

Reads COCO annotations directly, or the optional legacy boxes JSON.
The baseline uses uniform pooling inside/outside the GT support; VCR uses its
learned adaptive logits inside the same supports.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
import datasets.transforms as T
from main import build_model_main, load_model_state
from util.checkpoint import load_checkpoint
from inference_visualize import load_args

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-checkpoint', type=Path, required=True)
    p.add_argument('--vcr-checkpoint', type=Path, required=True)
    p.add_argument('--config-file', type=Path)
    p.add_argument('--run-config', type=Path)
    p.add_argument('--baseline-config-file', type=Path)
    p.add_argument('--baseline-run-config', type=Path)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--coco-annotations', type=Path)
    p.add_argument('--boxes', type=Path, help='Legacy JSON: {"image.jpg": [[x1,y1,x2,y2]]}')
    p.add_argument('--output-dir', type=Path, default=Path('visualizations/vcr'))
    p.add_argument('--device', default='cuda')
    return p


def make_args(cli, use_vcr, checkpoint_path):
    config_cli = argparse.Namespace(
        run_config=cli.run_config if use_vcr else cli.baseline_run_config,
        config_file=cli.config_file if use_vcr else (cli.baseline_config_file or cli.config_file),
        options=None, use_vcr=use_vcr,
        use_sadra=None if use_vcr else False, device=cli.device)
    return load_args(config_cli, checkpoint_path)


def load(cli, checkpoint_path, device, use_vcr):
    args = make_args(cli, use_vcr, checkpoint_path)
    model, _, _ = build_model_main(args)
    data = load_checkpoint(checkpoint_path)
    load_model_state(model, data.get('model', data), strict=True)
    model.visualize_vcr = True
    model.to(device).eval()
    return model


def box_mask(boxes, h, w, image_w, image_h, device):
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    for x1, y1, x2, y2 in boxes:
        left = max(0, min(w - 1, int(x1 / image_w * w)))
        top = max(0, min(h - 1, int(y1 / image_h * h)))
        right = max(left + 1, min(w, int(np.ceil(x2 / image_w * w))))
        bottom = max(top + 1, min(h, int(np.ceil(y2 / image_h * h))))
        mask[top:bottom, left:right] = True
    return mask


def read_boxes(cli, image):
    if cli.coco_annotations:
        with cli.coco_annotations.open(encoding='utf-8') as f:
            coco = json.load(f)
        image_rows = [row for row in coco['images']
                      if row['file_name'] == image.name or Path(row['file_name']).name == image.name]
        if not image_rows:
            raise ValueError('Image {} is not present in COCO annotations'.format(image.name))
        image_id = image_rows[0]['id']
        return [[x, y, x + width, y + height]
                for ann in coco['annotations'] if ann['image_id'] == image_id
                and not ann.get('iscrowd', 0)
                for x, y, width, height in [ann['bbox']]]
    if cli.boxes:
        with cli.boxes.open(encoding='utf-8') as f:
            entries = json.load(f)
        return entries.get(image.name, entries.get(str(image), entries.get('boxes', [])))
    raise ValueError('Pass --coco-annotations or --boxes')


def iter_images(input_path):
    if input_path.is_file():
        yield input_path
    elif input_path.is_dir():
        yield from sorted(path for path in input_path.rglob('*')
                          if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        raise FileNotFoundError('Input does not exist: {}'.format(input_path))


def similarity(model, samples, boxes, image):
    height, width = image.height, image.width
    target_boxes = torch.tensor([
        [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
         (x2 - x1) / width, (y2 - y1) / height]
        for x1, y1, x2, y2 in boxes
    ], device=samples.device, dtype=samples.dtype)
    targets = [{
        'boxes': target_boxes,
        'labels': torch.zeros(len(boxes), dtype=torch.long, device=samples.device),
    }]
    with torch.inference_mode():
        out = model(samples, targets=targets)
    feat = out['high_res_feat_proj'][0].float()
    # A baseline backbone can expose 512-channel features while decoder
    # queries live in the 256-dimensional hidden space. Reuse DINO's own
    # high-resolution input projection for a compatible representation.
    decoder_hs = out['hs']
    if torch.is_tensor(decoder_hs):
        query_dim = decoder_hs.shape[-1]
    else:
        query_dim = decoder_hs[-1].shape[-1]
    if feat.shape[0] != query_dim:
        if not hasattr(model, 'input_proj'):
            raise RuntimeError(
                'Feature/query dimensions differ ({} vs {}) and the model '
                'has no input projection.'.format(feat.shape[0], query_dim))
        with torch.inference_mode():
            feat = model.input_proj[0](feat.unsqueeze(0))[0].float()
    _, h, w = feat.shape
    positive = box_mask(boxes, h, w, image.width, image.height, feat.device)
    valid = torch.ones_like(positive)
    negative = valid & ~positive
    if not negative.any():
        raise ValueError('GT boxes cover the complete feature map; no negative prototype exists.')
    if 'vcr_logits' in out:
        logits = out['vcr_logits'][0].float()
        logits = F.interpolate(logits[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        pos_weight = torch.softmax(logits[0].masked_fill(~positive, -1e4).flatten(), 0).view(h, w)
        neg_weight = torch.softmax(logits[1].masked_fill(~negative, -1e4).flatten(), 0).view(h, w)
    else:
        pos_weight = positive.float()
        neg_weight = negative.float()
    tokens = feat.flatten(1).transpose(0, 1)
    pos = F.normalize((tokens * pos_weight.flatten()[:, None]).sum(0), dim=0)
    neg = F.normalize((tokens * neg_weight.flatten()[:, None]).sum(0), dim=0)
    # Match the training loss: use DN queries for the first GT and only the
    # first half of the embedding as the VCR semantic subspace.
    dn_meta = out.get('dn_meta')
    known = dn_meta.get('output_known_lbs_bboxes') if dn_meta else None
    if known is not None and 'query_embed' in known:
        query_embed = known['query_embed'][0]
        half_pad = (dn_meta['pad_size'] // dn_meta['num_dn_group']) // 2
        positive_query = query_embed[0]
        negative_query = query_embed[half_pad]
    else:
        # Fallback for configurations where DN is disabled.
        if torch.is_tensor(decoder_hs):
            positive_query = decoder_hs[-1, 0, 0]
            negative_query = decoder_hs[-1, 0, -1]
        else:
            last_layer_hs = decoder_hs[-1]
            positive_query = last_layer_hs[0, 0]
            negative_query = last_layer_hs[0, -1]
    semantic_dim = min(positive_query.shape[-1], pos.shape[-1]) // 2
    positive_query = F.normalize(positive_query[:semantic_dim].float(), dim=0)
    negative_query = F.normalize(negative_query[:semantic_dim].float(), dim=0)
    pos = F.normalize(pos[:semantic_dim].float(), dim=0)
    neg = F.normalize(neg[:semantic_dim].float(), dim=0)
    return {
        'q+_z+': float(positive_query @ pos), 'q+_z-': float(positive_query @ neg),
        'q-_z-': float(negative_query @ neg), 'q-_z+': float(negative_query @ pos),
    }


def visualize_image(cli, image_path, baseline_model, vcr_model, transform):
    image = Image.open(image_path).convert('RGB')
    boxes = read_boxes(cli, image_path)
    if not boxes:
        print('[skip] No boxes found for {}'.format(image_path.name))
        return False
    tensor, _ = transform(image, None)
    device = next(baseline_model.parameters()).device
    samples = tensor.unsqueeze(0).to(device)
    baseline = similarity(baseline_model, samples, boxes, image)
    vcr = similarity(vcr_model, samples, boxes, image)
    labels = list(baseline)
    image_output_dir = cli.output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = image_output_dir / (image_path.stem + '_vcr.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model'] + labels)
        writer.writerow(['DINO'] + [baseline[k] for k in labels])
        writer.writerow(['DINO+VCR'] + [vcr[k] for k in labels])
    x = np.arange(len(labels))
    width = .36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [baseline[k] for k in labels], width,
           label='DINO', color='#8da0cb')
    ax.bar(x + width / 2, [vcr[k] for k in labels], width,
           label='DINO+VCR', color='#fc8d62')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel('cosine similarity')
    ax.set_title('Query-prototype similarity')
    ax.legend()
    fig.tight_layout()
    figure_path = image_output_dir / (image_path.stem + '_vcr.png')
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print('[done] {} -> {}'.format(image_path.name, figure_path))
    return True


def main(cli):
    device = torch.device(cli.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(),
                           T.Normalize([.485, .456, .406], [.229, .224, .225])])
    baseline_model = load(cli, cli.baseline_checkpoint, device, False)
    vcr_model = load(cli, cli.vcr_checkpoint, device, True)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(iter_images(cli.input))
    if not image_paths:
        raise FileNotFoundError('No supported images found in {}'.format(cli.input))
    completed = 0
    for index, image_path in enumerate(image_paths, start=1):
        print('[{}/{}] Processing {}'.format(index, len(image_paths), image_path))
        completed += int(visualize_image(
            cli, image_path, baseline_model, vcr_model, transform))
    print('Finished: {}/{} images'.format(completed, len(image_paths)))


if __name__ == '__main__':
    main(parser().parse_args())
