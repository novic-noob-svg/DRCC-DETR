#!/usr/bin/env python3
"""Visualize final-decoder sampling geometry with GT-matched prediction queries.

Without a baseline checkpoint the left panel is pre-SADRA geometry from the
same forward pass, not an independently trained DINO or a full ablation.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import datasets.transforms as T
from util.slconfig import DictAction
from util import misc
from main import build_model_main
from util.checkpoint import migrate_state_dict
from inference_visualize import load_args, resolve_checkpoint, iter_images


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--run-config', type=Path)
    p.add_argument('--config-file', type=Path)
    p.add_argument('--weights-key', default='model')
    p.add_argument('--options', nargs='+', action=DictAction)
    p.add_argument('--baseline-checkpoint', type=Path)
    p.add_argument('--baseline-run-config', type=Path)
    p.add_argument('--baseline-config-file', type=Path)
    p.add_argument('--baseline-weights-key', default='model')
    p.add_argument('--baseline-options', nargs='+', action=DictAction)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--coco-annotations', type=Path)
    p.add_argument('--boxes', type=Path, help='Legacy JSON: image name -> xyxy boxes (class-agnostic matching)')
    p.add_argument('--category-map', type=Path, help='JSON mapping COCO category IDs to model label IDs')
    p.add_argument('--output-dir', type=Path, default=Path('visualizations/sadra'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--attention-head', type=int, default=0)
    p.add_argument('--feature-level', type=int, default=0)
    p.add_argument('--max-targets', type=int, default=6)
    p.add_argument('--score-threshold', type=float, default=0.3)
    p.add_argument('--match-iou', type=float, default=0.3)
    p.add_argument('--size-mode', choices=['coco', 'relative'], default='coco')
    p.add_argument('--crop-scale', type=float, default=6.0)
    p.add_argument('--min-crop-size', type=float, default=100.0)
    p.add_argument('--point-size', type=float, default=45.0)
    p.add_argument('--point-alpha', type=float, default=0.85)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--save-pdf', action='store_true')
    return p


def box_iou(boxes, target):
    lo = np.maximum(boxes[:, :2], target[:2])
    hi = np.minimum(boxes[:, 2:], target[2:])
    intersection = np.prod(np.maximum(hi - lo, 0), axis=1)
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    target_area = np.prod(np.maximum(target[2:] - target[:2], 0))
    return intersection / np.maximum(area + target_area - intersection, 1e-12)


def restore_image_locations(locations, ratios):
    """[Q,H,L,P,2] padded feature coordinates -> normalized image coordinates."""
    ratios = np.asarray(ratios)
    if ratios.shape != (locations.shape[2], 2) or not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError('Invalid feature valid ratios')
    return locations / ratios[None,None,:,None,:]


def match_query(predictions, target, score_threshold, min_iou, label=None):
    probabilities = predictions['probabilities']
    classes = probabilities.argmax(axis=1)
    scores = probabilities.max(axis=1)
    eligible = scores >= score_threshold
    if label is not None:
        if not 0 <= label < probabilities.shape[1]:
            raise ValueError('GT label outside model class range; check --category-map and training config')
        eligible &= classes == label
    overlaps = box_iou(predictions['boxes'], np.asarray(target, dtype=float))
    indices = np.flatnonzero(eligible & (overlaps >= min_iou))
    if not len(indices):
        return None
    query = int(max(indices, key=lambda q: (overlaps[q], scores[q])))
    return dict(query=query, score=float(scores[query]), iou=float(overlaps[query]),
                label=int(classes[query]), box=predictions['boxes'][query].tolist())


def size_category(box, width, height, mode):
    area = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
    if mode == 'relative':
        return 'small' if area/(width*height) < .02 else 'medium' if area/(width*height) < .15 else 'large'
    return 'small' if area < 32**2 else 'medium' if area < 96**2 else 'large'


def select_targets(targets, width, height, mode, limit):
    buckets = {key: [] for key in ('small', 'medium', 'large')}
    for target in targets:
        target = dict(target)
        target['size'] = size_category(target['box'], width, height, mode)
        buckets[target['size']].append(target)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
    return selected


def load_model(cli, baseline=False):
    checkpoint = resolve_checkpoint(cli.baseline_checkpoint if baseline else cli.checkpoint)
    config_cli = argparse.Namespace(
        run_config=cli.baseline_run_config if baseline else cli.run_config,
        config_file=cli.baseline_config_file if baseline else cli.config_file,
        options=cli.baseline_options if baseline else cli.options,
        use_vcr=None, use_sadra=False if baseline else True, device=cli.device)
    args = load_args(config_cli, checkpoint)
    if baseline and getattr(args, 'use_vcr', False):
        raise ValueError('Baseline config enables VCR/VCR; provide a plain DINO baseline config')
    model, _, _ = build_model_main(args)
    try:
        data = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    except TypeError:
        data = torch.load(str(checkpoint), map_location='cpu')
    key = cli.baseline_weights_key if baseline else cli.weights_key
    if isinstance(data, dict) and key in data:
        state = data[key]
    elif isinstance(data, dict) and data and all(torch.is_tensor(v) for v in data.values()) and key == 'model':
        state = data
    else:
        raise KeyError('Checkpoint has no requested weights key: ' + key)
    state = migrate_state_dict(state)
    # Fail instead of silently visualizing randomly initialized attention parameters.
    model.load_state_dict(state, strict=True)
    model.to(cli.device).eval()
    module = model.transformer.decoder.layers[-1].cross_attn
    module.capture_sampling = True
    if not hasattr(module, 'last_baseline_sampling_locations') and not hasattr(module, '_apply_sadra'):
        raise ValueError('Final decoder cross-attention does not support diagnostic sampling')
    config = {k: getattr(args, k, None) for k in (
        'use_sadra', 'use_vcr', 'num_classes', 'num_feature_levels', 'dec_n_points',
        'sadra_max_scale', 'sadra_scale_threshold', 'sadra_context_points',
        'sadra_use_boundary_band', 'sadra_boundary_band_width',
        'sadra_use_scale_expansion', 'sadra_use_region_prior')}
    config.update(checkpoint=str(checkpoint), weights_key=key)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return model, module, config


def capture(model, module, samples, width, height):
    record = {}
    def hook(attn, inputs, output):
        query, refs, _, shapes, starts = inputs[:5]
        mask = inputs[5] if len(inputs) > 5 else None
        ratios = []
        for level, (h, w) in enumerate(shapes.tolist()):
            if mask is None:
                ratios.append([1., 1.])
            else:
                level_mask = mask[0, int(starts[level]):int(starts[level])+h*w].reshape(h,w)
                ratios.append([float((~level_mask[0]).sum())/w,
                               float((~level_mask[:,0]).sum())/h])
        ratios = np.asarray(ratios)
        if (ratios <= 0).any():
            raise ValueError('Invalid feature valid ratios')
        before = attn.last_baseline_sampling_locations
        after = attn.last_sampling_locations if attn.use_sadra else before
        record['before'] = restore_image_locations(before[0].float().cpu().numpy(), ratios)
        record['after'] = restore_image_locations(after[0].float().cpu().numpy(), ratios)
        offsets = attn.sampling_offsets(query).view(1,query.shape[1],attn.n_heads,attn.n_levels,attn.n_points,2)
        logits = attn.attention_weights(query).view(1,query.shape[1],attn.n_heads,attn.n_levels,attn.n_points)
        if attn.use_sadra:
            _, logits = attn._apply_sadra(refs, offsets, logits)
        weights = logits.flatten(-2).softmax(-1).view_as(logits)
        record['weights'] = weights[0].float().cpu().numpy()
        record['references'] = refs[0].float().cpu().numpy()
        record['valid_ratios'] = ratios
        record['local_points'] = attn.n_points-attn.sadra_context_points if attn.use_sadra else attn.n_points
    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            outputs = model(samples)
    finally:
        handle.remove()
    boxes = outputs['pred_boxes'][0].float().cpu().numpy()
    xyxy = np.concatenate((boxes[:,:2]-boxes[:,2:]/2, boxes[:,:2]+boxes[:,2:]/2),axis=1)
    record['predictions'] = dict(boxes=xyxy*np.array([width,height,width,height]),
                                probabilities=outputs['pred_logits'][0].sigmoid().float().cpu().numpy())
    if len(xyxy) != record['after'].shape[0]:
        raise ValueError('Output query count differs from decoder cache; cannot safely match queries')
    return record


def annotations(cli):
    if cli.coco_annotations:
        data=json.loads(cli.coco_annotations.read_text(encoding='utf-8-sig'))
        by_id={}
        for ann in data['annotations']:
            x,y,w,h=ann['bbox']
            if ann.get('iscrowd',0) or ann.get('ignore',0) or w<=0 or h<=0:
                continue
            by_id.setdefault(ann['image_id'],[]).append(dict(box=[x,y,x+w,y+h],
                category_id=ann['category_id'], annotation_id=ann['id']))
        by_name={}
        for image in data['images']:
            name=Path(image['file_name']).name
            if name in by_name:
                raise ValueError('Duplicate annotation image basenames: '+name)
            by_name[name]=by_id.get(image['id'],[])
        return by_name
    if cli.boxes:
        data=json.loads(cli.boxes.read_text(encoding='utf-8-sig'))
        return {name:[dict(box=box,category_id=None) for box in boxes] for name,boxes in data.items()}
    raise ValueError('Pass --coco-annotations or --boxes')


def draw(ax, image, target, record, match, cli, kind, title, crop):
    ax.imshow(image)
    box=target['box']
    ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                           fill=False,edgecolor='gold',linewidth=1.8,label='GT'))
    if match is None:
        ax.text(.02,.97,'No class/score/IoU-matched prediction',transform=ax.transAxes,
                va='top',fontsize=8,color='red',bbox=dict(facecolor='white',alpha=.8))
    else:
        q=match['query']; h=cli.attention_head; level=cli.feature_level
        xy=record[kind][q,h,level]*np.array([image.width,image.height])
        weights=record['weights'][q,h,level]
        nlocal=record['local_points']
        groups=[('Local',slice(0,nlocal),'#00b4d8'),('Context',slice(nlocal,None),'#ff6b35')]
        for label, sl, color in groups:
            points=xy[sl]; ww=weights[sl]
            valid=np.isfinite(points).all(axis=1)
            if valid.any():
                # Relative weight within this selected head/level, for visual emphasis.
                sizes=cli.point_size*(.4+1.6*ww[valid]/max(float(weights.max()),1e-12))
                ax.scatter(*points[valid].T,s=sizes,c=color,alpha=cli.point_alpha,
                           edgecolors='white',linewidths=.4,label=label)
        pred=match['box']
        ax.add_patch(Rectangle((pred[0],pred[1]),pred[2]-pred[0],pred[3]-pred[1],
                               fill=False,edgecolor='#cf4cde',linestyle='--',linewidth=1,label='Prediction'))
        title += ' | q={} score={:.2f} IoU={:.2f}'.format(q,match['score'],match['iou'])
    ax.set_title(title,fontsize=9)
    ax.set_xlim(crop[0],crop[2]); ax.set_ylim(crop[3],crop[1]); ax.axis('off')
    ax.legend(loc='lower right',fontsize=7)


def main(cli):
    logging.basicConfig(level=logging.INFO)
    if not 0<=cli.score_threshold<=1 or not 0<=cli.match_iou<=1:
        raise ValueError('Score and IoU thresholds must be in [0,1]')
    if cli.max_targets<1 or cli.crop_scale<1 or cli.min_crop_size<=0:
        raise ValueError('Invalid target count or crop size')
    targets_by_name=annotations(cli)
    label_map=json.loads(cli.category_map.read_text()) if cli.category_map else {}
    model,module,config=load_model(cli)
    baseline=load_model(cli,True) if cli.baseline_checkpoint else None
    cli.output_dir.mkdir(parents=True,exist_ok=True)
    mode='independent_baseline' if baseline else 'same_forward_pre_post_geometry'
    print('Comparison mode:',mode)
    (cli.output_dir/'resolved_visualization_config.json').write_text(json.dumps(dict(
        mode=mode,sadra=config,baseline=baseline[2] if baseline else None,
        options={k:str(v) if isinstance(v,Path) else v for k,v in vars(cli).items()}),
        indent=2,ensure_ascii=False),encoding='utf-8')
    transform=T.Compose([T.RandomResize([800],max_size=1333),T.ToTensor(),
                         T.Normalize([.485,.456,.406],[.229,.224,.225])])
    paths=list(iter_images(cli.input))
    if not paths: raise FileNotFoundError('No input images')
    for image_path in paths:
        image=Image.open(image_path).convert('RGB')
        if image_path.name not in targets_by_name:
            raise ValueError('Image missing from annotations: '+image_path.name)
        targets=select_targets(targets_by_name[image_path.name],image.width,image.height,cli.size_mode,cli.max_targets)
        if not targets:
            print('[skip] No valid GT:',image_path); continue
        tensor,_=transform(image,None)
        samples=tensor.unsqueeze(0).to(cli.device)
        sadra=capture(model,module,samples,image.width,image.height)
        other=capture(baseline[0],baseline[1],samples,image.width,image.height) if baseline else sadra
        for rec in (sadra,other):
            if not 0<=cli.attention_head<rec['after'].shape[1]: raise ValueError('Invalid attention head')
            level=cli.feature_level
            if not -rec['after'].shape[2]<=level<rec['after'].shape[2]: raise ValueError('Invalid feature level')
        fig,axes=plt.subplots(len(targets),2,figsize=(12,4*len(targets)),squeeze=False)
        info=[]; arrays={}
        for i,target in enumerate(targets):
            category=target['category_id']
            label=int(label_map.get(str(category),category)) if category is not None else None
            sm=match_query(sadra['predictions'],target['box'],cli.score_threshold,cli.match_iou,label)
            bm=match_query(other['predictions'],target['box'],cli.score_threshold,cli.match_iou,label) if baseline else sm
            x1,y1,x2,y2=target['box']; cx=(x1+x2)/2; cy=(y1+y2)/2
            w=max((x2-x1)*cli.crop_scale,cli.min_crop_size); h=max((y2-y1)*cli.crop_scale,cli.min_crop_size)
            crop=[max(0,cx-w/2),max(0,cy-h/2),min(image.width,cx+w/2),min(image.height,cy+h/2)]
            left='DINO baseline' if baseline else 'Before SADRA geometry (same query)'
            draw(axes[i,0],image,target,other,bm,cli,'after' if baseline else 'before',left+' | '+target['size'],crop)
            right = 'VCR + SADRA' if config['use_vcr'] else 'SADRA'
            draw(axes[i,1],image,target,sadra,sm,cli,'after',right+' | '+target['size'],crop)
            info.append(dict(target=target,sadra_match=sm,baseline_match=bm if baseline else None,crop=crop))
            for name,rec,match in [('sadra',sadra,sm),('baseline',other,bm)]:
                if match is not None:
                    q=match['query']
                    for key in ('before','after','weights','references'):
                        arrays['target{}_{}_{}'.format(i,name,key)]=rec[key][q]
                    arrays['target{}_{}_valid_ratios'.format(i,name)]=rec['valid_ratios']
        fig.suptitle('{} | final decoder | head {} | level {}\nPoints sized by final attention weights; gold: GT; dashed: prediction'.format(
            image_path.name,cli.attention_head,cli.feature_level),fontsize=11)
        fig.tight_layout(rect=[0,0,1,.96])
        relative=image_path.relative_to(cli.input) if cli.input.is_dir() else Path(image_path.name)
        folder=cli.output_dir/relative.parent/relative.stem
        folder.mkdir(parents=True,exist_ok=True)
        fig.savefig(folder/'sampling.png',dpi=cli.dpi,bbox_inches='tight')
        if cli.save_pdf: fig.savefig(folder/'sampling.pdf',bbox_inches='tight')
        plt.close(fig)
        np.savez_compressed(folder/'sampling.npz',**arrays)
        (folder/'matches.json').write_text(json.dumps(dict(image=str(image_path),mode=mode,
            size_mode=cli.size_mode,targets=info),indent=2,ensure_ascii=False),encoding='utf-8')
        print('[done]',folder,'matched SADRA:',sum(t['sadra_match'] is not None for t in info),'/',len(info))


if __name__=='__main__':
    main(parser().parse_args())
