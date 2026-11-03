# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def _sample_cdn_box_noise(source_boxes, negative_mask, box_noise_scale):
    """Sample DINO positive/negative box perturbations in normalized xyxy form."""
    noised_boxes = torch.zeros_like(source_boxes)
    noised_boxes[:, :2] = source_boxes[:, :2] - source_boxes[:, 2:] / 2
    noised_boxes[:, 2:] = source_boxes[:, :2] + source_boxes[:, 2:] / 2

    diff = torch.zeros_like(source_boxes)
    diff[:, :2] = source_boxes[:, 2:] / 2
    diff[:, 2:] = source_boxes[:, 2:] / 2

    rand_sign = torch.randint_like(
        source_boxes, low=0, high=2, dtype=torch.float32
    ) * 2.0 - 1.0
    rand_part = torch.rand_like(source_boxes)
    rand_part[negative_mask] += 1.0
    rand_part *= rand_sign
    return (noised_boxes + rand_part * diff * box_noise_scale).clamp(
        min=0.0, max=1.0
    )


def _safe_pairwise_iou(boxes1, boxes2):
    """Pairwise IoU that treats crossed/degenerate xyxy boxes as zero-area."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    wh1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0)
    wh2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0)
    area1 = wh1[:, 0] * wh1[:, 1]
    area2 = wh2[:, 0] * wh2[:, 1]

    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_wh = (right_bottom - left_top).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / (union + 1e-6)


@torch.no_grad()
def _compute_cdn_collision_mask(noised_boxes_xyxy, known_bid,
                                source_indices, all_gt_boxes,
                                known_num, min_alien_iou,
                                ownership_margin):
    """Find DN boxes that are geometrically owned by a different GT instance.

    A collision is counted only when a different GT overlaps the noised box
    more than its source GT and the alien IoU reaches ``min_alien_iou``.
    ``source_indices`` indexes the concatenated, unexpanded GT tensor.
    """
    collision_mask = torch.zeros(
        noised_boxes_xyxy.shape[0], dtype=torch.bool,
        device=noised_boxes_xyxy.device
    )
    source_iou = noised_boxes_xyxy.new_zeros(noised_boxes_xyxy.shape[0])
    alien_iou = noised_boxes_xyxy.new_zeros(noised_boxes_xyxy.shape[0])

    gt_offset = 0
    for image_index, gt_count in enumerate(known_num):
        gt_count = int(gt_count)
        if gt_count <= 0:
            continue
        row_indices = torch.nonzero(
            known_bid == image_index, as_tuple=False
        ).flatten()
        if row_indices.numel() == 0:
            gt_offset += gt_count
            continue

        image_gt = all_gt_boxes[gt_offset:gt_offset + gt_count]
        image_gt_xyxy = torch.cat(
            (image_gt[:, :2] - image_gt[:, 2:] / 2,
             image_gt[:, :2] + image_gt[:, 2:] / 2),
            dim=1,
        )
        local_source_indices = source_indices[row_indices] - gt_offset
        ious = _safe_pairwise_iou(
            noised_boxes_xyxy[row_indices], image_gt_xyxy
        )
        local_rows = torch.arange(row_indices.numel(), device=ious.device)
        current_source_iou = ious[local_rows, local_source_indices]
        source_iou[row_indices] = current_source_iou

        if gt_count > 1:
            ious[local_rows, local_source_indices] = -1.0
            current_alien_iou = ious.max(dim=1).values.clamp(min=0.0)
            alien_iou[row_indices] = current_alien_iou
            collision_mask[row_indices] = (
                (current_alien_iou > current_source_iou + ownership_margin)
                & (current_alien_iou >= min_alien_iou)
            )
        gt_offset += gt_count

    return collision_mask, source_iou, alien_iou


@torch.no_grad()
def _apply_neighbor_safe_cdn(noised_boxes_xyxy, expanded_source_boxes,
                             all_gt_boxes, negative_mask, known_bid,
                             source_indices, known_num, box_noise_scale, max_trials,
                             min_alien_iou, ownership_margin):
    """Resample only collided DN boxes while keeping the query layout fixed."""
    before_collision, _, _ = _compute_cdn_collision_mask(
        noised_boxes_xyxy, known_bid, source_indices, all_gt_boxes,
        known_num, min_alien_iou, ownership_margin
    )
    current_collision = before_collision.clone()
    total_draws = int(noised_boxes_xyxy.shape[0])

    # max_trials includes the initial DINO sample already present above.
    for _ in range(1, max_trials):
        retry_indices = torch.nonzero(
            current_collision, as_tuple=False
        ).flatten()
        if retry_indices.numel() == 0:
            break
        noised_boxes_xyxy[retry_indices] = _sample_cdn_box_noise(
            expanded_source_boxes[retry_indices],
            negative_mask[retry_indices],
            box_noise_scale,
        )
        total_draws += int(retry_indices.numel())
        current_collision, _, _ = _compute_cdn_collision_mask(
            noised_boxes_xyxy, known_bid, source_indices, all_gt_boxes,
            known_num, min_alien_iou, ownership_margin
        )

    positive_mask = ~negative_mask

    def _rate(mask, query_mask):
        denominator = query_mask.sum().clamp(min=1).to(torch.float32)
        return (mask & query_mask).sum().to(torch.float32) / denominator

    query_count = max(1, int(noised_boxes_xyxy.shape[0]))
    stats = {
        'pos_collision_before': _rate(before_collision, positive_mask),
        'pos_collision_after': _rate(current_collision, positive_mask),
        'neg_collision_before': _rate(before_collision, negative_mask),
        'neg_collision_after': _rate(current_collision, negative_mask),
        'avg_trials': noised_boxes_xyxy.new_tensor(
            float(total_draws) / float(query_count)
        ),
    }
    return noised_boxes_xyxy, stats


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim,
                    label_enc, use_ns_cdn=False, dn_collision_trials=3,
                    dn_collision_iou_threshold=0.1,
                    dn_collision_ownership_margin=0.0):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :param use_ns_cdn: enable neighbour-safe contrastive denoising
        :param dn_collision_trials: maximum samples per final DN query,
            including the original sample
        :param dn_collision_iou_threshold: minimum alien IoU that is treated
            as a substantive collision
        :param dn_collision_ownership_margin: extra alien-vs-source IoU margin
        :return:
        """
    if training:
        if int(dn_collision_trials) < 1:
            raise ValueError('dn_collision_trials must be at least 1')
        if not 0.0 <= float(dn_collision_iou_threshold) <= 1.0:
            raise ValueError('dn_collision_iou_threshold must be in [0, 1]')
        if float(dn_collision_ownership_margin) < 0.0:
            raise ValueError('dn_collision_ownership_margin must be non-negative')
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        device = label_enc.weight.device
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [torch.ones_like(t['labels'], device=device) for t in targets]
        batch_size = len(known)
        known_num = [len(t['labels']) for t in targets]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        source_indices = torch.arange(
            len(boxes), device=device
        ).repeat(2 * dn_number)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.arange(len(boxes), device=device).long().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.arange(dn_number, device=device) * len(boxes) * 2).long().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        negative_mask = torch.zeros(
            known_bboxs.shape[0], dtype=torch.bool, device=device
        )
        negative_mask[negative_idx] = True
        ns_cdn_stats = None
        if box_noise_scale > 0:
            known_bbox_ = _sample_cdn_box_noise(
                known_bboxs, negative_mask, box_noise_scale
            )
            if use_ns_cdn and known_bbox_.shape[0] > 0:
                known_bbox_, ns_cdn_stats = _apply_neighbor_safe_cdn(
                    noised_boxes_xyxy=known_bbox_,
                    expanded_source_boxes=known_bboxs,
                    all_gt_boxes=boxes,
                    negative_mask=negative_mask,
                    known_bid=known_bid,
                    source_indices=source_indices,
                    known_num=known_num,
                    box_noise_scale=box_noise_scale,
                    max_trials=int(dn_collision_trials),
                    min_alien_iou=float(dn_collision_iou_threshold),
                    ownership_margin=float(dn_collision_ownership_margin),
                )
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to(device)
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim, device=device)
        padding_bbox = torch.zeros(pad_size, 4, device=device)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.empty(0, device=device)
        if len(known_num):
            map_known_indice = torch.cat([torch.arange(num, device=device) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.zeros(tgt_size, tgt_size, dtype=torch.bool, device=device)
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
        if use_ns_cdn and ns_cdn_stats is None:
            zero = known_bboxs.new_tensor(0.0)
            ns_cdn_stats = {
                'pos_collision_before': zero,
                'pos_collision_after': zero,
                'neg_collision_before': zero,
                'neg_collision_after': zero,
                'avg_trials': known_bboxs.new_tensor(
                    1.0 if known_bboxs.shape[0] > 0 else 0.0
                ),
            }
        if ns_cdn_stats is not None:
            dn_meta['ns_cdn_stats'] = ns_cdn_stats
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss, hs=None):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        output_known_hs = None
        if hs is not None:
            if isinstance(hs, (list, tuple)):
                output_known_hs = [layer_hs[:, :dn_meta['pad_size'], :] for layer_hs in hs]
            else:
                output_known_hs = hs[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if output_known_hs is not None:
            out['query_embed'] = output_known_hs[-1]
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord, output_known_hs)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord
