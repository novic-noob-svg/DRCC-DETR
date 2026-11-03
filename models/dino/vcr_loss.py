import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops


class VCRLoss(nn.Module):
    """GT-constrained adaptive contrastive loss for DINO DN queries.

    Ground-truth boxes only define where positive and negative evidence is
    allowed to come from.  VCR logits still learn the relative importance of
    spatial locations inside those valid regions:

    * positive support: cells covered by at least one GT box;
    * negative support: every other valid image cell.

    The branch is training-only and does not require GT boxes at inference.
    """

    def __init__(self, temperature=0.07, eps=1e-6,
                 gt_constrained=True, use_negative_queries=True):
        super().__init__()
        if temperature <= 0:
            raise ValueError('VCR temperature must be positive')
        self.temperature = temperature
        self.eps = eps
        self.gt_constrained = bool(gt_constrained)
        self.use_negative_queries = bool(use_negative_queries)

    def _resize_logits(self, logits, feat_h, feat_w):
        if logits.shape[-2:] != (feat_h, feat_w):
            logits = F.interpolate(
                logits[None], size=(feat_h, feat_w),
                mode='bilinear', align_corners=False,
            )[0]
        return logits.float()

    def _get_valid_mask(self, outputs, batch_idx, feat_h, feat_w, device):
        valid_masks = outputs.get('vcr_valid_mask', None)
        if valid_masks is None:
            return torch.ones((feat_h, feat_w), dtype=torch.bool, device=device)

        valid = valid_masks[batch_idx]
        if valid.dim() == 3:
            valid = valid.squeeze(0)
        if valid.shape != (feat_h, feat_w):
            valid = F.interpolate(
                valid[None, None].float(), size=(feat_h, feat_w),
                mode='nearest',
            )[0, 0]
        return valid.to(device=device, dtype=torch.bool)

    def _box_rects(self, boxes, valid_h, valid_w):
        """Convert normalized cxcywh boxes to integer, half-open rectangles."""
        if boxes.numel() == 0 or valid_h <= 0 or valid_w <= 0:
            return None

        boxes = boxes.float().clamp(0, 1)
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes).clamp(0, 1)

        x1 = torch.floor(boxes_xyxy[:, 0] * valid_w).long()
        y1 = torch.floor(boxes_xyxy[:, 1] * valid_h).long()
        x2 = torch.ceil(boxes_xyxy[:, 2] * valid_w).long()
        y2 = torch.ceil(boxes_xyxy[:, 3] * valid_h).long()

        # Every valid GT receives at least one feature cell.  This matters for
        # tiny VisDrone objects whose boxes may be smaller than the VCR stride.
        x1 = x1.clamp(min=0, max=valid_w - 1)
        y1 = y1.clamp(min=0, max=valid_h - 1)
        x2 = torch.maximum(x2.clamp(min=1, max=valid_w), x1 + 1)
        y2 = torch.maximum(y2.clamp(min=1, max=valid_h), y1 + 1)
        return x1, y1, x2, y2

    def _rect_union(self, rects, feat_h, feat_w, device):
        """Rasterize a union of rectangles with a 2-D difference map."""
        support = torch.zeros((feat_h, feat_w), dtype=torch.bool, device=device)
        if rects is None:
            return support

        x1, y1, x2, y2 = rects
        if x1.numel() == 0:
            return support

        diff = torch.zeros(
            (feat_h + 1, feat_w + 1), dtype=torch.float32, device=device)
        stride = feat_w + 1
        flat = diff.view(-1)
        ones = torch.ones_like(x1, dtype=diff.dtype)

        indices = torch.cat([
            y1 * stride + x1,
            y2 * stride + x1,
            y1 * stride + x2,
            y2 * stride + x2,
        ])
        values = torch.cat([ones, -ones, -ones, ones])
        flat.scatter_add_(0, indices, values)

        coverage = diff.cumsum(dim=0).cumsum(dim=1)
        return coverage[:-1, :-1] > 0

    def _build_gt_supports(self, boxes, valid_mask):
        feat_h, feat_w = valid_mask.shape
        valid_rows = valid_mask.any(dim=1)
        valid_cols = valid_mask.any(dim=0)
        valid_h = int(valid_rows.sum().item())
        valid_w = int(valid_cols.sum().item())
        if valid_h == 0 or valid_w == 0 or boxes.numel() == 0:
            return None, None

        positive = self._rect_union(
            self._box_rects(boxes, valid_h, valid_w),
            feat_h, feat_w, boxes.device,
        ) & valid_mask
        negative = valid_mask & ~positive

        # A fully covered image has no valid background prototype.  Skip VCR
        # for that image rather than fabricating negative evidence.
        if not bool(positive.any()) or not bool(negative.any()):
            return None, None
        return positive, negative

    def _masked_spatial_softmax(self, logits, support):
        output_shape = support.shape
        flat_logits = logits.reshape(-1)
        flat_support = support.reshape(-1)
        if not bool(flat_support.any()):
            return None

        masked_logits = flat_logits.masked_fill(~flat_support, -1e4)
        weight = F.softmax(masked_logits, dim=0)
        weight = weight * flat_support.to(flat_logits.dtype)
        weight = weight / (weight.sum() + self.eps)
        return weight.view(output_shape)

    def _get_maps(self, outputs, target, batch_idx, feat_h, feat_w, device):
        if 'vcr_logits' not in outputs:
            raise KeyError('GT-constrained VCR requires outputs[\'vcr_logits\']')

        logits = self._resize_logits(
            outputs['vcr_logits'][batch_idx], feat_h, feat_w)
        valid_mask = self._get_valid_mask(
            outputs, batch_idx, feat_h, feat_w, device)
        if self.gt_constrained:
            pos_support, neg_support = self._build_gt_supports(
                target['boxes'].to(device), valid_mask)
        else:
            pos_support = valid_mask
            neg_support = valid_mask
        if pos_support is None or neg_support is None:
            return None, None

        pos = self._masked_spatial_softmax(logits[0], pos_support)
        neg = self._masked_spatial_softmax(logits[1], neg_support)
        if pos is None or neg is None:
            return None, None
        return pos.view(feat_h, feat_w), neg.view(feat_h, feat_w)

    def _pool_token(self, feat_flat, weight_map):
        weight = weight_map.reshape(-1)
        weight = weight / (weight.sum() + self.eps)
        return (feat_flat * weight.unsqueeze(-1)).sum(dim=0)

    def _nce(self, queries, pos_token, neg_token):
        if queries.numel() == 0:
            return pos_token.new_tensor(0.0)

        # Use the complete embedding for VCR contrastive learning.  The
        # decoder queries and visual prototypes are compared in all channels
        # instead of treating the first half as a predefined semantic space.
        queries_sem = F.normalize(queries.float(), dim=-1)
        pos_token_sem = F.normalize(pos_token.float().unsqueeze(0), dim=-1)
        neg_token_sem = F.normalize(neg_token.float().unsqueeze(0), dim=-1)
        pos_logit = torch.matmul(queries_sem, pos_token_sem.t()) / self.temperature
        neg_logit = torch.matmul(queries_sem, neg_token_sem.t()) / self.temperature

        logits = torch.cat([pos_logit, neg_logit], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def _zero_loss(self, outputs, device):
        zero = None
        for key in ('query_embed', 'high_res_feat_proj', 'vcr_logits'):
            value = outputs.get(key, None)
            if torch.is_tensor(value):
                cur_zero = value.sum() * 0.0
                zero = cur_zero if zero is None else zero + cur_zero
        if zero is None:
            zero = torch.tensor(0.0, device=device)
        return zero

    def forward(self, outputs, targets, dn_meta):
        required = ('query_embed', 'high_res_feat_proj', 'vcr_logits')
        if any(key not in outputs for key in required):
            device = targets[0]['boxes'].device if len(targets) > 0 else torch.device('cpu')
            return {'loss_vcr': self._zero_loss(outputs, device)}

        dn_queries = outputs['query_embed']
        # Prototype values are stop-gradient anchors.  Gradients still reach
        # VCRHead/backbone through the adaptive weights produced by vcr_logits.
        feat = outputs['high_res_feat_proj'].detach()
        bs, _, feat_h, feat_w = feat.shape
        feat_flat = feat.flatten(2).transpose(1, 2)

        num_dn_groups = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        single_pad = pad_size // num_dn_groups
        half_pad = single_pad // 2

        total_loss = feat.new_tensor(0.0)
        valid = 0
        for b in range(bs):
            num_gt = len(targets[b]['labels'])
            if num_gt == 0:
                continue

            pos_map, neg_map = self._get_maps(
                outputs, targets[b], b, feat_h, feat_w, feat.device)
            if pos_map is None or neg_map is None:
                continue
            pos_token = self._pool_token(feat_flat[b], pos_map)
            neg_token = self._pool_token(feat_flat[b], neg_map)

            pos_queries = []
            neg_queries = []
            cur_len = min(num_gt, half_pad)
            if cur_len == 0:
                continue

            for g in range(num_dn_groups):
                base = g * single_pad
                pos_queries.append(dn_queries[b, base:base + cur_len])
                neg_queries.append(
                    dn_queries[b, base + half_pad:base + half_pad + cur_len])

            pos_queries = torch.cat(pos_queries, dim=0)
            loss_pos = self._nce(pos_queries, pos_token, neg_token)
            if self.use_negative_queries:
                neg_queries = torch.cat(neg_queries, dim=0)
                loss_neg = self._nce(neg_queries, neg_token, pos_token)
                total_loss = total_loss + 0.5 * (loss_pos + loss_neg)
            else:
                total_loss = total_loss + loss_pos
            valid += 1

        if valid == 0:
            return {'loss_vcr': self._zero_loss(outputs, feat.device)}
        return {'loss_vcr': total_loss / valid}
