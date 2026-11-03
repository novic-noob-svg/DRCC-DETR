"""Keep one weights-only checkpoint selected by validation COCO bbox AP."""

import json
import math
from pathlib import Path

import torch


METRIC_NAMES = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')


class BestWeightsSaver:
    """Call on the main process only; regular and EMA share one best file."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.best_ap = -math.inf

    def update(self, model, epoch, coco_metrics, args, model_type='regular'):
        if len(coco_metrics) < 6:
            raise ValueError('COCO bbox metrics must include at least the six AP values')
        score = float(coco_metrics[0])
        # and ties never replace an existing best checkpoint.
        if not math.isfinite(score) or score < 0 or score <= self.best_ap:
            return False
        raw_metrics = [float(value) if math.isfinite(float(value)) else None
                       for value in coco_metrics]
        metadata = {
            'selection_metric': 'validation COCO bbox AP@[0.50:0.95]',
            'epoch': int(epoch), 'epoch_number': int(epoch) + 1,
            'model_type': model_type, 'best_ap_raw': score,
            'metric_units': 'percent',
            'checkpoint': 'checkpoint_best.pth',
            'coco_eval_bbox': raw_metrics,
            **{name: round(float(value) * 100, 4)
               if math.isfinite(float(value)) and float(value) >= 0 else None
               for name, value in zip(METRIC_NAMES, coco_metrics[:6])},
        }
        # No optimizer/scheduler state: this file is for evaluation or loading
        # initial model weights, rather than exact training-state resume.
        weights = {'model': model.state_dict(), 'epoch': int(epoch),
                   'args': args, 'best_ap': score, 'best_metrics': metadata,
                   'model_type': model_type}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self.output_dir / 'checkpoint_best.pth'
        checkpoint_temp = self.output_dir / 'checkpoint_best.pth.tmp'
        metrics = self.output_dir / 'best_metrics.json'
        metrics_temp = self.output_dir / 'best_metrics.json.tmp'
        try:
            metrics_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                               allow_nan=False), encoding='utf-8')
            torch.save(weights, checkpoint_temp)
            # Keep the previous best intact if serialization fails. A temporary
            # second weights file exists only while saving an improvement.
            checkpoint_temp.replace(checkpoint)
            metrics_temp.replace(metrics)
        finally:
            checkpoint_temp.unlink(missing_ok=True)
            metrics_temp.unlink(missing_ok=True)
        self.best_ap = score
        return True
