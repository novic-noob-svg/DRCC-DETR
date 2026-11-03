"""Summarize the best epoch of each experiment without loading its weights."""

import csv
import json
from pathlib import Path


def write_summary(output_root, experiments, seeds):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fields = ['experiment', 'reference', 'seed', 'status', 'best_epoch',
              'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
              'model_type', 'checkpoint']
    path = output_root / 'ablation_summary.csv'
    temp = output_root / 'ablation_summary.csv.tmp'
    with temp.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment in experiments:
            for seed in seeds:
                directory = output_root / experiment['name'] / ('seed_{}'.format(seed))
                row = {'experiment': experiment['name'],
                       'reference': experiment['reference'] or '',
                       'seed': seed, 'status': 'pending'}
                record_path = directory / 'ablation_run.json'
                if record_path.is_file():
                    record = json.loads(record_path.read_text(encoding='utf-8'))
                    row['status'] = record.get('status', 'unknown')
                metrics_path = directory / 'best_metrics.json'
                if metrics_path.is_file():
                    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                    row.update(best_epoch=metrics['epoch_number'],
                               model_type=metrics['model_type'],
                               checkpoint=str(directory / metrics['checkpoint']))
                    row.update({key: metrics.get(key) for key in fields if key.startswith('AP')})
                writer.writerow(row)
    temp.replace(path)
    return path
