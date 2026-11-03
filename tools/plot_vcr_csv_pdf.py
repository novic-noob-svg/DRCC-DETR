#!/usr/bin/env python3
"""Create LaTeX-ready vector PDF figures from VCR similarity CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = ['q+_z+', 'q+_z-', 'q-_z-', 'q-_z+']


def parser():
    p = argparse.ArgumentParser(description='Plot VCR CSV results as vector PDF.')
    p.add_argument('--input', type=Path, required=True,
                   help='A VCR CSV file or a folder containing *_vcr.csv files.')
    p.add_argument('--output', type=Path, default=None,
                   help='Output PDF path. Required for a single CSV; optional for a folder.')
    p.add_argument('--aggregate', action='store_true',
                   help='For a folder, plot mean and standard deviation across CSV files.')
    p.add_argument('--no-baseline', action='store_true',
                   help='Plot only the DINO+VCR row.')
    p.add_argument('--width', type=float, default=7.2,
                   help='Figure width in inches.')
    p.add_argument('--height', type=float, default=4.3,
                   help='Figure height in inches.')
    return p


def configure_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [label for label in LABELS if label not in columns]
        if 'model' not in columns or missing:
            raise ValueError('{} must contain model and {}'.format(path, ', '.join(missing)))
        data = list(reader)
    return data


def plot_single(data, output, width, height, no_baseline):
    configure_style()
    labels = LABELS
    if no_baseline:
        data = [row for row in data if row['model'] == 'DINO+VCR']
    if not data:
        raise ValueError('No rows selected for plotting.')

    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    x = np.arange(len(labels))
    rows = data
    bar_width = min(0.72 / len(rows), 0.36)
    colors = ['#8da0cb', '#ef8354', '#2a9d8f', '#e9c46a']
    for row_index, row in enumerate(rows):
        values = [float(row[label]) for label in labels]
        offset = (row_index - (len(rows) - 1) / 2) * bar_width
        ax.bar(x + offset, values, bar_width * .92,
               label=row['model'], color=colors[row_index % len(colors)],
               edgecolor='white', linewidth=.4)

    ax.axhline(0, color='#333333', linewidth=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Query-prototype similarity')
    ax.grid(axis='y', color='#d9d9d9', linewidth=.6, alpha=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, loc='best')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format='pdf', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    print('Saved:', output)


def plot_aggregate(paths, output, width, height, no_baseline):
    tables = [read_csv(path) for path in paths]
    rows = []
    for model in ['DINO', 'DINO+VCR']:
        if no_baseline and model == 'DINO':
            continue
        values = np.asarray([
            [float(row[label]) for label in LABELS]
            for table in tables
            for row in table if row['model'] == model
        ])
        if len(values):
            rows.append((model, values.mean(axis=0), values.std(axis=0)))
    if not rows:
        raise ValueError('No compatible model rows found in CSV files.')

    configure_style()
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    x = np.arange(len(LABELS))
    bar_width = min(.72 / len(rows), .36)
    colors = ['#8da0cb', '#ef8354']
    for row_index, (model, mean, std) in enumerate(rows):
        offset = (row_index - (len(rows) - 1) / 2) * bar_width
        error_args = {}
        if len(paths) > 1:
            error_args = {
                'yerr': std,
                'capsize': 3,
                'error_kw': {'elinewidth': .8},
            }
        ax.bar(x + offset, mean, bar_width * .92,
               label=model, color=colors[row_index], edgecolor='white',
               linewidth=.4, **error_args)
    ax.axhline(0, color='#333333', linewidth=.8)
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Query-prototype similarity (mean $\\pm$ std)')
    ax.grid(axis='y', color='#d9d9d9', linewidth=.6, alpha=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, loc='best')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format='pdf', bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    print('Saved:', output)


def main(cli):
    path = cli.input.expanduser().resolve()
    if path.is_file():
        output = cli.output or path.with_suffix('.pdf')
        plot_single(read_csv(path), output, cli.width, cli.height, cli.no_baseline)
        return
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = sorted(path.rglob('*_vcr.csv'))
    if not paths:
        raise FileNotFoundError('No *_vcr.csv files found in {}'.format(path))
    output = cli.output or path / ('vcr_aggregate.pdf' if cli.aggregate else 'vcr_all.pdf')
    if cli.aggregate:
        plot_aggregate(paths, output, cli.width, cli.height, cli.no_baseline)
    else:
        output.mkdir(parents=True, exist_ok=True)
        for csv_path in paths:
            plot_single(read_csv(csv_path), output / (csv_path.stem + '.pdf'),
                        cli.width, cli.height, cli.no_baseline)


if __name__ == '__main__':
    main(parser().parse_args())
