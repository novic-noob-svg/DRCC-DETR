#!/usr/bin/env python3
"""Run or preview the seven SADRA sensitivity configurations in the paper."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = [(0.05, 2.5), (0.10, 2.5), (0.15, 2.5), (0.20, 2.5),
            (0.10, 1.5), (0.10, 2.0), (0.10, 3.0)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-file', type=Path, required=True,
                   help='Actual sensitivity-run config exported from its saved run metadata')
    p.add_argument('--coco-path', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=ROOT / 'outputs/sadra_sensitivity')
    p.add_argument('--nproc-per-node', type=int, default=1)
    p.add_argument('--cuda-devices')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.nproc_per_node < 1 or not a.config_file.is_file():
        p.error('Positive process count and an existing config are required.')
    env = os.environ.copy()
    if a.cuda_devices:
        env['CUDA_VISIBLE_DEVICES'] = a.cuda_devices
    for threshold, maximum in SETTINGS:
        output = a.output_root.resolve() / ('tau_%.2f_alpha_%.1f' % (threshold, maximum))
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node', str(a.nproc_per_node), str(ROOT / 'main.py'),
            '-c', str(a.config_file.resolve()), '--coco_path', str(a.coco_path.resolve()),
            '--output_dir', str(output), '--seed', str(a.seed), '--no_use_vcr', '--use_sadra',
            '--sadra_scale_threshold', str(threshold), '--sadra_max_scale', str(maximum)]
        print(subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command), flush=True)
        if not a.dry_run:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError('Choose an empty output root; refusing to mix runs: ' + str(output))
            subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
