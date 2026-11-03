"""List, preview or sequentially train the first five simple ablations.

This launcher uses only the standard library. Run from any working directory;
training runs in the project root. --dry-run never starts training or writes.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from ablation_results import write_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / 'config' / 'component_ablations.json'
# These are controlled by the suite, not by general config overrides.
CONTROLLED_OPTIONS = {
    'dec_n_points', 'use_ns_cdn', 'use_vcr', 'use_vcr', 'use_sadra',
    'vcr_enhancement_mode', 'vcr_loss_coef', 'vcr_temperature',
    'vcr_gt_constrained', 'vcr_use_negative_queries', 'sadra_max_scale',
    'sadra_scale_threshold', 'sadra_context_points',
    'sadra_use_scale_expansion', 'sadra_use_region_prior',
    'sadra_use_boundary_band', 'sadra_boundary_band_width',
    'seed', 'resume', 'output_dir', 'pretrain_model_path', 'coco_path', 'save_best_only',
}


def load_manifest():
    with MANIFEST_PATH.open(encoding='utf-8') as handle:
        return json.load(handle)


def experiment_args(manifest, experiment):
    """Replace suite defaults rather than emit conflicting boolean flags."""
    tokens = manifest['common_args']
    overrides = experiment.get('overrides', {})
    result = []
    index = 0
    while index < len(tokens):
        flag = tokens[index]
        end = index + 1
        while end < len(tokens) and not tokens[end].startswith('--'):
            end += 1
        result.extend(overrides.get(flag, tokens[index:end]))
        index = end
    return result + experiment['args']


def make_parser(manifest):
    names = [entry['name'] for entry in manifest['experiments']]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List controls and references')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without training')
    parser.add_argument('--experiments', nargs='+', choices=names + ['all'],
                        default=manifest['default_experiments'],
                        help='Default: six new ablations only. all also reruns the three reference experiments.')
    parser.add_argument('--coco-path', help='Same COCO dataset root for every run')
    parser.add_argument('--output-root', type=Path,
                        default=PROJECT_ROOT / 'outputs' / 'component_ablations')
    parser.add_argument('--config-file', type=Path,
                        default=PROJECT_ROOT / 'config' / 'DINO' / 'DINO_4scale.py')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--nproc-per-node', type=int, default=1)
    parser.add_argument('--master-port', type=int, default=29500,
                        help='Local static rendezvous port; master address is fixed to 127.0.0.1')
    parser.add_argument('--cuda-devices', help='e.g. 0 or 0,1 (sets CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--pretrain-model-path', type=Path,
                        help='Common initial checkpoint; never a per-control resume checkpoint')
    parser.add_argument('--vcr-loss-coef', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--use-tb', action='store_true')
    parser.add_argument('--options', nargs='+', default=[], metavar='KEY=VALUE',
                        help='Shared config overrides, e.g. batch_size=1 epochs=24 lr_drop=11')
    return parser


def validate_args(parser, args):
    if args.list:
        return
    if not args.coco_path:
        parser.error('--coco-path is required for preview or training')
    if args.nproc_per_node < 1 or args.num_workers < 0:
        parser.error('nproc-per-node must be positive; num-workers must be nonnegative')
    if not 1 <= args.master_port <= 65535:
        parser.error('--master-port must be in [1, 65535]')
    if not args.vcr_loss_coef > 0:
        parser.error('--vcr-loss-coef must be positive for the VCR comparisons')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('--seeds must not contain duplicates')
    for item in args.options:
        key, separator, value = item.partition('=')
        if not separator or not key or not value:
            parser.error('--options entries must be KEY=VALUE')
        if key in CONTROLLED_OPTIONS:
            parser.error('{} is controlled by the experiment suite'.format(key))
    if not args.config_file.is_file():
        parser.error('config file does not exist: {}'.format(args.config_file))
    if not args.dry_run:
        if not Path(args.coco_path).is_dir():
            parser.error('dataset directory does not exist: {}'.format(args.coco_path))
        if args.pretrain_model_path and not args.pretrain_model_path.is_file():
            parser.error('initial checkpoint does not exist: {}'.format(args.pretrain_model_path))


def select_experiments(manifest, names):
    if 'all' in names:
        return manifest['experiments']
    lookup = {entry['name']: entry for entry in manifest['experiments']}
    # Retain requested order while avoiding a duplicate run.
    return [lookup[name] for name in dict.fromkeys(names)]


def build_command(args, manifest, experiment, seed):
    output = args.output_root.resolve() / experiment['name'] / ('seed_{}'.format(seed))
    # standalone/c10d may advertise socket.getfqdn() to worker processes.
    # A server with an unresolvable hostname then times out even on one node.
    # Static rendezvous passes the literal loopback IP through to all workers.
    command = [sys.executable, '-m', 'torch.distributed.run',
               '--nnodes', '1', '--node_rank', '0', '--rdzv_backend', 'static',
               '--master_addr', '127.0.0.1', '--master_port', str(args.master_port),
               '--nproc_per_node', str(args.nproc_per_node),
               str(PROJECT_ROOT / 'main.py'),
               '-c', str(args.config_file.resolve()),
               '--coco_path', str(Path(args.coco_path).resolve()),
               '--output_dir', str(output), '--seed', str(seed),
               '--num_workers', str(args.num_workers),
               '--vcr_loss_coef', str(args.vcr_loss_coef), '--save_best_only']
    command += experiment_args(manifest, experiment)
    if not args.no_amp:
        command.append('--amp')
    if args.use_tb:
        command.append('--use_tb')
    if args.pretrain_model_path:
        command += ['--pretrain_model_path', str(args.pretrain_model_path.resolve())]
    command += ['--options'] + manifest['common_options'] + args.options
    return command, output


def format_command(command):
    return subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)


def can_retry_startup_failure(output):
    """Allow a recorded failed start, while protecting trained or active runs."""
    output = Path(output)
    record_path = output / 'ablation_run.json'
    if not output.is_dir() or not record_path.is_file():
        return False
    try:
        record = json.loads(record_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or record.get('status') != 'failed':
        return False
    if any(output.rglob('*.pth')) or (output / 'best_metrics.json').exists():
        return False
    training_log = output / 'log.txt'
    return not training_log.exists() or training_log.stat().st_size == 0


def main():
    manifest = load_manifest()
    parser = make_parser(manifest)
    args = parser.parse_args()
    validate_args(parser, args)
    if args.list:
        for entry in manifest['experiments']:
            print('{}: {} (reference: {})'.format(
                entry['name'], entry['description'], entry['reference'] or '-'))
        return 0

    selected = select_experiments(manifest, args.experiments)
    jobs = [(entry, seed, *build_command(args, manifest, entry, seed))
            for entry in selected for seed in args.seeds]
    if not args.dry_run:
        # Preflight all output directories before the first expensive run.
        # main.py otherwise automatically resumes an existing checkpoint.
        for _, _, _, output in jobs:
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                if can_retry_startup_failure(output):
                    print('Retrying a failed startup with no saved training results: {}'.format(output), flush=True)
                else:
                    parser.error('Output is not empty: {}. Choose a new --output-root '
                                 'to avoid overwriting or implicit resume.'.format(output))

    environment = os.environ.copy()
    if args.cuda_devices is not None:
        environment['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    environment['PYTHONUNBUFFERED'] = '1'
    for entry, seed, command, output in jobs:
        print('\n[{} / seed={}] {}'.format(entry['name'], seed, entry['description']), flush=True)
        print('Reference: {}'.format(entry['reference'] or '-'), flush=True)
        if args.cuda_devices is not None:
            print('CUDA_VISIBLE_DEVICES={}'.format(args.cuda_devices), flush=True)
        print(format_command(command), flush=True)
        if args.dry_run:
            continue
        output.mkdir(parents=True, exist_ok=True)
        record = {
            'experiment': entry, 'seed': seed, 'command': command,
            'cwd': str(PROJECT_ROOT),
            'cuda_visible_devices': environment.get('CUDA_VISIBLE_DEVICES'),
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'running',
        }
        record_path = output / 'ablation_run.json'
        if record_path.is_file():
            # Preserve the failed attempt's command/status when retrying it.
            record['previous_run'] = json.loads(record_path.read_text(encoding='utf-8'))
        def save_record():
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        save_record()
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
        except (KeyboardInterrupt, OSError) as exc:
            record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=str(exc), finished_utc=datetime.now(timezone.utc).isoformat())
            save_record()
            write_summary(args.output_root, selected, args.seeds)
            return 130 if isinstance(exc, KeyboardInterrupt) else 1
        record.update(status='completed' if result.returncode == 0 else 'failed',
                      returncode=result.returncode,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        save_record()
        summary_path = write_summary(args.output_root, selected, args.seeds)
        print('Best validation metrics: {}'.format(summary_path), flush=True)
        if result.returncode:
            print('Training failed; remaining experiments were not launched.', file=sys.stderr)
            return result.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
