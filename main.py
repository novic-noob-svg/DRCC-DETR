# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.release_config import ConfigArgumentParser, resolve_training_config
from util.checkpoint import load_model_state, load_checkpoint
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch


def get_args_parser():
    parser = ConfigArgumentParser('DRCC-DETR', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default=None)
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Keep a single weights-only checkpoint selected by validation bbox AP; no epoch/latest checkpoints')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", default=0, type=int,
                        help='local rank for DistributedDataParallel (normally set by torchrun)')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.set_defaults(
        use_vcr=None,
        use_sadra=None,
        sadra_use_boundary_band=True,
    )

    parser.add_argument('--use_vcr', dest='use_vcr', action='store_true',
                        help='Enable VCR-guided denoising contrastive loss')
    parser.add_argument('--no_use_vcr', dest='use_vcr', action='store_false',
                        help='Disable VCR-guided denoising contrastive loss')
    parser.add_argument('--vcr_temperature', default=0.07, type=float,
                        help='Temperature for VCR-guided DN contrastive loss')
    parser.add_argument('--vcr_loss_coef', default=0.5, type=float,
                        help='Loss coefficient for VCR-guided DN contrastive loss')
    parser.add_argument('--vcr_gt_constrained', dest='vcr_gt_constrained',
                        action='store_true',
                        help='Restrict VCR positive/negative supports using GT boxes (default)')
    parser.add_argument('--no_vcr_gt_constrained', dest='vcr_gt_constrained',
                        action='store_false',
                        help='Allow both VCR maps to select any valid image cell')
    parser.set_defaults(vcr_gt_constrained=True)
    parser.add_argument('--vcr_use_negative_queries',
                        dest='vcr_use_negative_queries',
                        action='store_true',
                        help='Include the second-half DN queries in VCR loss (default)')
    parser.add_argument('--no_vcr_use_negative_queries',
                        dest='vcr_use_negative_queries',
                        action='store_false',
                        help='Ablate the VCR loss for second-half DN queries')
    parser.set_defaults(vcr_use_negative_queries=True)

    parser.add_argument('--use_sadra', dest='use_sadra', action='store_true',
                        help='Enable SADRA in decoder cross-attention')
    parser.add_argument('--no_use_sadra', dest='use_sadra', action='store_false',
                        help='Disable SADRA and use standard decoder cross-attention')
    parser.add_argument('--sadra_max_scale', default=2.5, type=float,
                        help='Maximum expansion ratio for SADRA context sampling')
    parser.add_argument('--sadra_scale_threshold', default=0.1, type=float,
                        help='Normalized linear object scale below which SADRA expands context')
    parser.add_argument('--sadra_use_boundary_band',
                        dest='sadra_use_boundary_band', action='store_true',
                        help='Constrain SADRA context points to a soft band around the reference-box boundary (default)')
    parser.add_argument('--no_sadra_boundary_band',
                        dest='sadra_use_boundary_band', action='store_false',
                        help='Use the original unconstrained SADRA context expansion for ablation')
    parser.add_argument('--sadra_boundary_band_width', default=0.2, type=float,
                        help='Relative half-width of the soft reference-box boundary band in [0, 1)')

    parser.add_argument('--vcr_enhancement_mode', default='laplacian',
                        choices=('laplacian', 'identity', 'constant'),
                        help='VCR feature enhancement: original Laplacian gate, Fe=Fp, or Fe=1.5*Fp')
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument('--sadra_use_scale_expansion',
                             dest='sadra_use_scale_expansion', action='store_true')
    scale_group.add_argument('--no_sadra_scale_expansion',
                             dest='sadra_use_scale_expansion', action='store_false',
                             help='Keep SADRA grouping/prior but set the context expansion factor to one')
    prior_group = parser.add_mutually_exclusive_group()
    prior_group.add_argument('--sadra_use_region_prior',
                             dest='sadra_use_region_prior', action='store_true')
    prior_group.add_argument('--no_sadra_region_prior',
                             dest='sadra_use_region_prior', action='store_false',
                             help='Keep SADRA geometry but use the original attention logits')
    parser.set_defaults(sadra_use_scale_expansion=True,
                        sadra_use_region_prior=True)
    parser.add_argument('--sadra_context_points', default=None, type=int,
                        help='Context points K per head/level; default P//2, require 1 <= K < dec_n_points')

    parser.add_argument('--use_tb', action='store_true',
                        help='Enable TensorBoard logging (writes to <output_dir>/tensorboard)')
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    utils.init_distributed_mode(args)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    args, cfg = resolve_training_config(args)
    if not args.coco_path:
        raise ValueError('Pass --coco_path with the COCO-format dataset root.')
    if args.output_dir and utils.is_main_process():
        cfg.dump(os.path.join(args.output_dir, 'config_cfg.py'))
        with open(os.path.join(args.output_dir, 'config_args_raw.json'), 'w', encoding='utf-8') as handle:
            json.dump(vars(args), handle, indent=2, default=str)

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False
    if args.use_vcr is None:
        args.use_vcr = False
    if args.use_sadra is None:
        args.use_sadra = False

    if args.use_vcr:
        if args.vcr_temperature <= 0:
            raise ValueError('--vcr_temperature must be positive')
        if args.vcr_loss_coef < 0:
            raise ValueError('--vcr_loss_coef must be non-negative')

    if args.use_sadra:
        if args.sadra_max_scale < 1.0:
            raise ValueError('--sadra_max_scale must be >= 1.0')
        if not 0.0 < args.sadra_scale_threshold <= 1.0:
            raise ValueError('--sadra_scale_threshold must be in (0, 1]')
        if (args.sadra_use_boundary_band and
                not 0.0 <= args.sadra_boundary_band_width < 1.0):
            raise ValueError(
                '--sadra_boundary_band_width must be in [0, 1)')
        if args.dec_n_points < 2:
            raise ValueError(
                'SADRA requires dec_n_points >= 2')
        if (args.sadra_context_points is not None and
                not 1 <= args.sadra_context_points < args.dec_n_points):
            raise ValueError(
                '--sadra_context_points must satisfy 1 <= K < dec_n_points')
        if args.use_deformable_box_attn:
            raise ValueError(
                'SADRA cannot be combined with --use_deformable_box_attn')

    # setup logger
    log_path = os.path.join(args.output_dir, 'info.txt') if args.output_dir else None
    logger = setup_logger(output=log_path, distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0 and args.output_dir:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')


    print(args)

    if args.distributed and args.device.startswith('cuda'):
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params,
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)


    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)


    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
    if not args.resume and args.output_dir and os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = load_checkpoint(args.resume)
        load_model_state(model_without_ddp, checkpoint['model'], logger, strict=True)
        if args.use_ema:
            if 'ema_model' in checkpoint:
                load_model_state(ema_m.module, checkpoint['ema_model'], logger, strict=True)
            else:
                ema_m = ModelEma(model_without_ddp, args.ema_decay)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
            except ValueError as exc:
                logger.warning(
                    "Optimizer state is incompatible with the VCR-only model; model weights were "
                    "restored but optimizer/scheduler state was skipped: %s", exc)

    if (not args.resume) and args.pretrain_model_path:
        checkpoint_data = load_checkpoint(args.pretrain_model_path)
        checkpoint = checkpoint_data['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))

        _load_output = load_model_state(model_without_ddp, _tmp_st, logger)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint_data:
                load_model_state(ema_m.module, checkpoint_data['ema_model'], logger)
            else:
                ema_m = ModelEma(model_without_ddp, args.ema_decay)


    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    best_weights_saver = None
    if args.save_best_only and args.output_dir and utils.is_main_process():
        from util.best_weights import BestWeightsSaver
        best_weights_saver = BestWeightsSaver(output_dir)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir and not args.save_best_only:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)

        # eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if best_weights_saver is not None:
            best_weights_saver.update(model_without_ddp, epoch,
                                      test_stats['coco_eval_bbox'], args)
        if _isbest and args.output_dir and not args.save_best_only:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if best_weights_saver is not None:
                best_weights_saver.update(ema_m.module, epoch,
                                          ema_test_stats['coco_eval_bbox'], args,
                                          model_type='ema')
            if _isbest and args.output_dir and not args.save_best_only:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None and not args.save_best_only:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.


if __name__ == '__main__':
    # 寮哄埗浣跨敤 'spawn' 妯″紡, 瑙ｅ喅 DDP 鍜?fork 鐨勬閿侀棶棰?    mp.set_start_method('spawn', force=True)
    parser = ConfigArgumentParser('DRCC-DETR training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
