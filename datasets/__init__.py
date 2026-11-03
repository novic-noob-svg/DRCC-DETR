# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch.utils.data
import torchvision
from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, torchvision.datasets.CocoDetection):
        raise TypeError('Expected a COCO detection dataset')
    return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file != 'coco':
        raise ValueError('Use COCO-format detection annotations for this release.')
    return build_coco(image_set, args)
