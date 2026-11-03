#!/usr/bin/env python3
"""Run DINO inference on local images and save visualized detections."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import datasets.transforms as T
from main import build_model_main, get_args_parser, load_model_state
from util.slconfig import DictAction, SLConfig
from util.release_config import normalize_config
from util.checkpoint import load_checkpoint


LOGGER = logging.getLogger("dino_visualize")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize DINO detections on local images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="An image file or a directory of images.")
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations"))
    parser.add_argument("--config-file", "-c", type=Path, default=None)
    parser.add_argument("--run-config", type=Path, default=None,
                        help="config_args_all.json saved beside the checkpoint.")
    parser.add_argument("--weights-key", default="model",
                        help="Checkpoint key, normally model or ema_model.")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--options", nargs="+", action=DictAction, default=None,
                        metavar="KEY=VALUE")
    parser.add_argument("--use-vcr", dest="use_vcr", action="store_true")
    parser.add_argument("--no-use-vcr", dest="use_vcr", action="store_false")
    parser.add_argument("--use-sadra", dest="use_sadra", action="store_true")
    parser.add_argument("--no-use-sadra", dest="use_sadra", action="store_false")
    parser.set_defaults(use_vcr=None, use_sadra=None)
    return parser


def absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_checkpoint(path: Path) -> Path:
    path = absolute(path)
    if path.is_file():
        return path
    candidates = ("checkpoint_best_regular.pth", "checkpoint_best_ema.pth", "checkpoint.pth")
    for name in candidates:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No checkpoint found at {}".format(path))


def load_args(cli: argparse.Namespace, checkpoint: Path) -> argparse.Namespace:
    run_config = absolute(cli.run_config) if cli.run_config else checkpoint.parent / 'config_args_all.json'
    values = vars(get_args_parser().parse_args(['-c', str(cli.config_file or 'unused')]))
    if run_config.is_file():
        source = json.loads(run_config.read_text(encoding='utf-8-sig'))
    elif cli.config_file:
        source = SLConfig.fromfile(str(absolute(cli.config_file)))._cfg_dict.to_dict()
    else:
        data = load_checkpoint(checkpoint)
        if 'args' not in data:
            raise FileNotFoundError('Pass --run-config/--config-file, or keep config_args_all.json beside weights.')
        source = data['args']
    values.update(normalize_config(source))
    if cli.options:
        values.update(normalize_config(cli.options, defaults=False))
    for name in ('vcr', 'sadra'):
        override = getattr(cli, 'use_' + name, None)
        if override is not None:
            values['use_' + name] = override
        if values.get('use_' + name) is None:
            raise ValueError('Specify use_' + name + ' in the saved run config or command line.')
    values.update(distributed=False, rank=0, local_rank=0, gpu=0, world_size=1,
                  device=cli.device, eval=True, backbone_pretrained=False)
    return argparse.Namespace(**values)


def load_model(args: argparse.Namespace, checkpoint: Path, device: torch.device,
               weights_key: str):
    model, _, postprocessors = build_model_main(args)
    try:
        data = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(str(checkpoint), map_location="cpu")
    state_dict = data[weights_key] if weights_key in data else data
    load_model_state(model, state_dict, LOGGER, strict=True)
    model.to(device).eval()
    return model, postprocessors


def iter_images(input_path: Path):
    if input_path.is_file():
        yield input_path
    elif input_path.is_dir():
        yield from sorted(path for path in input_path.rglob("*")
                          if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        raise FileNotFoundError("Input does not exist: {}".format(input_path))


def draw_result(image: Image.Image, result: dict, threshold: float) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for score, box in zip(result["scores"], result["boxes"]):
        if float(score) < threshold:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=3)
    return canvas


def main(cli: argparse.Namespace) -> None:
    if not 0.0 <= cli.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    checkpoint = resolve_checkpoint(cli.checkpoint)
    args = load_args(cli, checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    model, postprocessors = load_model(args, checkpoint, device, cli.weights_key)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    output_dir = absolute(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(iter_images(absolute(cli.input)))
    if not image_paths:
        raise FileNotFoundError("No supported images found in {}".format(cli.input))

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths, start=1):
            original = Image.open(image_path).convert("RGB")
            image, _ = transform(original, None)
            samples = image.unsqueeze(0).to(device)
            outputs = model(samples)
            height, width = original.height, original.width
            result = postprocessors["bbox"](
                outputs, torch.tensor([[height, width]], device=device)
            )[0]
            result = {key: value.detach().cpu() for key, value in result.items()}
            rendered = draw_result(original, result, cli.threshold)
            destination = output_dir / image_path.name
            rendered.save(destination)
            kept = int((result["scores"] >= cli.threshold).sum())
            LOGGER.info("[%d/%d] %s -> %s (%d boxes)", index, len(image_paths), image_path, destination, kept)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main(get_parser().parse_args())
