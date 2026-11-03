# DRCC-DETR

Source implementation for **DRCC-DETR: Dual-Region Contrastive and Context Learning for Small Object Detection**, built on DINO.

The two paper modules are:

- **Visual Contrastive Regularization (VCR)**: a training-only auxiliary objective using GT-constrained foreground/background visual prototypes and denoising queries.
- **Scale-Adaptive Dual-Region Attention (SADRA)**: decoder sampling with a fixed local/context slot allocation, scale-dependent contextual offsets and a per-layer regional prior.

This source release is reconstructed against the saved final AI-TOD-V2 training configuration. See [reproduction details](docs/REPRODUCTION.md) for the exact settings and the distinction between that run and the component-ablation suite. Model weights and datasets are distributed separately; this archive contains source code only.

## Get the source

```bash
git clone https://github.com/novic-noob-svg/DRCC-DETR.git
cd DRCC-DETR
```

Saved configurations using the previous `modelname` are normalized automatically. The rename does not change model tensors or checkpoint filenames.

## Installation

Use a Python 3.10 environment for the original training workflow. Install a matching CUDA-enabled PyTorch/torchvision pair for your server first, then:

```bash
pip install -r requirements.txt
cd models/dino/ops
bash make.sh
python test.py
cd ../../..
```

The CUDA toolkit/compiler must match the PyTorch build. The original CUDA implementation is retained. A PyTorch reference path supports CPU tests and small inference checks; it is not the performance benchmark implementation.

For AI-TOD-V2 evaluation, also install its dataset-specific API:

```bash
pip install -r requirements-aitod.txt
```

## Dataset format

The datasets use COCO-format instance annotations:

```text
DATASET_ROOT/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

For AI-TOD-V2, `train2017` contains the **trainval** split and `val2017` contains the **test** split in the supplied conversion. These are folder names, not a different benchmark split. Keep image IDs, category IDs and annotation areas unchanged.

```bash
python tools/check_dataset.py --coco-path /path/to/AITODV2 --num-classes 9
```

The label space uses `num_classes = max(category_id) + 1`, rather than automatically renumbering COCO category IDs. The final AI-TOD configuration has 9 output channels. Check the category IDs in your conversion and keep classifier dimensions unchanged when loading its weights.

## Train the final AI-TOD-V2 configuration

Run from the repository root. The example uses four GPUs, as in the saved run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 main.py \
  -c config/DRCC_DETR/aitodv2_r50_5scale.py \
  --coco_path /path/to/AITODV2 \
  --output_dir outputs/aitodv2 \
  --seed 42
```

Important captured settings include 5 feature scales, 900 queries, 24 epochs, batch size 1 per process, EMA enabled, mixed precision disabled, VCR weight 0.5, and SADRA threshold/cap 0.1/2.5. Boundary-band projection is **enabled** with width 0.2. Neighbor-safe CDN is **disabled**, matching the absent setting's effective default in the final run. Do not add `--amp` or change these settings when reproducing this configuration.

The original trainer retains regular/EMA checkpoints and writes `config_args_all.json`, `config_cfg.py`, `info.txt`, and `log.txt` into the output directory. Keep the saved full configuration with every checkpoint. `--save_best_only` is available when only the selected model weights are needed; such a file is not an optimizer-resume checkpoint.

Resume an interrupted run from the full training checkpoint:

```bash
torchrun --standalone --nproc_per_node=4 main.py \
  -c config/DRCC_DETR/aitodv2_r50_5scale.py \
  --coco_path /path/to/AITODV2 --output_dir outputs/aitodv2 \
  --resume outputs/aitodv2/checkpoint.pth
```

## Evaluate AI-TOD-V2

The dedicated evaluator uses the AI-TOD API and reports AP, AP50, AP75, APvt, APt, APs and APm in percent. Standard COCO APs/APm/APl are not substitutes for these size groups.

```bash
python tools/eval_aitodv2.py \
  --checkpoint checkpoints/checkpoint_best_regular.pth \
  --run-config checkpoints/config_args_all.json \
  --coco-path /path/to/AITODV2 \
  --output-dir outputs/aitodv2_test
```

The evaluator selects 1500 candidates and uses `maxDets=[1,100,1500]`. Use the same prediction settings for compared models. The saved run configuration determines module switches and geometry. Historical `use_vca` fields and checkpoint tensor names are migrated automatically to VCR names.

The evaluator writes `predictions.json`, `metrics.json`, `metrics.csv` and `latex_row.txt`. Per-image evaluation is cached before accumulation, so a compatible saved cache can resume accumulation. Existing predictions can be evaluated without inference:

```bash
python tools/eval_aitodv2.py \
  --predictions outputs/aitodv2_test/predictions.json \
  --coco-path /path/to/AITODV2 --output-dir outputs/aitodv2_recheck
```

Use `--eval-cache /path/to/evaluation_cache.pkl --output-dir /path/to/new_results` to resume from this evaluator's own cache. The deprecated NumPy aliases used by the dataset API are handled locally, without modifying global NumPy.

## Other datasets and ablations

`config/DRCC_DETR/visdrone_r50_4scale.py` is a VisDrone scaffold, not an assertion that the AI-TOD run configuration reproduces every VisDrone table. Export the actual saved run settings for each reported experiment:

```bash
python tools/export_run_config.py \
  --input /path/to/experiment/config_args_all.json \
  --output config/DRCC_DETR/my_verified_run.py
```

VisDrone and COCO-format HIT-UAV use the same detection loader. Set their actual category-ID space and saved training configuration. Standard COCO evaluation uses:

```bash
python main.py -c config/DRCC_DETR/my_verified_run.py \
  --coco_path /path/to/dataset --resume checkpoints/model.pth \
  --eval --output_dir outputs/eval
```

Preview the retained component-ablation controls:

```bash
python tools/run_ablations.py --list
python tools/run_ablations.py --coco-path /path/to/VisDrone2019 \
  --experiments all --output-root outputs/component_ablations \
  --nproc-per-node 4 --cuda-devices 0,1,2,3 --dry-run
```

Remove `--dry-run` to train. This suite deliberately preserves its historical settings: boundary band **off**, neighbor-safe CDN **on**. It therefore has different training settings from the final AI-TOD run. It includes its own DINO, VCR and SADRA references; compare controls against those matched references.

Preview/run the seven parameter-sensitivity settings with the actual sensitivity-run config:

```bash
python tools/run_sadra_sensitivity.py \
  --config-file config/DRCC_DETR/my_verified_run.py \
  --coco-path /path/to/VisDrone2019 --nproc-per-node 4 \
  --cuda-devices 0,1,2,3 --dry-run
```

## Visualizations

Detection overlays:

```bash
python tools/inference_visualize.py --checkpoint checkpoints/model.pth \
  --run-config checkpoints/config_args_all.json \
  --input /path/to/images --output-dir visualizations/detections
```

SADRA sampling comparison between independently trained models:

```bash
python tools/visualize_sadra.py \
  --checkpoint checkpoints/sadra/model.pth \
  --run-config checkpoints/sadra/config_args_all.json \
  --baseline-checkpoint checkpoints/dino/model.pth \
  --baseline-run-config checkpoints/dino/config_args_all.json \
  --input /path/to/images \
  --coco-annotations /path/to/instances_val2017.json \
  --score-threshold 0.3 --match-iou 0.5 \
  --output-dir visualizations/sadra --save-pdf
```

Without a baseline checkpoint, the comparison shows pre/post SADRA geometry within the same model; it is not an independently trained DINO comparison. Cyan local slots are unexpanded learned samples and are not guaranteed to stay inside a GT box. Context slots may lie outside GT/final prediction boxes. Point sizes encode relative attention weights within the displayed head/level. Sampling diagnostics are enabled only for visualization.

VCR query/prototype comparison:

```bash
python tools/visualize_vcr.py \
  --baseline-checkpoint checkpoints/dino/model.pth \
  --baseline-run-config checkpoints/dino/config_args_all.json \
  --vcr-checkpoint checkpoints/vcr_or_drcc/model.pth \
  --run-config checkpoints/vcr_or_drcc/config_args_all.json \
  --input /path/to/images \
  --coco-annotations /path/to/instances_val2017.json \
  --output-dir visualizations/vcr
```

The VCR/DRCC model retains its saved SADRA setting; the baseline uses ordinary DINO attention. `tools/plot_vcr_csv_pdf.py` exports single-image or aggregate vector PDFs from the resulting CSV files.

## Source map and tests

| Paper component | Implementation |
|---|---|
| DRCC-DETR detector and training loss wiring | `models/dino/dino.py` |
| DRCC-DETR public builder | `models/drcc_detr.py` |
| VCR spatial scoring head | `models/dino/vcr_head.py` |
| VCR Laplacian modulation and controls | `models/dino/vcr_enhancement.py` |
| VCR GT supports, prototypes and bidirectional contrastive loss | `models/dino/vcr_loss.py` |
| SADRA paper-facing import | `models/dino/sadra.py` |
| SADRA offset/prior implementation | `models/dino/ops/modules/ms_deform_attn.py` |
| Legacy configuration and weight migration | `util/release_config.py`, `util/checkpoint.py` |

```bash
python tests/run_checks.py
```

CPU checks exercise the modules, checkpoint migration, metric layout, and a reduced-size end-to-end training graph. Run `models/dino/ops/test.py` on the CUDA training machine as well. Local checks do not establish a fresh reproduction of all dataset AP numbers.

## Acknowledgments and license

This project builds on [DINO](https://github.com/IDEA-Research/DINO), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [DETR](https://github.com/facebookresearch/detr), and their upstream dependencies. Original source copyright headers and the Apache-2.0 license are retained. See [NOTICE](NOTICE).

For work using this implementation, cite the DRCC-DETR manuscript once its final bibliographic record is available and the upstream DINO work:

```bibtex
@inproceedings{zhang2023dino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author={Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang
          and Zhu, Jun and Ni, Lionel M. and Shum, Heung-Yeung},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
