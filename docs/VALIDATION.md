# Validation performed during source preparation

Date: 2026-11-02.

Local environment: Python 3.10, PyTorch 2.14.1+cpu, torchvision 0.29.1, and the dependencies listed in requirements. These are the preparation machine's test versions; the original GPU environment version list was not included in the supplied run metadata.

## Actual final checkpoint

- Loaded the supplied `output_aitod/checkpoint_best_regular.pth` into both the original backup and the prepared release using strict state-dictionary checks.
- Both models contain 47,151,522 parameters, including frozen parameters. No missing or unexpected tensor keys were found.
- The saved checkpoint epoch is 23 (the zero-based training epoch index).
- Compared FP32 evaluation outputs on the same seeded 128 x 128 CPU input with all 900 queries and the full five-scale/six-layer model.
- Maximum absolute difference in `pred_boxes`: **0.0**.
- Maximum absolute difference in `pred_logits`: **0.0**.

The CPU operator for the original backup was supplied by the same upstream PyTorch reference aggregation used for this comparison. This is an output-equivalence check, not a CUDA performance measurement.

## VCR/SADRA regression

The default VCR loss and query/head gradients were compared against the supplied backup. SADRA offset outputs, prior logits, offset gradients, attention gradients and regional-scalar gradients were also compared. All tested differences were zero.

`python tests/run_checks.py` runs 24 unit cases and 8 additional module checks. In the local preparation environment, **23 unit cases and all 8 module checks passed**. One optional case requiring the AI-TOD API's compiled package was skipped because that package was not installed locally. The API is included as an explicit optional install requirement.

The checks cover old configuration/tensor-name migration, strict mismatch detection, NumPy-alias compatibility, AI-TOD metric-index mapping, annotation preservation, cached evaluation records, best-checkpoint retention, VCR valid-region/query-gradient behavior, standard versus neighbor-safe DN, SADRA point allocation/prior controls, and a reduced-size full-model training graph.

## Executed application paths

- Actual `main.py` training, evaluation, logging, configuration serialization and best-weight saving on a two-image synthetic COCO fixture for one epoch, with VCR and SADRA enabled.
- All nine component-ablation commands parsed and resolved to the expected module switches. Their historical boundary-band-off/neighbor-safe-CDN-on settings were preserved.
- Help/startup checks for eight evaluation, visualization, data-check and config-export tools.
- VCR single-image and aggregate CSV-to-PDF exports on a synthetic fixture.
- All Python source files parsed successfully.
- The original CUDA/C++ source files were retained, and the release archive excludes caches, datasets and model weights.

## Server-side checks

CUDA operator compilation/testing, GPU mixed-precision behavior, multi-GPU training, dataset-scale final evaluation and a fresh reproduction of the reported AP tables require the original training server/environment. The local checks above do not replace those runs.

Before recording a new reproduction, save `pip freeze` together with `config_args_all.json`, the checkpoint and the evaluation metadata. Use the final-run configuration for AI-TOD and matched reference configurations for component ablations.

## Public model rename: 2026-11-03

The public model name is DRCC-DETR, its registry key is `drcc_detr`, and its configuration folder is `config/DRCC_DETR`. Old saved model registry names are migrated during configuration loading.

After renaming, 23 unit tests and 8 module checks passed; the optional AI-TOD compiled-API case remained skipped. The training CLI starts with the DRCC-DETR title. The model, loss, attention, checkpoint loading and training-loop implementations listed in the external rename audit are byte-identical to the prior release. This rename did not rerun dataset-scale AP or CUDA benchmarks.
