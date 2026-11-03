# Reproduction scope

## Authoritative final run

The final AI-TOD-V2 preset was exported from the user-designated `output_aitod/config_args_all.json`. Its source hash and file-level source inventory are recorded in `source_manifest.json`. Private machine paths were not copied into the source release.

| Setting | Final AI-TOD-V2 run |
|---|---|
| Backbone | ResNet-50 |
| Feature levels | 5; backbone intermediate indices `[0,1,2,3]` |
| Output class channels | 9; annotation category IDs are preserved |
| Queries / decoder points | 900 / 4 |
| Encoder / decoder layers | 6 / 6 |
| Epochs | 24 |
| Seed | 42 |
| Processes / per-process batch | 4 / 1 |
| Optimizer / learning rate | AdamW / 0.0001 |
| Backbone learning rate | 0.00001 |
| LR schedule | MultiStepLR, milestones `[13,23]` |
| EMA / AMP | on / off |
| `embed_init_tgt` | false |
| DN groups setting / box noise | `dn_number=100` / 0.6 |
| VCR coefficient / temperature | 0.5 / 0.07 |
| GT constraints / negative-query objective | enabled / enabled |
| SADRA threshold / cap / contextual points | 0.1 / 2.5 / 2 |
| SADRA region prior / scale expansion | enabled / enabled |
| SADRA boundary band / width | enabled / 0.2 |
| Neighbor-safe CDN | disabled, matching the original absent-field default |
| Training validation selection | original standard COCO evaluator |
| Final AI-TOD-specific evaluation | dedicated API, 1500 candidates / maxDets 1500 |

Saved fields `ccm_params`, `ccm_cls_num`, `dynamic_query_list`, `val_epoch`, `find_unused_parameters`, `dn_scalar`, `dn_label_coef`, and `dn_bbox_coef` were removed from the public preset because the retained final DINO execution path does not read them. Their names must not be taken as evidence that a counting branch or dynamic query mechanism was used. The effective settings such as `num_queries` and `dn_number` were retained.

## Configuration is not inferred from weights alone

Boundary-band width, expansion threshold/cap, and neighbor-safe CDN are not all identifiable from a tensor state dictionary. Always pair a model file with its saved full run config. Tensor-name migration changes names only; it does not alter those settings.

## Paper and experiment correspondence

The saved AI-TOD-V2 configuration explicitly enables boundary-band projection. This adds a geometry step beyond the scale-expansion-only SADRA description in the current manuscript. It is retained to reproduce the supplied run. No claim is made that disabling it would preserve the measured AP. The manuscript was not edited during source preparation.

The historical component-ablation manifest uses boundary-band projection off and neighbor-safe CDN on. All reference runs in that suite must use the same settings. The final AI-TOD preset is not substituted for its references.

The VisDrone preset is a scaffold; exact VisDrone/HIT-UAV/sensitivity configs require their own saved run metadata. The sensitivity launcher changes only threshold/cap and module switches, keeping other values from its explicitly supplied config.

## Removed and preserved source

Removed from the release: duplicated backbone implementations, segmentation/panoptic execution paths, unrelated dataset hooks/projects, obsolete scripts, local data-copy hooks, private paths, logs, datasets, weights, temporary figures and Python/build caches. The original directories remain intact.

Preserved: DINO box detection, CUDA C++/CUDA kernels, matching/DN/auxiliary losses, original evaluation metrics, VCR and SADRA formulas, EMA and checkpoint handling. Component-control additions retain the same default formulas as the supplied backup.

Normal CUDA execution retains the compiled operator. CPU execution uses the upstream PyTorch reference aggregation for small correctness checks. Sampling diagnostics are off by default and enabled by the sampling visualization tool.

See `VALIDATION.md` for the checks actually performed; these checks are distinct from full dataset training or multi-GPU/CUDA performance verification.
