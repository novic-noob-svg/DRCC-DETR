"""VisDrone scaffold. Final AI-TOD settings are not substituted for a VisDrone run."""
_base_ = ['../DINO/DINO_4scale.py']
num_classes = 11
modelname = 'drcc_detr'
use_vcr = True
use_sadra = True
vcr_loss_coef = 0.5
vcr_temperature = 0.07
vcr_gt_constrained = True
vcr_use_negative_queries = True
vcr_enhancement_mode = 'laplacian'
sadra_scale_threshold = 0.1
sadra_max_scale = 2.5
sadra_use_boundary_band = False
sadra_boundary_band_width = 0.2
sadra_use_scale_expansion = True
sadra_use_region_prior = True
sadra_context_points = 2
