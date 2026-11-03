"""Canonical configuration names and explicit command-line override handling."""
import argparse
import sys

LEGACY_KEYS = {
    'use_vca': 'use_vcr', 'use_vca_dn': 'use_vcr',
    'vca_dn_temperature': 'vcr_temperature',
    'vca_dn_loss_coef': 'vcr_loss_coef',
    'vca_enhancement_mode': 'vcr_enhancement_mode',
    'vca_gt_constrained': 'vcr_gt_constrained',
    'vca_use_negative_queries': 'vcr_use_negative_queries',
}


def normalize_config(values, defaults=True):
    if isinstance(values, argparse.Namespace):
        values = vars(values)
    result = {}
    for key, value in values.items():
        canonical = LEGACY_KEYS.get(key, key)
        if canonical in result and result[canonical] != value:
            raise ValueError('Conflicting legacy/canonical settings for ' + canonical)
        result[canonical] = value
    # Accept configuration files saved before the public model rename.
    if result.get('modelname') == 'dcc_detr':
        result['modelname'] = 'drcc_detr'
    # Missing in the authoritative final run: build_dino used getattr(..., False).
    if defaults:
        result.setdefault('use_ns_cdn', False)
    return result


class ConfigArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        tokens = list(sys.argv[1:] if args is None else args)
        parsed, remaining = super().parse_known_args(tokens, namespace)
        explicit = []
        for token in tokens:
            action = self._option_string_actions.get(token.split('=', 1)[0])
            if action is not None and action.dest not in explicit:
                explicit.append(action.dest)
        parsed._explicit_args = explicit
        return parsed, remaining


def resolve_training_config(args):
    from util.slconfig import SLConfig
    cfg = SLConfig.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)
    values = normalize_config(cfg._cfg_dict.to_dict())
    explicit = set(getattr(args, '_explicit_args', []))
    for key, value in values.items():
        if key not in explicit:
            setattr(args, key, value)
    args.use_vcr = bool(getattr(args, 'use_vcr', False))
    args.use_sadra = bool(getattr(args, 'use_sadra', False))
    if getattr(args, 'masks', False) or getattr(args, 'frozen_weights', None):
        raise ValueError('This release implements bbox detection, not segmentation.')
    if getattr(args, 'use_pseudo_cluster', False):
        raise ValueError('Pseudo-clustering is not part of the DRCC-DETR paper implementation.')
    return args, cfg
