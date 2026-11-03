"""Read local experiment checkpoints and migrate historical VCA tensor names."""
from collections import OrderedDict
import logging
import torch


def load_checkpoint(path):
    """Load a trusted local training checkpoint, including its saved arguments."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location='cpu')


def canonical_parameter_name(name):
    while name.startswith('module.'):
        name = name[7:]
    parts = name.split('.')
    aliases = {'vca_feat_proj': 'vcr_feat_proj', 'glb_feat_proj': 'vcr_feat_proj',
               'vca_head': 'vcr_head', 'vca_dn_loss': 'vcr_loss'}
    return '.'.join(aliases.get(part, part) for part in parts)


def migrate_state_dict(state_dict):
    migrated = OrderedDict()
    for name, tensor in state_dict.items():
        canonical = canonical_parameter_name(name)
        if canonical in migrated:
            raise ValueError('Multiple checkpoint parameters map to ' + canonical)
        migrated[canonical] = tensor
    return migrated


def load_model_state(model, state_dict, logger=None, strict=False):
    """Strict loading is required for final evaluation; partial is for pretraining."""
    logger = logger or logging.getLogger(__name__)
    migrated = migrate_state_dict(state_dict)
    if not strict:
        expected = model.state_dict()
        mismatches = [key for key, value in migrated.items()
                      if key in expected and expected[key].shape != value.shape]
        if mismatches:
            logger.warning('Pretraining only: skipping shape-mismatched parameters: %s', mismatches)
            migrated = {key: value for key, value in migrated.items() if key not in mismatches}
    result = model.load_state_dict(migrated, strict=strict)
    if result.missing_keys or result.unexpected_keys:
        logger.warning('Checkpoint/config differences: %s', result)
    return result
