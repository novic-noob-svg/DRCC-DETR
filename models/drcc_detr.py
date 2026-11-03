"""Paper-facing DRCC-DETR model builder using the shared DINO implementation."""
from .dino.dino import build_dino
from .registry import MODULE_BUILD_FUNCS


@MODULE_BUILD_FUNCS.registe_with_name(module_name='drcc_detr')
def build_drcc_detr(args):
    return build_dino(args)
