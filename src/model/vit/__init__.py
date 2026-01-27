from typing import Optional, Type, Union
from .vision_transformer import VisionTransformerBaseCfg, VisionTransformerBase
from .dinov2 import DinoV2Cfg, DinoV2


ViTS = {
    "dinov2": DinoV2,
}

ViTCfgs = {
    "dinov2": DinoV2Cfg,
}

# Option 3: Provide a generic ViTCfg as a Union of all supported config types
ViTCfg = Union[DinoV2Cfg]

def get_vit(cfg: VisionTransformerBaseCfg) -> VisionTransformerBase:
    vit = ViTS[cfg.name](cfg)
    return vit

def get_vit_cfg(name: str) -> Type[VisionTransformerBaseCfg]:
    return ViTCfgs[name]