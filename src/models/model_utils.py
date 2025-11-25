from __future__ import annotations

import torch.nn as nn

from .inception_v3 import build_inception_v3
from .inception_resnet_v2 import build_inception_resnet_v2


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    name = name.lower()
    if name in {"inception_v3", "iv3"}:
        return build_inception_v3(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    if name in {"inception_resnet_v2", "irnv2", "inception-resnet-v2"}:
        return build_inception_resnet_v2(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"Unknown model name: {name}")