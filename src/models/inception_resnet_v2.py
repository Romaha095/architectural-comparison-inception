from __future__ import annotations

import torch.nn as nn
import timm


def build_inception_resnet_v2(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    model = timm.create_model(
        "inception_resnet_v2",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    return model