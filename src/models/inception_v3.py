from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights


def build_inception_v3(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    if pretrained:
        weights = Inception_V3_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.inception_v3(weights=weights, aux_logits=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model