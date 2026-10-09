"""Transfer-learning classifiers for histopathology patches."""
import torch.nn as nn
from torchvision import models


def build_model(arch: str = "resnet18", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained backbone with a new classification head.

    Supported: resnet18, resnet50, efficientnet_b0.
    """
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model
