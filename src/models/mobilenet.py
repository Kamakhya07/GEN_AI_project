import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def build_mobilenet_v2(num_classes: int = 5, pretrained: bool = True, fine_tune_layers: int = 4) -> nn.Module:
    """
    Builds MobileNetV2 with transfer learning for road damage classification.
    Optionally freezes bottom backbone layers and selectively fine-tunes top inverted residual blocks.
    """
    if pretrained:
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
    else:
        model = mobilenet_v2(weights=None)

    # Freeze base parameters if fine-tuning
    if pretrained and fine_tune_layers > 0:
        for param in model.features.parameters():
            param.requires_grad = False

        # Unfreeze the top N layers of features
        layers = list(model.features.children())
        for layer in layers[-fine_tune_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model
