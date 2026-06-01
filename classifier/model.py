import torch
import torch.nn as nn
from torchvision import models

VIEWS = ["A2C", "A4C", "PLAX", "PSAX"]  # sorted alphabetically to match ImageFolder


class EchoViewClassifier(nn.Module):
    def __init__(self, num_classes: int = len(VIEWS), pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "EchoViewClassifier":
        checkpoint = torch.load(path, map_location=device)
        num_classes = checkpoint.get("num_classes", len(VIEWS))
        model = cls(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
