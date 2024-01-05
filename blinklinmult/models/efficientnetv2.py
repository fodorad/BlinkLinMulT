from torch import nn
from torchvision import models
from blinklinmult.models.backbone import AbstractModule


class EfficientNetV2(AbstractModule):

    def __init__(self, output_dim: int = 1, freeze: bool = False):
        super().__init__()
        self.model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

        self.features = nn.Sequential(
            self.model.features,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512, momentum=0.999, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(512, output_dim),
        )

        if freeze:
            self.freeze_feature_extractor()