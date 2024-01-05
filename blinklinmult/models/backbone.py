import torch.nn as nn
from torchvision import models
from blinklinmult.models import AbstractModule


class Dense(AbstractModule):

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.features = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(128, output_dim),
        )


class EyeNet(AbstractModule):
    """EyeNet: An Improved Eye States Classification System using Convolutional Neural Network (2020)

    Paper: https://www.researchgate.net/publication/340757067
    """
    def __init__(self, output_dim: int = 1, freeze: bool = False):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 384, 3, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(384, 512, 3, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(84),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(84, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

        if freeze:
            self.freeze_feature_extractor()


class ResNet50(AbstractModule):

    def __init__(self, output_dim: int = 1, freeze: bool = False):
        super().__init__()

        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4,
            resnet50.avgpool,
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(512),
            nn.BatchNorm1d(512, momentum=0.999, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(512, output_dim),
        )

        if freeze:
            self.freeze_feature_extractor()