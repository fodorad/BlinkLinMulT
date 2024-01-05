import torch
from torch import nn
import clip
from blinklinmult.models.backbone import AbstractModule


class CLIP(AbstractModule):

    def __init__(self, backbone: str = "ViT-B/32",
                       output_dim: int = 1,
                       freeze: bool = False):
        super().__init__()

        model, preprocess = clip.load(backbone, jit=False, device=torch.device("cpu"))

        self.preprocess = preprocess
        self.features = model.visual

        self.classifier = nn.Sequential(
            nn.LazyLinear(512),
            nn.BatchNorm1d(512, momentum=0.999, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(512, output_dim),
        )

        if freeze:
            self.freeze_feature_extractor()


if __name__ == '__main__':

    x = torch.zeros((32, 3, 224, 224)).to("cuda:0")
    for backbone in ["RN50", "ViT-B/32", "ViT-L/14"]:
        model = CLIP(backbone=backbone).to("cuda:0")
        y_pred = model(x)
        assert y_pred.shape == (32, 1)