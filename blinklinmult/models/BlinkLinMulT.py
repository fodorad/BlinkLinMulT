from abc import abstractmethod
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from linmult import LinMulT, LinT
from exordium.utils.ckpt import download_file
from blinklinmult import PathType, WEIGHTS_DIR


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


PRETRAINED_WEIGHTS = {
    'densenet121-union': 'https://github.com/fodorad/LinMulT/releases/download/v1.0.0/densenet121-union-64.pt',
    'blinklint-union': 'https://github.com/fodorad/LinMulT/releases/download/v1.0.0/densenetlint-union-64.pt',
    'blinklinmult-union': 'https://github.com/fodorad/LinMulT/releases/download/v1.0.0/blinklinmult-union.pt'
}


preprocess_eye_fcn = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AbstractModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Identity()
        self.classifier = nn.Identity()

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, (nn.Linear, nn.LazyLinear)):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def freeze_feature_extractor(self):
        for layer in self.features.parameters():
            layer.requires_grad = False

    def unfreeze_feature_extractor(self):
        for layer in self.features.parameters():
            layer.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def load_weights(self, weights_path: PathType) -> None:
        local_path = Path(weights_path)

        if local_path.name in PRETRAINED_WEIGHTS.keys():
            remote_path = PRETRAINED_WEIGHTS[local_path.name]
            local_path = WEIGHTS_DIR / Path(remote_path).name
            download_file(remote_path, local_path)

        state_dict = torch.load(str(local_path), map_location="cpu")
        self.load_state_dict(state_dict)
        logging.info(f'Weights are loaded from {local_path}')


class DenseNet121(AbstractModule):

    def __init__(self, output_dim: int = 1,
                       weights: PathType | None = 'densenet121-union',
                       freeze: bool = False):
        super().__init__()
        densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        features = [module for module in densenet121.features]
        features.append(nn.ReLU(inplace=True))
        features.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        features.append(nn.Flatten())
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.LazyLinear(512),
            nn.BatchNorm1d(512, momentum=0.999, eps=1e-3),
            nn.GELU(),
            nn.Dropout(p=0.6),
            nn.Linear(512, output_dim),
        )

        if freeze:
            self.freeze_feature_extractor()

        if weights is not None:
            self.load_weights(weights)


class BlinkLinMulT(AbstractModule):

    def __init__(self, input_dim: int = 160,
                       output_dim: int = 1,
                       weights: PathType | None = 'blinklinmult-union',
                       weights_backbone: PathType | None = 'densenet121-union',
                       **kwargs):
        super().__init__()
        self.img_backbone = DenseNet121(output_dim=output_dim, weights=weights_backbone)
        self.img_backbone = self.img_backbone.features
        logging.info(self.img_backbone)

        self.rnn_backbone = LinMulT(
            input_modality_channels=[1024, input_dim],
            output_dim=output_dim,
            projected_modality_dim=32,
            number_of_layers=5,
            add_projection_fusion=False,
            aggregation='meanpooling',
            **kwargs,
        )
        logging.info(self.rnn_backbone)

        if weights is not None:
            self.load_weights(weights)

    def forward(self, x):
        time_dim = x[0].size(1)
        rgb_texture = x[0] # (B, L, C, H, W)
        high_level_features = x[1] # (B, L, C)

        eyes_x = []
        for t in range(time_dim):
            eyes_x.append(torch.flatten(self.img_backbone(rgb_texture[:, t, :, :, :]), 1))

        x0 = torch.stack(eyes_x, dim=1)
        x_seq = self.rnn_backbone([x0, high_level_features])
        return x_seq


class BlinkLinT(AbstractModule):

    def __init__(self, output_dim: int = 1,
                       weights: str | Path | None = 'blinklint-union'):
        super().__init__()
        self.img_backbone = DenseNet121(output_dim=output_dim, weights='densenet121-union')
        self.img_backbone = self.img_backbone.features
        self.rnn_backbone = LinT(1024, projected_modality_dim=32, number_of_layers=5, output_dim=output_dim)

        logging.info(self.img_backbone)
        logging.info(self.rnn_backbone)

        if weights is not None:
            self.load_weights(weights)

    def forward(self, rgb_texture):
        time_dim = rgb_texture.size(1) # (B, L, C, H, W)

        eyes_x = []
        for t in range(time_dim):
            eyes_x.append(torch.flatten(self.img_backbone(rgb_texture[:, t, :, :, :]), 1))

        x0 = torch.stack(eyes_x, dim=1)
        x_seq = self.rnn_backbone(x0)
        return x_seq