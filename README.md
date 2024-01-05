# BlinkLinMulT
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

BlinkLinMulT is trained for blink presence detection and eye state recognition tasks.
Our results demonstrate comparable or superior performance compared to state-of-the-art models on 2 tasks, using 7 public benchmark databases.
* paper: **BlinkLinMulT: Transformer-based Eye Blink Detection** ([website](https://www.mdpi.com/2313-433X/9/10/196))
* code: https://github.com/fodorad/BlinkLinMulT

# Setup
### Install package from PyPI for inference
```
pip install blinklinmult
```

### Install package for training
```
git clone https://github.com/fodorad/BlinkLinMulT
cd BlinkLinMulT
pip install -e .[all]
pip install -U -r requirements.txt
python -m unittest discover -s test
```

#### Supported extras definitions:
| extras tag | description |
| --- | --- |
| train | dependencies for training the model from scratch |
| all | extends the train dependencies for development, e.g. to include CLIP models |

# Quick start
### Load models from the paper with pre-trained weights
The pre-trained weights are loaded by default.
```
from blinklinmult.models import DenseNet121, BlinkLinT, BlinkLinMulT

model1 = DenseNet121()
model2 = BlinkLinT()
model3 = BlinkLinMulT()
```
In the next sessions there are more detailed examples with dummy data, e.g. a forward pass is performed, shapes are mentioned.


### Inference with dummy data
Pre-trained DenseNet121 for eye state recognition.
The forward pass is performed using an input image.
```
import torch
from blinklinmult.models import DenseNet121

# input shape: (batch_size, channels, height, width)
x = torch.rand((32, 3, 64, 64), device='cuda')
model = DenseNet121(output_dim=1, weights='densenet121-union').cuda()
y_pred = model(x)

# output shape: (batch_size, output_dimension)
assert y_pred.size() == torch.Size([32, 1])
```

Pre-trained BlinkLinT for blink presence detection and eye state recognition.
The forward pass is performed using sequence of images.
```
import torch
from blinklinmult.models import BlinkLinT

# input shape: (batch_size, time_dimension, channels, height, width)
x = torch.rand((8, 15, 3, 64, 64), device='cuda')
model = BlinkLinT(output_dim=1, weights='blinklint-union').cuda()
y_seq = model(x)

# output shape: (batch_size, time_dimension, output_dimension)
assert y_seq.size() == torch.Size([8, 15, 1])
```

Pre-trained BlinkLinMulT for blink presence detection and eye state recognition.
The forward pass is performed using sequence of images and sequence of high-level features.
```
import torch
from blinklinmult.models import BlinkLinMulT

# inputs with shapes: [(batch_size, time_dimension, channels, height, width), (batch_size, time_dimension, feature_dimension)]
x1 = torch.rand((8, 15, 3, 64, 64), device='cuda')
x2 = torch.rand((8, 15, 160), device='cuda')
model = BlinkLinMulT(input_dim=160, output_dim=1, weights='blinklinmult-union').cuda()
y_cls, y_seq = model([x1, x2])

# output shape: (batch_size, time_dimension, output_dimension)
assert y_seq.size() == torch.Size([8, 15, 1])
assert y_cls.size() == torch.Size([8, 1])
```

# Related projects

### exordium
Collection of preprocessing functions and deep learning methods. This repository contains revised codes for fine landmark detection (including face, eye region, iris and pupil landmarks), head pose estimation, and eye feature calculation.
* code: https://github.com/fodorad/exordium

### (2022) LinMulT
General-purpose Multimodal Transformer with Linear Complexity Attention Mechanism. This base model is further modified and trained for various tasks and datasets.
* code: https://github.com/fodorad/LinMulT

### (2022) PersonalityLinMulT for personality trait and sentiment estimation
LinMulT is trained for Big Five personality trait estimation using the First Impressions V2 dataset and sentiment estimation using the MOSI and MOSEI datasets.
* paper: Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures ([pdf](https://proceedings.mlr.press/v173/fodor22a/fodor22a.pf), [website](https://proceedings.mlr.press/v173/fodor22a.html))
* code: https://github.com/fodorad/PersonalityLinMulT (soon)


# Citation - BibTex
If you found our research helpful or influential please consider citing:

### (2023) BlinkLinMulT for blink presence detection and eye state recognition
```
@Article{fodor2023blinklinmult,
  title = {BlinkLinMulT: Transformer-Based Eye Blink Detection},
  author = {Fodor, Ádám and Fenech, Kristian and Lőrincz, András},
  journal = {Journal of Imaging},
  volume = {9},
  year = {2023},
  number = {10},
  article-number = {196},
  url = {https://www.mdpi.com/2313-433X/9/10/196},
  PubMedID = {37888303},
  ISSN = {2313-433X},
  DOI = {10.3390/jimaging9100196}
}
```

### (2022) LinMulT for personality trait and sentiment estimation
```
@InProceedings{pmlr-v173-fodor22a,
  title = {Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures},
  author = {Fodor, {\'A}d{\'a}m and Saboundji, Rachid R. and Jacques Junior, Julio C. S. and Escalera, Sergio and Gallardo-Pujol, David and L{\H{o}}rincz, Andr{\'a}s},
  booktitle = {Understanding Social Behavior in Dyadic and Small Group Interactions},
  pages = {218--241},
  year = {2022},
  editor = {Palmero, Cristina and Jacques Junior, Julio C. S. and Clapés, Albert and Guyon, Isabelle and Tu, Wei-Wei and Moeslund, Thomas B. and Escalera, Sergio},
  volume = {173},
  series = {Proceedings of Machine Learning Research},
  month = {16 Oct},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf},
  url = {https://proceedings.mlr.press/v173/fodor22a.html}
}
```

# What's next
* Preprocessed data hosting for easier reproduction and benchmarking
* Add train and test scripts for various databases

# Updates
* 1.0.0: Release version. Inference only with pre-trained models.

# Contact
* Ádám Fodor (foauaai@inf.elte.hu)