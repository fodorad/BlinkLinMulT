from .BlinkLinMulT import (AbstractModule,
                           DenseNet121,
                           BlinkLinT,
                           LinT,
                           BlinkLinMulT,
                           preprocess_eye_fcn)

from .backbone import (EyeNet,
                       ResNet50,
                       Dense)

SEQUENCE_MODELS = {
    "lint": LinT,
    "blinklint": BlinkLinT,
    "blinklinmult": BlinkLinMulT,
}

BACKBONE_MODELS = {
    "dense": Dense,
    "eyenet": EyeNet,
    "resnet50": ResNet50,
    "densenet121": DenseNet121
}