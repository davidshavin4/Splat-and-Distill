# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_segmentor, init_segmentor
from .test import multi_gpu_test, single_gpu_test
from .train import train_segmentor

__all__ = [
    "train_segmentor",
    "init_segmentor",
    "inference_segmentor",
    "multi_gpu_test",
    "single_gpu_test",
]
