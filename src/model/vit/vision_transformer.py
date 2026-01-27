from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch.nn as nn

@dataclass
class VisionTransformerBaseCfg:
    name: str
    embed_dim: int  


class VisionTransformerBase(nn.Module, ABC):

    @abstractmethod
    def training_step(self, *args, **kwargs):
        """
        Perform a single training step.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def extract_features(self, *args, **kwargs):
        """
        Extract features from input data.
        Must be implemented by subclasses.
        """
        pass