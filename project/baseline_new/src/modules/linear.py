from typing import Union

import torch
import torch.nn as nn

from src.modules.base_generator import GeneratorAbstract
from src.utils.torch_utils import Activation

class Linear(nn.Module):
    
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        activation: Union[str, None]
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.activation = Activation(activation)()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

class LinearGenerator(GeneratorAbstract):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def out_channel(self) -> int:
        return self.args[0]
    
    def __call__(self, repeat: int = 1):
        act = self.args[1] if len(self.args) > 1 else None

        return self._get_module(
            Linear(self.in_channel, self.out_channel, activation=act)
        )