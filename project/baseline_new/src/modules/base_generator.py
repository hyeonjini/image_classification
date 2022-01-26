from abc import ABC, abstractmethod
from typing import List, Union

import torch
import torch.nn as nn

class GeneratorAbstract(ABC): # abstract base class
    """Abstract class for Module Generator. """
    
    CHANNEL_DIVISOR: int = 8

    def __init__(
        self,
        in_channel: int,
        *args,
        from_idx:Union[int, List[int]] = -1,
        width_multiply: float = 1.0,
    ):
        """[summary]

        Args:
            in_channel (int): [description]
            from_idx (Union[int, List[int]], optional): [description]. Defaults to -1.
            width_multiply (float, optional): [description]. Defaults to 1.0.
        """
        self.args = tuple(args)
        self.from_idx = from_idx
        self.in_channel = in_channel
        self.width_multiply = width_multiply
    
    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Generator", "")
    
    def _get_module(self, module: Union[nn.Module, List[nn.Module]]) -> nn.Module:
        """Get module from __cal__ function."""
        
        if isinstance(module, list):
            module = nn.Sequential(*module)
        
        module.n_params = sum([x.numel() for x in module.parameters()])
        module.type = self.name

        return module
    
    # @classmethod
    # def _get_divisible_channel(cls, n_channel: int) -> int:
    #     return make_divisible(n_channel, divisor=cls.CHANNEL_DIVISOR)

    @property
    @abstractmethod
    def out_channel(self) -> int:
        """Out channel of the module."""

    @abstractmethod
    def __call__(self, repeat: int=1):
        """Returns nn.Module component"""

class ModuleGenerator:

    def __init__(self, module_name: str, in_channel: int):
        """Generate a module using {module_name}

        Args:
            module_name: Each module must have '{module_name}Generator'
            in_channel (int): [description]
        """
        self.module_name = module_name
        self.in_channel = in_channel
    
    def __call__(self, *args, **kwargs):
        return getattr(
            __import__("src.modules", fromlist=[""]),
            f"{self.module_name}Generator",
        )(self.in_channel, *args, **kwargs)

