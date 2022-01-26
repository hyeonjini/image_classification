from abc import ABC, abstractmethod
from typing import List, Union

import torch
import torch.nn as nn

class GeneratorAbstract(ABC):
    pass

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

