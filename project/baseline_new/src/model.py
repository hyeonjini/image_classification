
from typing import Dict, List, Type, Union
import yaml

import torch
import torch.nn as nn

from src.modules import ModuleGenerator

class Model(nn.Module):

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]] = "./model_configs/show_case.yaml",
        verbose: bool = False,
    ) -> None:
        """Parse model from the model config file.

        Args:
            cfg (Union[str, Dict[str, Type]], optional): yaml file path or dictionary of the model.
            verbose (bool, optional): print information.
        """
        super().__init__()

class ModelParser:
    """Generate PyTorch model from yaml file."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]] = "./model_config/show_case.yaml",
        verbose: bool = False,
    ) -> None:
        """Generate PyTorch model from yaml file.

        Args:
            cfg: model config file path or dictionary of the model.
            verbose: print information.
        """

        self.verbose = verbose
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            with open(cfg) as f:
                self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        self.in_channel = self.cfg['input_channel']

        self.depth_multiply = self.cfg["depth_multiple"]
        self.width_multiply = self.cfg["width_multiple"]

        self.model_cfg: List[Union[int, str, float]] = self.cfg["backbone"]

        self.model = self._parse_model()

    def log(self, msg:str):
        pass

    def _parse_model(self) -> nn.Sequential:
        """Parse model"""

        layers: List[nn.Module] = []

        # make a log message

        in_channel = self.in_channel

        for i, (repeat, module, args) in enumerate(self.model_cfg):
            repeat = (
                max(round(repeat * self.depth_multiply), 1) if repeat > 1 else repeat
            )
            moduel_generator = ModuleGenerator()