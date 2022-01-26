
from typing import Dict, List, Type, Union
import yaml

import torch
import torch.nn as nn

from src.modules import ModuleGenerator

class Model(nn.Module):
    """Base model class"""

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
        self.model_parser = ModelParser(cfg=cfg, verbose=verbose)
        self.model = self.model_parser.model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_one(x)
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
        """Log"""
        if self.verbose:
            print(msg)

    def _parse_model(self) -> nn.Sequential:
        """Parse model"""

        layers: List[nn.Module] = []
        log: str = (

        )

        in_channel = self.in_channel

        for i, (repeat, module, args) in enumerate(self.model_cfg):
            repeat = (
                max(round(repeat * self.depth_multiply), 1) if repeat > 1 else repeat
            )
            # Module generator 
            module_generator = ModuleGenerator(module, in_channel)(
                *args,
                width_multiply=self.width_multiply,
            )

            m = module_generator(repeat=repeat)
            
            layers.append(m)

            # modifying in_channel for next layer's input
            in_channel = module_generator.out_channel

            # add log
            log = (
                f"{i:3d} | {repeat:3d} | "
                f"{m.n_params:10,d} | {m.type:>15} | {str(args):>20} | "
                f"{str(module_generator.in_channel):12}"
                f"{str(module_generator.out_channel):>13}"
            )
            self.log(log)
        
        parsed_model = nn.Sequential(*layers)
        n_param = sum([x.numel() for x in parsed_model.parameters()])
        n_grad = sum([x.numel() for x in parsed_model.parameters() if x.requires_grad])

        self.log(
            f"Model Summary: {len(list(parsed_model.modules())):,d}"
            f"layers, {n_param:,d} parametsers, {n_grad:,d} gradients"
        )

        return parsed_model


