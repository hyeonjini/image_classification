from typing import Any, Dict, Union

import numpy as np
import yaml
from torchvision.datasets import ImageFolder, VisionDataset

def read_yaml(cfg: Union[str, Dict[str, Any]]):

    if not isinstance(cfg, dict): # if 'cfg' is not dict data, load config using yaml module.
        with open(cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
    else:
        config = cfg
    
    return config