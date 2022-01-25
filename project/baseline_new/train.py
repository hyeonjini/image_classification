"""
Baseline for train
- Author: HyeonJin Choi
- Contact: choihj94@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

import yaml

from src.utils.common import read_yaml

def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16:bool,
    device:torch.device
) -> Tuple[float, float, float]:
    pass

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # Model config file (YAML)
    parser.add_argument("--model_config", type=str, default="configs/model/example.yaml", help="")
    
    # Data and hyperparameter config file (YAML)
    parser.add_argument("--data_config", type=str, default="configs/data/mnist.yaml", help="")

    args = parser.parse_args()
    print(args)

    # Load config from yaml file
    model_config = read_yaml(cfg=args.model_config)
    data_config = read_yaml(cfg=args.data_config)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", "latest"))

    if os.path.exists(log_dir):
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%s")
        os.rename(log_dir, new_log_dir)
    
    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device
    )
