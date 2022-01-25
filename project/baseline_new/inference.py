"""
Baseline for train
- Author: HyeonJin Choi
- Contact: choihj94@gmail.com
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

import yaml

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    