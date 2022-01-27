from typing import Dict, Any, Tuple

import torch
from torchaudio import transforms
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder, VisionDataset

def create_dataloader(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Dataloader

    Args:
        config: yaml file path or dictionary of the data.

    Returns:
        train_dl
        valid_dl
        test_dl
    """
    train_dataset, val_dataset, test_dataset = get_dataset(

    )

    return get_dataloader(

    )


def get_dataset(
    data_path: str = "./sva/data",
    dataset_name: str = "CIFAR10",
    img_size: float = 32,
    val_ratio: float = 0.2,
    transform_train: str = "simple_augment_train",
    transform_test: str ="simple_aumgent_test",
    transform_train_params: Dict[str, int] = None,
    transform_test_params: Dict[str, int] = None,
) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:

    if not transform_train_params:
        transform_train_params = dict()
    if not transform_test_params:
        transform_test_params = dict()
    
    transform_train = getattr(
        __import__("arc.augmentation.policies", fromlist=[""])
    )

def get_dataloader(
    train_dataset: VisionDataset,
    val_dataset: VisionDataset,
    test_dataset: VisionDataset,
    batch_size: int
):
    pass
