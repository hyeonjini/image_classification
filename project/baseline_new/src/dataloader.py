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
        data_path=config["DATA_PATH"],
        dataset_name=config["DATASET"],
        img_size=config["IMG_SIZE"],
        val_ratio=config["VAL_RATIO"],
        transform_train=config["AUG_TRAIN"],
        transform_test=config["AUG_TEST"],
        transform_train_params=config["AUG_TRAIN_PARAMS"],
        transform_test_params=config["AUG_TEST_PARAMS"],
    )

    return get_dataloader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config["BATCH_SIZE"],
    )


def get_dataset(
    data_path: str = "./save/data",
    dataset_name: str = "CIFAR10",
    img_size: float = 32,
    val_ratio: float = 0.2,
    transform_train: str = "simple_augment_train",
    transform_test: str ="simple_augment_test",
    transform_train_params: Dict[str, int] = None,
    transform_test_params: Dict[str, int] = None,
) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:

    # if there is no transform params, initialized by default dictionary.
    if not transform_train_params:
        transform_train_params = dict()
    if not transform_test_params:
        transform_test_params = dict()
    
    # preprocessing policies
    # return torchvision.transforms.Compose 
    transform_train = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_train,
    )(dataset=dataset_name, img_size=img_size, **transform_train_params)

    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_test,
    )(dataset=dataset_name, img_size=img_size, **transform_test_params)

    label_weights = None

    Dataset = getattr(
        __import__("torchvision.datasets", fromlist=[""]), dataset_name
    )

    train_dataset = Dataset(
        root=data_path,
        train=True,
        download=True,
        transform=transform_train
    )
    # split train dataset to train and val.
    train_length = int(len(train_dataset) * (1.0-val_ratio))
    train_dataset, val_dataset = random_split(
        train_dataset, [train_length, len(train_dataset) - train_length]
    )

    test_dataset = Dataset(
        root=data_path, train=False, download=False, transform=transform_test
    )

    return train_dataset, val_dataset, test_dataset

def get_dataloader(
    train_dataset: VisionDataset,
    val_dataset: VisionDataset,
    test_dataset: VisionDataset,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=batch_size,
        num_workers=5
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=batch_size,
        num_workers=5
    )
    
    return train_loader, val_loader, test_loader
