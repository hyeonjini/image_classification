"""PyTorch transforms"""

from torch import square
import torchvision.transforms as transforms

DATASET_NORMALIZE_INFO = {
    "CIFAR10": {"MEAN": (0.4914, 0.4822, 0.4465), "STD": (0.2470, 0.2435, 0.2616)},
    "CIFAR100": {"MEAN": (0.5071, 0.4865, 0.4409), "STD": (0.2673, 0.2564, 0.2762)},
    "IMAGENET":{},
    "MNIST":{},
    "FashionMNIST":{}
}

def simple_augment_train(
    dataset: str,
    img_size: float = 32
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))),
        transforms.RandomResizedCrop(size=img_size, ratio=(0.75, 1.0, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            DATASET_NORMALIZE_INFO[dataset]["MEAN"],
            DATASET_NORMALIZE_INFO[dataset]["STD"],   
        )
    ])

def simple_augment_test(
    dataset: str,
    img_size: float = 32
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            DATASET_NORMALIZE_INFO[dataset]["MEAN"],
            DATASET_NORMALIZE_INFO[dataset]["STD"],
        ),
    ])

def randaugment_train():
    pass