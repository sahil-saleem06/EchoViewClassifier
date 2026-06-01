from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from .transforms import get_train_transforms, get_val_transforms


def get_dataset(root: str, split: str = "train", image_size: int = 224) -> ImageFolder:
    transform = get_train_transforms(image_size) if split == "train" else get_val_transforms(image_size)
    return ImageFolder(root=str(Path(root) / split), transform=transform)


def get_dataloader(
    root: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> DataLoader:
    dataset = get_dataset(root, split, image_size)
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
