import argparse
import sys
import time
from collections import Counter
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data_utils.data_transforms import train_transforms, validate_transforms

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a runway classifir on the CPU")
    parser.add_argument("--model", choices = ["resnet18", "vit"], default = "resnet18")
    parser.add_argument("--vit-variant", choices = ["b16", "l14"], default = "b16")
    parser.add_argument("--epochs", type = int, default = 10)
    parser.add_argument("--learning-rate", type = float, default = 1e-3)
    parser.add_argument("--batch-size", type=int, default=None, help="Default: 16 for resnet18, 4 for vit")
    parser.add_argument("--num-workers", type = int, default = 2)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def get_model(model_name: str, num_classes: int) -> nn.Module:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    if model_name == "resnet18":
        from models.resnet18_model import get_resnet18_model
        return get_resnet18_model(num_classes = 2, freeze_backbone = True)

def main() -> None:
    # Reproducibility
    torch.manual_seed(42)

    device = torch.device("cpu")

    project_root = Path(__file__).resolve().parents[1]
    train_dir = project_root / "data" / "TrainSet"
    validation_dir = project_root / "data" / "ValidationSet"

    # Datasets
    train_data_set = ImageFolder(str(train_dir), transform=train_transforms)
    validation_data_set = ImageFolder(str(validation_dir), transform=validate_transforms)

    # Check class balance
    print("Train distribution:", Counter(train_data_set.targets))

    # DataLoaders (CPU optimized)
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )

    validation_data_loader = DataLoader(
        validation_data_set,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    # Avoid lint warnings while preserving loader initialization for future training.
    _ = (device, validation_data_loader)

    # Debug check
    images, labels = next(iter(train_data_loader))
    print("Image shape:", images.shape)
    print("Labels shape:", labels.shape)


if __name__ == "__main__":
    main()