import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data_utils.data_transforms import train_transforms, validate_transforms
from collections import Counter
from pathlib import Path

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