import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scripts.data_utils.data_transforms import train_transforms, validate_transforms
from collections import Counter

# Reproducibility
torch.manual_seed(42)

device = torch.device("cpu")

# Datasets
train_data_set = ImageFolder("data/TrainSet", transform=train_transforms)
validation_data_set = ImageFolder("data/ValidationSet", transform=validate_transforms)

# Check class balance
print("Train distribution:", Counter(train_data_set.targets))

# DataLoaders (CPU optimized)
train_data_loader = DataLoader(
    train_data_set,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    persistent_workers=True
)

validation_data_loader = DataLoader(
    validation_data_set,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    persistent_workers=True
)

# Debug check
images, labels = next(iter(train_data_loader))
print("Image shape:", images.shape)
print("Labels shape:", labels.shape)