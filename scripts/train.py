import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scripts.data_utils import train_transforms, validate_transforms

train_data_set = ImageFolder("data/TrainSet", transform=train_transforms)
validation_data_set = ImageFolder("data/ValidationSet", transform=validate_transforms)

train_data_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)
validatopn_data_loader = DataLoader(validation_data_set, batch_size=32, shuffle=False)

# training loop ......