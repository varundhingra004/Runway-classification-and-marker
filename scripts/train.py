import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scripts.data_utils.data_transforms import train_transforms, validate_transforms

train_data_set = ImageFolder("data/TrainSet", transform=train_transforms)
validation_data_set = ImageFolder("data/ValidationSet", transform=validate_transforms)

train_data_loader = DataLoader(train_data_set, batch_size=32, shuffle=True, num_workers=2)
validation_data_loader = DataLoader(validation_data_set, batch_size=32, shuffle=False, num_workers=2)

# Debug check
images, labels = next(iter(train_data_loader))
print(images.shape)
print(labels.shape)

# training loop ......