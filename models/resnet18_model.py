import torch.nn as nn
from torchvision import models

# This is a pretrained CNN model. Architecture RESNET18.
# LEARNING : Since this project has no GPU accesss and a limited number of training samples, we will use a pretrained model and only train the final layer.
# Default weights will be selected by the pretrained RESNET

def get_resnet18_model(num_classes=2, freeze_backbone=True):
    pre_trained_weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights = pre_trained_weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False # Learning : After freezing the backbone, we need to ensure that new gradietns are not calculated.

