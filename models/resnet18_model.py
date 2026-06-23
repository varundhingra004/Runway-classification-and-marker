import torch.nn as nn
from torchvision import models

# LEARNING : Torch is the core tensor and deep learning framework. Torchvision is a library built on top or torch for Computer Vision tasks.

# This is a pretrained CNN model. Architecture RESNET18.
# LEARNING : Since this project has no GPU accesss and a limited number of training samples, we will use a pretrained model and only train the final layer.
# Default weights will be selected by the pretrained RESNET

def get_resnet18_model(num_classes=2, freeze_backbone=True):
    pre_trained_weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights = pre_trained_weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False 
            # Learning : After freezing the backbone, we need to ensure that new gradients are not calculated.
            # Learning : parameters() returns an iterator over the parameter tensor. The parameter tensor is in the torch.nn module.i
            # Learning : requires_grad is a parameter of the Parameters tensor that indicates if the parameter requires a gradient.
        
        # Unfreezing the final fully connected layer. 
        # Learning : models.fc is the final fully connected layer. This final output layer  is found in all CNNs.
        for param in model.fc.parameters():
            param.requires_grad = True
        
        in_features = model.fc.in_features # Input features for the fully conncted layer
        model.fc = nn.Linear(in_features, num_classes) # Replacing the model's original fully connected layer wuth a new linear layer containing the original input features and number of classes as output.
    
    return model