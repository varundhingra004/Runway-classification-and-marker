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
    '''

    Since there are multiple models to choose from, the model and the corresponding parametrs
    will be configured in the comand line via the parser.

    '''

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

    '''
    
    Based on the parsed arguments in the command line, the appropriate model is chosen
    and the corrsponding arguments are entered into the model.

    '''

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    if model_name == "resnet18":
        from models.resnet18_model import get_resnet18_model
        return get_resnet18_model(num_classes = 2, freeze_backbone = True)

def train_one_epoch(model, loader, criterion, optimizer, device):
    '''

    Train the model for a single epoch only. This function will be called multiple times by the main().

    Step 1) Set the model to train.
    Step 2) Loop over the batches from train_loader
    Step 3) Clear all old gradients
    Step 4) Run the forward pass, loss, backeard pass, optimizer step
    Step 5) Track average loss and accuracy fir that epoch
    Step 6) Return those metrics to main() for printing and checkpointing

    '''

    # step 1) Set the model into training mode.
    # In  this mode, certail layers will act differently
    # during the reaining phase. Like droput (deactivated for training), bacthnorm, etc.
    model.train() 
    running_loss = 0.0
    correct = 0
    total = 0

    # Step 2) Loop over the batches from the train_loader
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # Clear all gradients
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward()
        optimizer.step()

        # Need to understand !!!!
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    '''
        Run the trained model on the Validation set and check the accuracy.
        This means that there is no gradient tracking(hence the @torch.no_grad decorator)
        and no backpropogation.
    '''
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Need to understand !!!!
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total

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

    args = parse_args()
    model = get_model(args.model, num_classes = 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # Enter the training loop. Call one_epoch multiple times
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_data_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, validation_data_loader, criterion, device)
        print(f"Epoch {epoch} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")
        # save best checkpoint

if __name__ == "__main__":
    main()