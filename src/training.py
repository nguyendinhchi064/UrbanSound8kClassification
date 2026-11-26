
import torch
import torch.nn as nn
from tqdm import tqdm


# Select CPU or GPU
def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    else:
        return torch.device("cpu")


# Training function
def train_one_epoch(model, loader, criterion,
                    optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training")

    for inputs, labels in progress:

        # Move data to GPU or CPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward propagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            accuracy=f"{100 * correct / total:.2f}%"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# Validation function
def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        progress = tqdm(loader, desc="Validation")

        for inputs, labels in progress:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = model(inputs)

            # Calculate validation loss
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            progress.set_postfix(
                loss=f"{running_loss / total:.4f}",
                accuracy=f"{100 * correct / total:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy
