import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------
# Data Preparation Utils
# ----------------------
def get_all_datasets(
    pretrain_path: Path,
    val_path: Path,
    test_path: Path,
    transforms_dict: dict
):
    """
    Creates pretrain, train, validation, and test datasets.

    1. Uses ImageFolder on `pretrain_path` for 'pretrain_dataset'.
    2. Uses ImageFolder on `val_path` for splitting into 'train_dataset' and 'val_dataset'.
    3. Uses ImageFolder on `test_path` for 'test_dataset'.

    Args:
        pretrain_path (Path): Path to pretrain folder.
        val_path (Path): Path to valid folder.
        test_path (Path): Path to test folder.
        transforms_dict (dict): Dictionary containing transforms for 'pretrain', 'valid', and 'test'.

    Returns:
        (Dataset, Dataset, Dataset, Dataset):
            Tuple of pretrain_dataset, train_dataset, val_dataset, test_dataset.
    """

    pretrain_dataset = datasets.ImageFolder(pretrain_path, transform=transforms_dict['pretrain'])
    val_dataset_full = datasets.ImageFolder(val_path, transform=transforms_dict['valid'])
    test_dataset = datasets.ImageFolder(test_path, transform=transforms_dict['test'])

    # Split validation dataset into train/val
    train_idx, validation_idx = train_test_split(
        np.arange(len(val_dataset_full)),
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=val_dataset_full.targets
    )

    train_dataset = Subset(val_dataset_full, train_idx)
    val_dataset = Subset(val_dataset_full, validation_idx)

    return pretrain_dataset, train_dataset, val_dataset, test_dataset

# -------------------------
# Training & Validation Loop
# -------------------------
def train_one_epoch(model, optimizer, data_loader, loss_fn, device, use_amp=True):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        data_loader (DataLoader): Data loader for training data.
        loss_fn: Loss function.
        device: Torch device (CPU/GPU).
        use_amp (bool): Whether to use Automatic Mixed Precision.

    Returns:
        (list[float]): List of loss values for each batch in the epoch.
    """
    model.train()
    losses = []

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for x, y in tqdm(data_loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
    return losses


def validate(model, data_loader, loss_fn, device, use_amp=True):
    """
    Validate model on given dataset.

    Args:
        model: PyTorch model.
        data_loader (DataLoader): Data loader for validation or test data.
        loss_fn: Loss function.
        device: Torch device (CPU/GPU).
        use_amp (bool): Whether to use Automatic Mixed Precision.

    Returns:
        (tuple):
            losses (list[float]): List of loss values for each batch in the epoch.
            correct_predictions (int): Total number of correct predictions.
    """
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Validating", leave=False):
            x = x.to(device)
            y = y.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

            losses.append(loss.item())
            correct_predictions += (y_hat.argmax(dim=1) == y).sum().item()

    return losses, correct_predictions


def plot_curves(train_losses, val_losses, val_accuracies, save_path=None, title_suffix=""):
    """
    Plot training/validation loss curves and validation accuracy.

    Args:
        train_losses (list[float]): Training losses over all epochs (flattened or per epoch).
        val_losses (list[float]): Validation losses over all epochs (flattened or per epoch).
        val_accuracies (list[float]): Validation accuracy values.
        save_path (str): Path to save the resulting figure (optional).
        title_suffix (str): Suffix to append to plot titles for clarity.
    """
    # If you have multiple loss entries per epoch in train_losses or val_losses, you might need to reshape
    # or average them per epoch. For simplicity, assume they're already aligned by epoch.

    epochs_range = range(len(val_accuracies))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title(f'Training and Validation Loss {title_suffix}')
    axes[0].legend()

    # Plot Accuracy
    axes[1].plot(epochs_range, val_accuracies, label='Val Accuracy')
    axes[1].set_title(f'Validation Accuracy {title_suffix}')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_checkpoint(
    save_path,
    model,
    optimizer,
    epoch,
    val_accuracy,
    amp_scaler=None
):
    """
    Save model checkpoint.

    Args:
        save_path (str): File path to save checkpoint.
        model (nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer state.
        epoch (int): Current epoch number.
        val_accuracy (float): Current validation accuracy.
        amp_scaler (torch.cuda.amp.GradScaler, optional): GradScaler for AMP if used.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
    }
    if amp_scaler is not None:
        checkpoint['amp_scaler'] = amp_scaler.state_dict()
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    amp_scaler=None
):
    """
    Load model checkpoint.

    Args:
        checkpoint_path (str): File path of the checkpoint.
        model (nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to restore state.
        amp_scaler (torch.cuda.amp.GradScaler, optional): GradScaler for AMP if used.

    Returns:
        (dict): The loaded checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if amp_scaler is not None and 'amp_scaler' in checkpoint:
        amp_scaler.load_state_dict(checkpoint['amp_scaler'])
    return checkpoint
