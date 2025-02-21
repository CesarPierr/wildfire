import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import ImageFile

# Import wandb
import wandb

from utils import (
    get_all_datasets,
    train_one_epoch,
    validate,
    plot_curves,
    save_checkpoint,
    load_checkpoint
)
from models import create_baseline_model, create_swin_transformer, create_vit_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    parser = argparse.ArgumentParser(description="Train a wildfire prediction model.")
    parser.add_argument("--model", type=str, default="vit", choices=["baseline", "swin", "vit"], help="Which model to train.")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--data_dir", type=str, default="./wildfire-prediction-dataset", help="Root directory for dataset.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save checkpoints and plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--big_head", action="store_true", help="Use a larger head.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze the backbone")
    args = parser.parse_args()

    # -------------------
    # Initialize WandB
    # -------------------
    wandb.init(
        project="wildfire-prediction-2",  # Change to your W&B project name
        config={
            "model": args.model,
            "checkpoint": args.checkpoint,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "big_head" : args.big_head,
            "freeze_backbone": args.freeze_backbone
        },
        name = f"{args.model}_epochs_{args.epochs}_has_checkpoint_{args.checkpoint}_big_head_{args.big_head}_freeze_backbone_{args.freeze_backbone}"
    )

    # -------------------
    # Prepare directories
    # -------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # -------------------
    # Prepare Data
    # -------------------
    dataset_path = Path(args.data_dir)
    pretrain_path = dataset_path / 'train'
    val_path = dataset_path / 'valid'
    test_path = dataset_path / 'test'

    data_transforms = {
        'pretrain': transforms.Compose([transforms.ToTensor()]),
        'valid':    transforms.Compose([transforms.ToTensor()]),
        'test':     transforms.Compose([transforms.ToTensor()]),
    }

    _, train_dataset, val_dataset, test_dataset = get_all_datasets(
        pretrain_path=pretrain_path,
        val_path=val_path,
        test_path=test_path,
        transforms_dict=data_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=8)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=8)

    # -------------------
    # Create Model
    # -------------------
    if args.model == "baseline":
        model = create_baseline_model()
        best_model_path = output_dir / "best_baseline_model.pth"
    elif args.model == "swin":
        model = create_swin_transformer(
            checkpoint_path=args.checkpoint,
            num_classes=2,
            big_head=args.big_head,
            freeze_backbone=args.freeze_backbone
        )
        best_model_path = output_dir / f"best_swin_model_fb-{args.freeze_backbone}_bh-{args.big_head}_checkpoint-{False if args.checkpoint is None else True}.pth"
    elif args.model == "vit":
        model = create_vit_model(
            checkpoint_path=args.checkpoint,
            num_classes=2,
            big_head=args.big_head,
            freeze_backbone=args.freeze_backbone
        )
        best_model_path = output_dir / f"best_vit_model_fb-{args.freeze_backbone}_bh-{args.big_head}_checkpoint-{False if args.checkpoint is None else True}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # -------------------
    # Loss & Optimizer
    # -------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # -------------------
    # Training Loop
    # -------------------
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")

        # Train
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            loss_fn=criterion,
            device=device,
        )
        epoch_train_loss = np.mean(train_loss)

        # Validate
        val_loss, correct_predictions = validate(
            model=model,
            data_loader=val_loader,
            loss_fn=criterion,
            device=device,
        )
        epoch_val_loss = np.mean(val_loss)
        val_accuracy = correct_predictions / len(val_dataset)

        # Store metrics for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # -------------------
        # Log metrics to W&B
        # -------------------
        wandb.log({
            "train_loss": epoch_train_loss,
            "val_loss":   epoch_val_loss,
            "val_accuracy": val_accuracy
        }, step=epoch)

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                save_path=str(best_model_path),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=val_accuracy
            )

    # -------------------
    # Load and Evaluate Best Model
    # -------------------
    checkpoint = load_checkpoint(str(best_model_path), model, optimizer=None)
    print(
        f"Loaded best model from epoch {checkpoint['epoch']+1} "
        f"with validation accuracy: {checkpoint['val_accuracy']:.4f}"
    )

    # Plot and save training curves
    plot_filename = output_dir / f"{args.model}_training_plot.png"
    plot_curves(train_losses, val_losses, val_accuracies, save_path=str(plot_filename), title_suffix=f"({args.model})")

    # -------------------
    # Test
    # -------------------
    test_loss, test_correct = validate(model, test_loader, criterion, device=device)
    test_accuracy = test_correct / len(test_dataset)

    print(
        f"\nTest Loss: {np.mean(test_loss):.4f} | "
        f"Test Accuracy: {test_accuracy:.4f}"
    )

    # Log final test metrics to W&B
    wandb.log({
        "test_loss":      np.mean(test_loss),
        "test_accuracy":  test_accuracy
    })

    # Optional: Save final model or attach best model to W&B as an artifact
    # wandb.save(str(best_model_path))

if __name__ == "__main__":
    main()
