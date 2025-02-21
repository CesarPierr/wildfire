import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from vision_transformer import VisionTransformer, vit_small

def create_baseline_model():
    """
    Returns a simple CNN baseline model.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        #average the whole channels
        nn.AdaptiveAvgPool2d(1),

        nn.Flatten(),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    return model

class VitWithHead(nn.Module):
    def __init__(self, backbone, num_classes=2, freeze_backbone=True, big_head=False):
        super().__init__()
        self.backbone = backbone
        #delete the original head
        del self.backbone.head
        self.head = nn.Sequential(
            nn.Linear(384, num_classes)
        )
        if big_head:
            self.head = nn.Sequential(
                nn.Linear(384, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
            

        #freeze the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def create_vit_model(checkpoint_path=None, 
                     num_classes=2,  freeze_backbone=True, big_head=False) :
    """
    Loads a ViT tiny model, optionally from a DINO checkpoint.
    
    Args:
        checkpoint_path (str): Path to the DINO checkpoint (.pth) that contains
                               { 'student': <state_dict>, 'teacher': ..., ... }.
        num_classes (int): Number of classes for the new classification head.
        replace_head (bool): If True, completely replace the head with a new one.
                             If False, stack a new Linear after the existing one.
    
    Returns:
        model (nn.Module): The ViT model with updated head.
    """
    
    # Create the model with the same defaults DINO used for vit_tiny
    model = vit_small(
        patch_size=16,
        drop_path_rate=0.1,  # default in the training script
    )
    
    # If a checkpoint path is provided, load the "student" state dict
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if "student" in checkpoint:
            # The keys you actually want are in checkpoint["student"]
            state_dict = checkpoint["student"]
        else:
            # In case you're using a raw state dict (not the full DINO checkpoint dict)
            state_dict = checkpoint
        
        # Load the model weights
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint")

    model = VitWithHead(model, num_classes, freeze_backbone, big_head)
    return model

def create_swin_transformer(checkpoint_path=None, num_classes=2, freeze_backbone=True, big_head=False):

    model = torchvision.models.swin_transformer.swin_v2_b(weights=None)
    
    if checkpoint_path is not None:
        full_state_dict = torch.load(checkpoint_path, map_location='cpu')
        swin_prefix = 'backbone.backbone.'
        filtered_dict = {
            k[len(swin_prefix):]: v
            for k, v in full_state_dict.items()
            if k.startswith(swin_prefix)
        }
        model.load_state_dict(filtered_dict, strict=False)
    
    # Replace the classifier head
    if big_head:
        model.head = nn.Sequential(
            nn.Linear(model.head.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    else :
        model.head = nn.Linear(model.head.in_features, num_classes)

    if freeze_backbone:
        # Freeze all layers except the new head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    return model
