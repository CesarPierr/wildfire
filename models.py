import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from vision_transformer import VisionTransformer

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
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    return model


def create_vit_model(checkpoint_path=None, num_classes=2,replace_head=True):
    model = VisionTransformer()
    
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    if replace_head:
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        model.head = nn.Sequential(
            model.head,
            nn.ReLU(),
            nn.Linear(model.head.out_features, num_classes)
        )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    return model

def create_swin_transformer(checkpoint_path=None, num_classes=2, replace_head = True) :

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
    if replace_head : 
        model.head = nn.Linear(model.head.in_features, num_classes)
    else :
        #keep the original head and add a new mlp layer after
        model.head = nn.Sequential(
            model.head,
            nn.ReLU(),
            nn.Linear(model.head.out_features, num_classes)
        )

    # Freeze all layers except the new head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    return model
