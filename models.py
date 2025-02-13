import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


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


class PatchEmbed(nn.Module):
    """Splits the image into patches and projects to a vector (i.e. a linear embedding)."""
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=1536):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=14,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=1536,   # for ViT-g/14 from the paper
                 depth=40,
                 num_heads=24,
                 mlp_ratio=4.0,
                 drop_rate=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embeddings for [CLS] + patches.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Use the built-in TransformerEncoderLayer and TransformerEncoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation="gelu",  
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Optionally initialize the classification head.
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        # Prepend the class token.
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        # Add positional embeddings.
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Pass through the transformer encoder.
        x = self.encoder(x)  # [B, 1 + num_patches, embed_dim]
        x = self.norm(x)
        # Use the class token representation for classification.
        cls_out = x[:, 0]
        x = self.head(cls_out)
        return x
    
def create_vit_model(checkpoint_path=None, img_size=330,patch_size=16,in_chans=3,num_classes=1000,embed_dim=768,depth=12,num_heads=12,mlp_ratio=4.0,drop_rate=0.0,replace_head=True):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate
    )
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
