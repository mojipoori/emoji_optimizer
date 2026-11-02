"""
Inception Next Tiny model adapted from inception_016_AdamW (2) - d0.5 - L0almost.py
"""
import torch
import torch.nn as nn
import timm
from functools import partial


class InceptionNextTiny(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.0, drop_path_rate=0.0):
        super(InceptionNextTiny, self).__init__()
        torch.manual_seed(42)
        self.model = timm.create_model(
            'inception_next_tiny',
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            in_chans=1, # Changed from 3 to 1 for MNIST
            img_size=28, # Added for MNIST image size
            depths=(2, 2, 4, 2),
            dims=(64, 128, 256, 512),
            mlp_ratios=(2, 2, 2, 2),
            token_mixers=partial(
                timm.models.inception_next.InceptionDWConv2d,
                square_kernel_size=3,
                band_kernel_size=5,
                branch_ratio=0.25
            ),
        )

        # Customize stem
        self.model.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1), # Changed input channels to 1
            nn.BatchNorm2d(64)
        )

        # Customize downsample layers
        self.model.stages[0].downsample = nn.Identity()  # Stage 1: No downsampling
        self.model.stages[1].downsample = nn.Sequential(  # Stage 2: Downsampling
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, stride=2)
        )
        self.model.stages[2].downsample = nn.Sequential(  # Stage 3: Channel change, no stride
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=1, stride=1)
        )
        self.model.stages[3].downsample = nn.Sequential(  # Stage 4: Channel change, no stride
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=1, stride=1)
        )

        # Customize head
        self.model.head = timm.models.inception_next.MlpClassifierHead(
            in_features=self.model.num_features,
            num_classes=num_classes,
            pool_type='avg',
            mlp_ratio=2,
            drop=drop_rate
        )

    def forward(self, x):
        return self.model(x)


def inception_next_tiny():
    return InceptionNextTiny(num_classes=10, drop_rate=0.0, drop_path_rate=0.0)


def test():
    net = inception_next_tiny()
    x = torch.randn(1, 1, 28, 28) # Changed from 3 to 1 channels and 32 to 28 size
    y = net(x)
    print(y.size())

# test()