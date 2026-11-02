"""
RegNetX_016 model adapted from regnetx_016_AdamW (2) - L0.3.py
"""
import torch
import torch.nn as nn
import timm


class RegNetX_016(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.0, drop_path_rate=0.0):
        super(RegNetX_016, self).__init__()
        torch.manual_seed(42)
        self.model = timm.create_model(
            'regnetx_016',
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            in_chans=1, # Changed from 3 to 1 for MNIST
            img_size=28, # Added for MNIST image size
        )

    def forward(self, x):
        return self.model(x)


def regnetx_016():
    return RegNetX_016(num_classes=10, drop_rate=0.0, drop_path_rate=0.0)


def test():
    net = regnetx_016()
    x = torch.randn(1, 1, 28, 28) # Changed from 3 to 1 channels and 32 to 28 size
    y = net(x)
    print(y.size())