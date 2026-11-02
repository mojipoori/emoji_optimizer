"""
GCViT-XXTiny model adapted from gcvit_AdamW (2) - L0.3.py
"""
import torch
import torch.nn as nn
import timm


class GCViTXXTiny(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.0, drop_path_rate=0.0):
        super(GCViTXXTiny, self).__init__()
        torch.manual_seed(42)
        self.model = timm.create_model(
            'gcvit_xxtiny',
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            img_size=(28, 28), # Changed from 32 to 28 for MNIST
            in_chans=1, # Added for single channel MNIST images
            window_size=((4, 4), (2, 2), (1, 1), (1, 1)),
            proj_drop_rate=0.1,
            attn_drop_rate=0.1,
            mlp_ratio=4.0,
            layer_scale=1e-5,
        )

    def forward(self, x):
        return self.model(x)


def gcvit_xxtiny():
    return GCViTXXTiny(num_classes=10, drop_rate=0.0, drop_path_rate=0.0)


def test():
    net = gcvit_xxtiny()
    x = torch.randn(1, 1, 28, 28) # Changed from 3 to 1 channels and 32 to 28 size
    y = net(x)
    print(y.size())

# test()