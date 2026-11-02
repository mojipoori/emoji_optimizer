import torch
import torch.nn as nn
import timm

class MobileViT(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.0, drop_path_rate=0.0):
        super(MobileViT, self).__init__()
        torch.manual_seed(42)
        self.model = timm.create_model(
            'mobilevit_s',
            pretrained=False,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            img_size=28,  # Changed from 32 to 28 for MNIST
            in_chans=1 # Added for single channel MNIST images
        )

    def forward(self, x):
        return self.model(x)


def mobilevit_s():
    return MobileViT(num_classes=10, drop_rate=0.0, drop_path_rate=0.0)


def test():
    net = mobilevit_s()
    x = torch.randn(1, 1, 28, 28) # Changed from 3 to 1 channels and 32 to 28 size
    y = net(x)
    print(y.size())

# test()