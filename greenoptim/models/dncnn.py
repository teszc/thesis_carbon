import torch.nn as nn


class DnCNN(nn.Module):
    """
    Lightweight DnCNN denoising network.
    Three convolutional layers with ReLU activations.
    Channels are configurable for hyperparameter search.
    Input/output: 3-channel RGB images.
    """

    def __init__(self, channels: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)