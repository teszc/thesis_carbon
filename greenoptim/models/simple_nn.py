import torch.nn as nn


class SimpleNN(nn.Module):
    """
    Simple feedforward neural network for MNIST classification.
    Two linear layers with ReLU activation.
    Hidden size is configurable for hyperparameter search.
    """

    def __init__(self, hidden_size: int = 128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        return self.fc(x)