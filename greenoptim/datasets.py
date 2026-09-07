import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Standard transform for DnCNN (resize to 64x64)
DNCNN_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Standard transform for MNIST
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])


class NoisyImageDataset(Dataset):
    """
    Wraps an ImageFolder dataset and adds Gaussian noise
    to each image on-the-fly.

    noise_factor is a hyperparameter sampled by Optuna
    each trial, so the dataset must be rebuilt per trial.

    Args:
        root_dir:     path to ImageFolder-structured directory
        noise_factor: std of Gaussian noise added to images
        transform:    torchvision transforms to apply first
    """

    def __init__(
        self,
        root_dir,
        noise_factor: float = 0.1,
        transform=None
    ):
        self.dataset      = datasets.ImageFolder(
            root=str(root_dir),
            transform=transform
        )
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.dataset)

    def add_noise(self, img):
        noise = torch.randn_like(img) * self.noise_factor
        return torch.clamp(img + noise, 0.0, 1.0)

    def __getitem__(self, idx):
        clean_img, _ = self.dataset[idx]
        noisy_img    = self.add_noise(clean_img)
        return noisy_img, clean_img