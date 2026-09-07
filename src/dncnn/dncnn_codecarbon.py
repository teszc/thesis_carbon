import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from codecarbon import EmissionsTracker

from src.utils.config import (
    DEVICE,
    TINY_IMAGENET_DIR,
    RESULTS_DIR
)

# =========================
# 1. Setup paths
# =========================
csv_dir = RESULTS_DIR / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)

results_file = csv_dir / "results_dncnn_cc.csv"

# =========================
# 2. Print device
# =========================
print(f"Using device: {DEVICE}")

# =========================
# 3. Custom Dataset
# =========================
class NoisyImageDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        noise_std=0.1
    ):

        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=transform
        )

        self.noise_std = noise_std

    def __len__(self):

        return len(self.dataset)

    def add_noise(self, img):

        noise = torch.randn_like(img) * self.noise_std

        noisy = img + noise

        return torch.clamp(noisy, 0.0, 1.0)

    def __getitem__(self, idx):

        clean_img, _ = self.dataset[idx]

        noisy_img = self.add_noise(clean_img)

        return noisy_img, clean_img

# =========================
# 4. Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# 5. Dataset
# =========================
data_path = TINY_IMAGENET_DIR / "val"
print(f"Dataset path: {data_path}")

train_dataset = NoisyImageDataset(
    root_dir=data_path,
    transform=transform,
    noise_std=0.1
)

test_dataset = NoisyImageDataset(
    root_dir=data_path,
    transform=transform,
    noise_std=0.1
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =========================
# 6. DnCNN Model
# =========================
class DnCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                3,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return self.net(x)

model = DnCNN().to(DEVICE)

# =========================
# 7. Training setup
# =========================
criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 2

# =========================
# 8. Start tracking
# =========================
tracker = EmissionsTracker()

tracker.start()

start_time = time.time()

# =========================
# 9. Training
# =========================
for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for noisy, clean in train_loader:

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad()

        output = model(noisy)

        loss = criterion(output, clean)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"- Loss: {avg_loss:.6f}"
    )

# =========================
# 10. Evaluation (PSNR)
# =========================
model.eval()

total_psnr = 0.0
count = 0

with torch.no_grad():

    for noisy, clean in test_loader:

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        output = model(noisy)

        mse = criterion(output, clean)

        psnr = 10 * torch.log10(1.0 / mse)

        total_psnr += psnr.item()

        count += 1

accuracy = total_psnr / count

# =========================
# 11. Stop tracking
# =========================
emissions = tracker.stop()

end_time = time.time()

execution_time = end_time - start_time

# =========================
# 12. Print results
# =========================
print("\n===== RESULTS =====")

print(f"PSNR: {accuracy:.4f}")

print(
    f"Execution Time: "
    f"{execution_time:.4f} seconds"
)

print(
    f"CO2 Emissions: "
    f"{emissions:.8f} kg"
)

# =========================
# 13. Save results
# =========================
results_data = {
    "model": ["DnCNN"],
    "dataset": ["TinyImageNet"],
    "device": [str(DEVICE)],
    "epochs": [epochs],
    "batch_size": [16],
    "learning_rate": [0.001],
    "noise_std": [0.1],
    "accuracy": [accuracy],
    "time_sec": [execution_time],
    "co2_kg": [emissions]
}

df = pd.DataFrame(results_data)

df.to_csv(
    results_file,
    mode="a",
    header=not results_file.exists(),
    index=False
)

print(f"\nResults saved to: {results_file}")