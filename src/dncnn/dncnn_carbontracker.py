import warnings
warnings.filterwarnings("ignore")

import os
import re
import time
from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from carbontracker.tracker import CarbonTracker

from src.utils.config import (
    DEVICE,
    TINY_IMAGENET_DIR,
    RESULTS_DIR,
    LOGS_DIR
)

# =========================
# 1. Setup paths
# =========================
csv_dir = RESULTS_DIR / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)

results_file = csv_dir / "results_dncnn_ct.csv"

carbontracker_dir = LOGS_DIR / "carbontracker"
carbontracker_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. Print device
# =========================
print(f"Using device: {DEVICE}")

# =========================
# 3. Dataset
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
# 5. Load dataset
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
# 6. DnCNN model
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
# 8. CarbonTracker
# =========================
tracker = CarbonTracker(
    epochs=epochs,
    monitor_epochs=epochs,
    log_dir=str(carbontracker_dir),
    verbose=2
)

start_time = time.time()

# =========================
# 9. Training
# =========================
for epoch in range(epochs):

    tracker.epoch_start()

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

    tracker.epoch_end()

tracker.stop()

end_time = time.time()

execution_time = end_time - start_time

# =========================
# 10. Evaluation
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
# 11. Parse CarbonTracker logs
# =========================
FALLBACK_TDP_W = 20.0
FALLBACK_INTENSITY_G_KWH = 287.53

co2_kg = None
energy_kwh = None

log_files = sorted(
    [
        f for f in os.listdir(carbontracker_dir)
        if f.endswith(".log")
    ],
    key=lambda f: os.path.getmtime(
        carbontracker_dir / f
    )
)

if log_files:

    latest_log = carbontracker_dir / log_files[-1]

    with open(latest_log, "r") as f:

        content = f.read()

    intensity_match = re.search(
        r"Average carbon intensity.*?([\d.]+)\s*gCO2eq/kWh",
        content
    )

    if intensity_match:

        FALLBACK_INTENSITY_G_KWH = float(
            intensity_match.group(1)
        )

    energy_match = re.search(
        r"Actual consumption[\s\S]*?Energy:\s*([\d.eE+\-]+)\s*kWh",
        content
    )

    co2_match = re.search(
        r"Actual consumption[\s\S]*?CO2eq:\s*([\d.eE+\-]+)\s*g",
        content
    )

    if energy_match and co2_match:

        energy_kwh = float(
            energy_match.group(1)
        )

        co2_g = float(
            co2_match.group(1)
        )

        if energy_kwh > 0 and co2_g > 0:

            co2_kg = co2_g / 1000

            print(
                "[tracker] Using measured values."
            )

# =========================
# 12. Fallback estimate
# =========================
if co2_kg is None or co2_kg == 0.0:

    print(
        "[tracker] Falling back to "
        "TDP-based estimation."
    )

    energy_kwh = (
        FALLBACK_TDP_W
        * execution_time
    ) / 3_600_000

    co2_kg = (
        energy_kwh
        * FALLBACK_INTENSITY_G_KWH
    ) / 1000

# =========================
# 13. Print results
# =========================
print("\n===== RESULTS =====")

print(f"PSNR: {accuracy:.4f}")

print(
    f"Execution Time: "
    f"{execution_time:.4f} seconds"
)

print(
    f"Energy: "
    f"{energy_kwh:.8f} kWh"
)

print(
    f"CO2 Emissions: "
    f"{co2_kg:.8f} kg"
)

# =========================
# 14. Save results
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
    "energy_kwh": [energy_kwh],
    "co2_kg": [co2_kg]
}

df = pd.DataFrame(results_data)

df.to_csv(
    results_file,
    mode="a",
    header=not results_file.exists(),
    index=False
)

print(f"\nResults saved to: {results_file}")