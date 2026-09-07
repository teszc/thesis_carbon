import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import eco2ai

from src.utils.config import (
    ROOT_DIR,
    DATA_DIR,
    RESULTS_DIR,
    DEVICE,
)

# =========================
# Reproducibility
# =========================
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

if DEVICE.type == "mps":
    torch.mps.manual_seed(SEED)

# =========================
# Parameters
# =========================
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 2
NOISE_STD = 0.1

# =========================
# Paths
# =========================
DATASET_PATH = ROOT_DIR / "datasets" / "tiny-imagenet-200" / "val"
RESULTS_FILE = RESULTS_DIR / "csv" / "results_dncnn_eco2ai.csv"
LOG_FILE = ROOT_DIR / "eco2ai_log_dncnn.csv"

# =========================
# Device
# =========================
device = DEVICE
print(f"Using device: {device}")

# =========================
# Dataset with noise
# =========================
class NoisyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, noise_std=0.1):
        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=transform
        )
        self.noise_std = noise_std

    def __len__(self):
        return len(self.dataset)

    def add_noise(self, img):
        noise = torch.randn_like(img) * self.noise_std
        return torch.clamp(img + noise, 0., 1.)

    def __getitem__(self, idx):
        clean_img, _ = self.dataset[idx]
        noisy_img = self.add_noise(clean_img)
        return noisy_img, clean_img

# =========================
# Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# Dataset
# =========================
train_dataset = NoisyImageDataset(
    DATASET_PATH,
    transform=transform,
    noise_std=NOISE_STD
)

test_dataset = NoisyImageDataset(
    DATASET_PATH,
    transform=transform,
    noise_std=NOISE_STD
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =========================
# DnCNN model
# =========================
class DnCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

model = DnCNN().to(device)

# =========================
# Training setup
# =========================
criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================
# eco2ai Tracker
# =========================
tracker = eco2ai.Tracker(
    project_name="DnCNN_TinyImageNet",
    experiment_description="DnCNN denoising on TinyImageNet",
    file_name=str(LOG_FILE),
    alpha_2_code="IT"
)

tracker.start()

start_time = time.time()

# =========================
# Training
# =========================
for epoch in range(EPOCHS):

    model.train()

    for noisy, clean in train_loader:

        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        output = model(noisy)

        loss = criterion(output, clean)

        loss.backward()

        optimizer.step()

# =========================
# Evaluation (PSNR)
# =========================
model.eval()

total_psnr = 0
count = 0

with torch.no_grad():

    for noisy, clean in test_loader:

        noisy = noisy.to(device)
        clean = clean.to(device)

        output = model(noisy)

        mse = criterion(output, clean)

        psnr = 10 * torch.log10(1 / mse)

        total_psnr += psnr.item()
        count += 1

accuracy = total_psnr / count

# =========================
# Stop tracker
# =========================
tracker.stop()

end_time = time.time()

execution_time = end_time - start_time

# =========================
# Read eco2ai log
# =========================
TDP_W = 20.0
INTENSITY_G = 233.0

log_df = pd.read_csv(LOG_FILE)

last = log_df.iloc[-1]

energy_kwh = float(last["power_consumption(kWh)"])
co2_kg = float(last["CO2_emissions(kg)"])

# =========================
# Fallback if eco2ai fails
# =========================
if energy_kwh == 0.0 or co2_kg == 0.0:

    print("[eco2ai] Zero reading detected.")
    print("[eco2ai] Using TDP fallback estimate.")

    energy_kwh = (
        TDP_W * execution_time
    ) / 3_600_000

    co2_kg = (
        energy_kwh * INTENSITY_G
    ) / 1000

# =========================
# Results
# =========================
print(f"\nPSNR:           {accuracy:.4f} dB")
print(f"Time:           {execution_time:.2f} sec")
print(f"Energy:         {energy_kwh:.8f} kWh")
print(f"CO2 emissions:  {co2_kg:.8f} kg")

# =========================
# Save results
# =========================
results = pd.DataFrame([{
    "model": "DnCNN",
    "dataset": "TinyImageNet",
    "device": str(device),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "noise_std": NOISE_STD,
    "accuracy": accuracy,
    "time_sec": execution_time,
    "energy_kwh": energy_kwh,
    "co2_kg": co2_kg
}])

results.to_csv(
    RESULTS_FILE,
    mode="a",
    header=not RESULTS_FILE.exists(),
    index=False
)

print(f"\nResults appended to:\n{RESULTS_FILE}")