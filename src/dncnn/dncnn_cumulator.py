import warnings
warnings.filterwarnings("ignore")

import time

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

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

results_file = csv_dir / "results_dncnn_cumulator.csv"

# =========================
# 2. Print device
# =========================
print(f"Using device: {DEVICE}")

# =========================
# 3. Energy Cumulator
# =========================
class EnergyCumulator:

    def __init__(self, power_watts=20.0):

        self.power_watts = power_watts
        self.energy_joules = 0.0
        self.last_time = None

    def start(self):

        self.last_time = time.time()

    def step(self):

        current = time.time()

        dt = current - self.last_time

        self.energy_joules += (
            self.power_watts * dt
        )

        self.last_time = current

    def stop(self):

        self.step()

    def get_energy_kwh(self):

        return (
            self.energy_joules / 3_600_000
        )

    def get_co2(self, intensity=233.0):

        return (
            self.get_energy_kwh()
            * intensity
        ) / 1000

# =========================
# 4. Dataset
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

        noise = (
            torch.randn_like(img)
            * self.noise_std
        )

        noisy = img + noise

        return torch.clamp(
            noisy,
            0.0,
            1.0
        )

    def __getitem__(self, idx):

        clean_img, _ = self.dataset[idx]

        noisy_img = self.add_noise(clean_img)

        return noisy_img, clean_img

# =========================
# 5. Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# 6. Dataset path
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
# 7. DnCNN Model
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
# 8. Training setup
# =========================
criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 2

# =========================
# 9. Start cumulator
# =========================
cumulator = EnergyCumulator(
    power_watts=20.0
)

cumulator.start()

start_time = time.time()

# =========================
# 10. Training
# =========================
for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for noisy, clean in train_loader:

        cumulator.step()

        noisy = noisy.to(DEVICE)
        clean = clean.to(DEVICE)

        optimizer.zero_grad()

        output = model(noisy)

        loss = criterion(output, clean)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss / len(train_loader)
    )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"- Loss: {avg_loss:.6f}"
    )

# =========================
# 11. Evaluation
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

        psnr = 10 * torch.log10(
            1.0 / mse
        )

        total_psnr += psnr.item()

        count += 1

accuracy = total_psnr / count

# =========================
# 12. Stop cumulator
# =========================
cumulator.stop()

end_time = time.time()

execution_time = (
    end_time - start_time
)

energy_kwh = (
    cumulator.get_energy_kwh()
)

co2_kg = cumulator.get_co2()

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