# ==========================================
# dncnn_evaluation.py
# Final DnCNN Thesis Evaluation
# ==========================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from codecarbon import EmissionsTracker

# ==========================================
# DEVICE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LOAD CSV
# ==========================================
df = pd.read_csv("dncnn_optuna_unified.csv")

# ==========================================
# ECO METRIC
# ==========================================
epsilon = 1e-12

df["PSNR_per_CO2"] = df["psnr"] / (
    (df["co2_kg"] * 1000) + epsilon
)

# ==========================================
# SELECT CONFIGS
# ==========================================
baseline = df.iloc[0]
best_psnr = df.loc[df["psnr"].idxmax()]
best_green = df.loc[df["co2_kg"].idxmin()]
best_eco = df.loc[df["PSNR_per_CO2"].idxmax()]

configs = [
    ("baseline", baseline),
    ("best_psnr", best_psnr),
    ("best_green", best_green),
    ("best_eco", best_eco)
]

# ==========================================
# TRANSFORMS
# ==========================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

data_path = "./tiny-imagenet-200/val"

# ==========================================
# DATASET
# ==========================================
class NoisyImageDataset(Dataset):

    def __init__(self, root_dir, noise_factor, transform=None):

        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=transform
        )

        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.dataset)

    def add_noise(self, img):

        noise = torch.randn_like(img) * self.noise_factor

        return torch.clamp(img + noise, 0., 1.)

    def __getitem__(self, idx):

        clean_img, _ = self.dataset[idx]

        noisy_img = self.add_noise(clean_img)

        return noisy_img, clean_img

# ==========================================
# MODEL
# ==========================================
class DnCNN(nn.Module):

    def __init__(self, channels=64):

        super(DnCNN, self).__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(channels, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# RESULTS
# ==========================================
results = []

# ==========================================
# LOOP
# ==========================================
for config_name, row in configs:

    print(f"\nRunning: {config_name}")

    batch_size = int(row["batch_size"])
    learning_rate = float(row["learning_rate"])
    epochs = int(row["epochs"])
    channels = int(row["channels"])
    noise_factor = float(row["noise_factor"])

    train_dataset = NoisyImageDataset(
        data_path,
        noise_factor=noise_factor,
        transform=transform
    )

    test_dataset = NoisyImageDataset(
        data_path,
        noise_factor=noise_factor,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    model = DnCNN(
        channels=channels
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ======================================
    # TRACKER
    # ======================================
    tracker = EmissionsTracker(
        project_name=f"dncnn_eval_{config_name}"
    )

    tracker.start()

    # ======================================
    # TRAINING
    # ======================================
    model.train()

    for epoch in range(epochs):

        for noisy, clean in train_loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            output = model(noisy)

            loss = criterion(output, clean)

            loss.backward()

            optimizer.step()

    # ======================================
    # EVALUATION
    # ======================================
    model.eval()

    total_psnr = 0
    total_ssim = 0
    total_mse = 0
    count = 0

    with torch.no_grad():

        for noisy, clean in test_loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)

            mse = criterion(output, clean).item()

            output_np = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            clean_np = clean.squeeze().cpu().numpy().transpose(1, 2, 0)

            psnr = peak_signal_noise_ratio(
                clean_np,
                output_np,
                data_range=1.0
            )

            ssim = structural_similarity(
                clean_np,
                output_np,
                channel_axis=2,
                data_range=1.0
            )

            total_psnr += psnr
            total_ssim += ssim
            total_mse += mse

            count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_mse = total_mse / count

    emissions = tracker.stop()

    # ======================================
    # ECO METRICS
    # ======================================
    co2_g = emissions * 1000

    psnr_per_co2 = avg_psnr / (co2_g + epsilon)
    ssim_per_co2 = avg_ssim / (co2_g + epsilon)

    # ======================================
    # SAVE
    # ======================================
    results.append({
        "config": config_name,
        "tracker": row["tracker"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "channels": channels,
        "noise_factor": noise_factor,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "mse": avg_mse,
        "co2_kg": emissions,
        "psnr_per_co2": psnr_per_co2,
        "ssim_per_co2": ssim_per_co2
    })

# ==========================================
# SAVE CSV
# ==========================================
results_df = pd.DataFrame(results)

results_df.to_csv(
    "dncnn_final_evaluation.csv",
    index=False
)

print("\nSaved: dncnn_final_evaluation.csv")