import warnings
warnings.filterwarnings("ignore")

import os
import re
import time
from pathlib import Path

import optuna
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

results_file = csv_dir / "results_optuna_dncnn_carbontracker.csv"

carbontracker_dir = LOGS_DIR / "carbontracker_optuna_dncnn"
carbontracker_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. Print device
# =========================
print(f"Using device: {DEVICE}")

# =========================
# 3. Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# 4. Dataset with noise
# =========================
class NoisyImageDataset(Dataset):

    def __init__(
        self,
        root_dir,
        noise_std=0.1,
        transform=None
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

        return torch.clamp(noisy, 0.0, 1.0)

    def __getitem__(self, idx):

        clean_img, _ = self.dataset[idx]

        noisy_img = self.add_noise(clean_img)

        return noisy_img, clean_img

# =========================
# 5. DnCNN model
# =========================
class DnCNN(nn.Module):

    def __init__(self, channels=64):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                3,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                channels,
                3,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return self.net(x)

# =========================
# 6. Objective function
# =========================
def objective(trial):

    # -------------------------
    # Hyperparameters
    # -------------------------
    batch_size = trial.suggest_categorical(
        "batch_size",
        [8, 16, 32]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-2,
        log=True
    )

    epochs = trial.suggest_int(
        "epochs",
        1,
        4
    )

    channels = trial.suggest_categorical(
        "channels",
        [32, 64, 128]
    )

    noise_std = trial.suggest_float(
        "noise_std",
        0.05,
        0.30
    )

    # -------------------------
    # Dataset
    # -------------------------
    data_path = TINY_IMAGENET_DIR / "val"

    train_dataset = NoisyImageDataset(
        root_dir=data_path,
        transform=transform,
        noise_std=noise_std
    )

    test_dataset = NoisyImageDataset(
        root_dir=data_path,
        transform=transform,
        noise_std=noise_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # -------------------------
    # Model
    # -------------------------
    model = DnCNN(
        channels=channels
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # -------------------------
    # CarbonTracker
    # -------------------------
    tracker = CarbonTracker(
        epochs=epochs,
        monitor_epochs=epochs,
        log_dir=str(carbontracker_dir),
        verbose=0
    )

    start_time = time.time()

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(epochs):

        tracker.epoch_start()

        model.train()

        for noisy, clean in train_loader:

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            optimizer.zero_grad()

            output = model(noisy)

            loss = criterion(output, clean)

            loss.backward()

            optimizer.step()

        tracker.epoch_end()

    tracker.stop()

    end_time = time.time()

    execution_time = (
        end_time - start_time
    )

    # -------------------------
    # Evaluation (PSNR)
    # -------------------------
    model.eval()

    total_psnr = 0.0
    count = 0

    with torch.no_grad():

        for noisy, clean in test_loader:

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            output = model(noisy)

            mse = criterion(output, clean)

            psnr = (
                10
                * torch.log10(1.0 / mse)
            )

            total_psnr += psnr.item()

            count += 1

    avg_psnr = total_psnr / count

    # =========================
    # Parse CarbonTracker logs
    # =========================
    FALLBACK_TDP_W = 20.0
    FALLBACK_INTENSITY_G_KWH = 287.53

    co2_kg = None
    energy_kwh = None

    log_files = sorted(
        [
            f for f in os.listdir(
                carbontracker_dir
            )
            if f.endswith(".log")
        ],
        key=lambda f: os.path.getmtime(
            carbontracker_dir / f
        )
    )

    if log_files:

        latest_log = (
            carbontracker_dir
            / log_files[-1]
        )

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

            if (
                energy_kwh > 0
                and co2_g > 0
            ):

                co2_kg = co2_g / 1000

                print(
                    f"[Trial {trial.number}] "
                    "Using measured values."
                )

    # =========================
    # Fallback estimate
    # =========================
    if co2_kg is None or co2_kg == 0.0:

        print(
            f"[Trial {trial.number}] "
            "Using fallback estimation."
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
    # Save trial results
    # =========================
    row = {

        "trial": trial.number,

        "device": str(DEVICE),

        "batch_size": batch_size,

        "learning_rate": learning_rate,

        "epochs": epochs,

        "channels": channels,

        "noise_std": noise_std,

        "psnr": avg_psnr,

        "time_sec": execution_time,

        "energy_kwh": energy_kwh,

        "co2_kg": co2_kg
    }

    df = pd.DataFrame([row])

    df.to_csv(
        results_file,
        mode="a",
        header=not results_file.exists(),
        index=False
    )

    # =========================
    # Trial summary
    # =========================
    print("\n===================================")
    print(f"Trial {trial.number}")
    print("===================================")

    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {learning_rate}")
    print(f"Epochs:         {epochs}")
    print(f"Channels:       {channels}")
    print(f"Noise std:      {noise_std}")

    print(f"\nPSNR:           {avg_psnr:.4f}")
    print(f"Time:           {execution_time:.2f} sec")
    print(f"Energy:         {energy_kwh:.8f} kWh")
    print(f"CO2 emissions:  {co2_kg:.8f} kg")

    return avg_psnr, co2_kg

# =========================
# 7. Main
# =========================
if __name__ == "__main__":

    study = optuna.create_study(

        directions=[
            "maximize",
            "minimize"
        ],

        study_name="dncnn_accuracy_vs_emissions_carbontracker",

        storage="sqlite:///optuna_dncnn_carbontracker.db",

        load_if_exists=True
    )

    print(
        f"\nExisting trials: "
        f"{len(study.trials)}"
    )

    TARGET_TRIALS = 20

    remaining_trials = (
        TARGET_TRIALS
        - len(study.trials)
    )

    if remaining_trials > 0:

        print(
            f"Running "
            f"{remaining_trials} "
            f"additional trials...\n"
        )

        study.optimize(
            objective,
            n_trials=remaining_trials
        )

    else:

        print(
            "Target already reached."
        )

    # =========================
    # Pareto Front
    # =========================
    print("\n===================================")
    print("Pareto-optimal trials")
    print("===================================")

    for trial in study.best_trials:

        print("\n-----------------------------------")

        print(
            f"Trial number: "
            f"{trial.number}"
        )

        print(
            f"PSNR: "
            f"{trial.values[0]}"
        )

        print(
            f"CO2 emissions: "
            f"{trial.values[1]} kg"
        )

        print(
            f"Parameters: "
            f"{trial.params}"
        )

    print("\nStudy completed.")
    print(
        f"Total trials: "
        f"{len(study.trials)}"
    )