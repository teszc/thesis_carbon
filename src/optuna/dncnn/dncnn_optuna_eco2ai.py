import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
from pathlib import Path

import optuna
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import eco2ai

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

results_file = (
    csv_dir
    / "results_optuna_dncnn_eco2ai.csv"
)

eco2ai_dir = LOGS_DIR / "eco2ai_dncnn_optuna"
eco2ai_dir.mkdir(parents=True, exist_ok=True)

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
    # eco2ai tracker
    # -------------------------
    trial_id = trial.number

    eco2ai_log_file = (
        eco2ai_dir
        / f"eco2ai_dncnn_trial_{trial_id}.csv"
    )

    tracker = eco2ai.Tracker(

        project_name="DnCNN_Optuna",

        experiment_description=(
            f"Trial_{trial_id}"
        ),

        file_name=str(eco2ai_log_file),

        alpha_2_code="IT"
    )

    tracker.start()

    start_time = time.time()

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for noisy, clean in train_loader:

            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            optimizer.zero_grad()

            output = model(noisy)

            loss = criterion(
                output,
                clean
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = (
            running_loss
            / len(train_loader)
        )

        print(
            f"Trial {trial.number} "
            f"| Epoch [{epoch+1}/{epochs}] "
            f"| Loss: {avg_loss:.6f}"
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

            mse = criterion(
                output,
                clean
            )

            psnr = (
                10
                * torch.log10(1.0 / mse)
            )

            total_psnr += psnr.item()

            count += 1

    avg_psnr = total_psnr / count

    # -------------------------
    # Stop tracking
    # -------------------------
    tracker.stop()

    end_time = time.time()

    execution_time = (
        end_time - start_time
    )

    # =========================
    # Read eco2ai logs
    # =========================
    TDP_W = 20.0
    INTENSITY_G = 233.0

    energy_kwh = 0.0
    co2_kg = 0.0

    try:

        log_df = pd.read_csv(
            eco2ai_log_file
        )

        last = log_df.iloc[-1]

        energy_kwh = float(
            last[
                "power_consumption(kWh)"
            ]
        )

        co2_kg = float(
            last[
                "CO2_emissions(kg)"
            ]
        )

    except Exception as e:

        print(
            f"[eco2ai] "
            f"Log read failed: {e}"
        )

    # =========================
    # Fallback estimate
    # =========================
    if (
        energy_kwh == 0.0
        or co2_kg == 0.0
    ):

        print(
            f"[Trial {trial.number}] "
            "Using fallback estimation."
        )

        energy_kwh = (
            TDP_W
            * execution_time
        ) / 3_600_000

        co2_kg = (
            energy_kwh
            * INTENSITY_G
        ) / 1000

    # =========================
    # Save results
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
    print(f"Noise std:      {noise_std:.4f}")

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

        study_name=(
            "dncnn_accuracy_vs_emissions_eco2ai"
        ),

        storage=(
            "sqlite:///"
            "optuna_dncnn_eco2ai.db"
        ),

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