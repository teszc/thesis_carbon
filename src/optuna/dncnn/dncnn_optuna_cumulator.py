import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import time
from pathlib import Path

import optuna
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
# Energy Cumulator
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

        self.energy_joules += self.power_watts * dt

        self.last_time = current

    def stop(self):

        self.step()

    def get_energy_kwh(self):

        return self.energy_joules / 3_600_000

    def get_co2(self, intensity=233.0):

        return (
            self.get_energy_kwh() * intensity
        ) / 1000


# =========================
# Setup paths
# =========================
csv_dir = RESULTS_DIR / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)

results_file = (
    csv_dir
    / "results_optuna_dncnn_cumulator.csv"
)

# =========================
# Print device
# =========================
print(f"Using device: {DEVICE}")

# =========================
# Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# Dataset path
# =========================
data_path = TINY_IMAGENET_DIR / "val"

# =========================
# Dataset with tunable noise
# =========================
class NoisyImageDataset(Dataset):

    def __init__(
        self,
        root_dir,
        noise_factor,
        transform=None
    ):

        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=transform
        )

        self.noise_factor = noise_factor

    def __len__(self):

        return len(self.dataset)

    def add_noise(self, img):

        noise = (
            torch.randn_like(img)
            * self.noise_factor
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
# DnCNN Model
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
# Objective Function
# =========================
def objective(trial):

    # -----------------------------------
    # Hyperparameters
    # -----------------------------------
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

    noise_factor = trial.suggest_float(
        "noise_factor",
        0.05,
        0.30
    )

    # -----------------------------------
    # Datasets
    # -----------------------------------
    train_dataset = NoisyImageDataset(
        root_dir=data_path,
        noise_factor=noise_factor,
        transform=transform
    )

    test_dataset = NoisyImageDataset(
        root_dir=data_path,
        noise_factor=noise_factor,
        transform=transform
    )

    # -----------------------------------
    # Data loaders
    # -----------------------------------
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

    # -----------------------------------
    # Model
    # -----------------------------------
    model = DnCNN(
        channels=channels
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # -----------------------------------
    # Energy tracking
    # -----------------------------------
    cumulator = EnergyCumulator(
        power_watts=20.0
    )

    cumulator.start()

    start_time = time.time()

    # -----------------------------------
    # Training
    # -----------------------------------
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
            running_loss
            / len(train_loader)
        )

        print(
            f"Trial {trial.number} | "
            f"Epoch [{epoch+1}/{epochs}] "
            f"- Loss: {avg_loss:.6f}"
        )

    # -----------------------------------
    # Evaluation
    # -----------------------------------
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

            psnr = 10 * torch.log10(
                1.0 / mse
            )

            total_psnr += psnr.item()

            count += 1

    avg_psnr = total_psnr / count

    # -----------------------------------
    # Stop tracking
    # -----------------------------------
    cumulator.stop()

    end_time = time.time()

    execution_time = (
        end_time - start_time
    )

    energy_kwh = (
        cumulator.get_energy_kwh()
    )

    co2_kg = cumulator.get_co2()

    # -----------------------------------
    # Save results
    # -----------------------------------
    row = {
        "trial": trial.number,
        "device": str(DEVICE),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "channels": channels,
        "noise_factor": noise_factor,
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

    # -----------------------------------
    # Print trial summary
    # -----------------------------------
    print("\n===================================")
    print(f"Trial {trial.number}")
    print("===================================")

    print(f"Batch size:    {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs:        {epochs}")
    print(f"Channels:      {channels}")
    print(f"Noise factor:  {noise_factor:.4f}")

    print(f"\nPSNR:          {avg_psnr:.4f}")

    print(
        f"Time:          "
        f"{execution_time:.2f} sec"
    )

    print(
        f"Energy:        "
        f"{energy_kwh:.8f} kWh"
    )

    print(
        f"CO2 emissions: "
        f"{co2_kg:.8f} kg"
    )

    # -----------------------------------
    # Multi-objective optimization
    # -----------------------------------
    return avg_psnr, co2_kg

# =========================
# Main
# =========================
if __name__ == "__main__":

    study = optuna.create_study(

        directions=[
            "maximize",
            "minimize"
        ],

        study_name=(
            "dncnn_accuracy_vs_"
            "emissions_cumulator"
        ),

        storage=(
            "sqlite:///"
            "optuna_dncnn_cumulator.db"
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
            f"{trial.values[0]:.4f}"
        )

        print(
            f"CO2 emissions: "
            f"{trial.values[1]:.10f} kg"
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