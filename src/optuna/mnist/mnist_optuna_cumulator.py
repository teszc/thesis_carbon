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

from src.utils.config import (
    DEVICE,
    RESULTS_DIR
)

# =========================
# 1. Energy Cumulator
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
            self.energy_joules
            / 3_600_000
        )

    def get_co2(
        self,
        intensity=233.0
    ):

        return (
            self.get_energy_kwh()
            * intensity
        ) / 1000

# =========================
# 2. Setup paths
# =========================
csv_dir = RESULTS_DIR / "csv"
csv_dir.mkdir(
    parents=True,
    exist_ok=True
)

results_file = (
    csv_dir
    / "results_optuna_mnist_cumulator.csv"
)

db_file = (
    RESULTS_DIR
    / "optuna_mnist_cumulator.db"
)

# =========================
# 3. Device
# =========================
print(f"Using device: {DEVICE}")

# =========================
# 4. Dataset
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform
)

# =========================
# 5. Model
# =========================
class SimpleNN(nn.Module):

    def __init__(
        self,
        hidden_size=128
    ):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                28 * 28,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                10
            )
        )

    def forward(self, x):

        return self.fc(x)

# =========================
# 6. Objective Function
# =========================
def objective(trial):

    # -------------------------
    # Hyperparameters
    # -------------------------
    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128, 256]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        1e-2,
        log=True
    )

    hidden_size = trial.suggest_categorical(
        "hidden_size",
        [64, 128, 256]
    )

    epochs = trial.suggest_int(
        "epochs",
        2,
        6
    )

    # -------------------------
    # Data loaders
    # -------------------------
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )

    # -------------------------
    # Model setup
    # -------------------------
    model = SimpleNN(
        hidden_size=hidden_size
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # -------------------------
    # Energy tracking
    # -------------------------
    cumulator = EnergyCumulator(
        power_watts=20.0
    )

    cumulator.start()

    start_time = time.time()

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for data, target in train_loader:

            cumulator.step()

            data = data.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(
                output,
                target
            )

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
            f"| Loss: {avg_loss:.6f}"
        )

    # -------------------------
    # Evaluation
    # -------------------------
    model.eval()

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)

            pred = output.argmax(dim=1)

            correct += pred.eq(
                target
            ).sum().item()

    accuracy = (
        correct
        / len(test_dataset)
    )

    # -------------------------
    # Stop tracking
    # -------------------------
    cumulator.stop()

    end_time = time.time()

    execution_time = (
        end_time - start_time
    )

    energy_kwh = (
        cumulator.get_energy_kwh()
    )

    co2_kg = (
        cumulator.get_co2()
    )

    # -------------------------
    # Save trial results
    # -------------------------
    row = {
        "trial": trial.number,
        "device": str(DEVICE),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "accuracy": accuracy,
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

    # -------------------------
    # Print trial summary
    # -------------------------
    print("\n===================================")
    print(f"Trial {trial.number}")
    print("===================================")

    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {learning_rate}")
    print(f"Hidden size:    {hidden_size}")
    print(f"Epochs:         {epochs}")

    print(f"\nAccuracy:       {accuracy:.4f}")

    print(
        f"Time:           "
        f"{execution_time:.2f} sec"
    )

    print(
        f"Energy:         "
        f"{energy_kwh:.8f} kWh"
    )

    print(
        f"CO2 emissions:  "
        f"{co2_kg:.8f} kg"
    )

    return accuracy, co2_kg

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
            "mnist_accuracy_vs_emissions_cumulator"
        ),

        storage=f"sqlite:///{db_file}",

        load_if_exists=True
    )

    print(
        f"\nExisting trials: "
        f"{len(study.trials)}"
    )

    TARGET_TRIALS = 50

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
            f"Accuracy: "
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