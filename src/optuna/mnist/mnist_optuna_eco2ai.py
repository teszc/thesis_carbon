import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import optuna
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

import eco2ai

from src.utils.config import (
    DEVICE,
    DATA_DIR,
    RESULTS_DIR,
    LOGS_DIR
)

# =========================
# 1. Setup paths
# =========================
csv_dir = RESULTS_DIR / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)

eco2ai_dir = LOGS_DIR / "eco2ai_optuna"
eco2ai_dir.mkdir(parents=True, exist_ok=True)

results_file = (
    csv_dir /
    "results_optuna_mnist_eco2ai.csv"
)

# =========================
# 2. Device
# =========================
print(f"Using device: {DEVICE}")

# =========================
# 3. Dataset
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

mnist_dir = DATA_DIR / "mnist"

train_dataset = datasets.MNIST(
    root=mnist_dir,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=mnist_dir,
    train=False,
    download=True,
    transform=transform
)

# =========================
# 4. Model
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
# 5. Objective Function
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
    # eco2ai setup
    # -------------------------
    trial_id = trial.number

    eco2ai_log_file = (
        eco2ai_dir /
        f"eco2ai_trial_{trial_id}.csv"
    )

    tracker = eco2ai.Tracker(

        project_name="MNIST_Optuna",

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

        for data, target in train_loader:

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
            running_loss /
            len(train_loader)
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
        correct /
        len(test_dataset)
    )

    # -------------------------
    # Stop tracking
    # -------------------------
    tracker.stop()

    end_time = time.time()

    execution_time = (
        end_time - start_time
    )

    # -------------------------
    # Read eco2ai log
    # -------------------------
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
            last["power_consumption(kWh)"]
        )

        co2_kg = float(
            last["CO2_emissions(kg)"]
        )

    except Exception as e:

        print(
            "[eco2ai] "
            "Could not parse log."
        )

        print(e)

    # -------------------------
    # Fallback estimation
    # -------------------------
    if (
        energy_kwh == 0.0
        or co2_kg == 0.0
    ):

        print(
            "[eco2ai] Falling back "
            "to TDP estimate."
        )

        energy_kwh = (
            TDP_W
            * execution_time
        ) / 3_600_000

        co2_kg = (
            energy_kwh
            * INTENSITY_G
        ) / 1000

    # -------------------------
    # Save results
    # -------------------------
    row = {
        "trial": [trial.number],
        "model": ["SimpleNN"],
        "dataset": ["MNIST"],
        "device": [str(DEVICE)],
        "batch_size": [batch_size],
        "learning_rate": [learning_rate],
        "hidden_size": [hidden_size],
        "epochs": [epochs],
        "accuracy": [accuracy],
        "time_sec": [execution_time],
        "energy_kwh": [energy_kwh],
        "co2_kg": [co2_kg]
    }

    df = pd.DataFrame(row)

    df.to_csv(
        results_file,
        mode="a",
        header=not results_file.exists(),
        index=False
    )

    # -------------------------
    # Trial summary
    # -------------------------
    print("\n===================================")

    print(
        f"Trial {trial.number}"
    )

    print("===================================")

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"Learning rate: "
        f"{learning_rate}"
    )

    print(
        f"Hidden size: "
        f"{hidden_size}"
    )

    print(
        f"Epochs: "
        f"{epochs}"
    )

    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Time: "
        f"{execution_time:.2f} sec"
    )

    print(
        f"Energy: "
        f"{energy_kwh:.8f} kWh"
    )

    print(
        f"CO2 emissions: "
        f"{co2_kg:.10f} kg"
    )

    # -------------------------
    # Multi-objective return
    # -------------------------
    return accuracy, co2_kg

# =========================
# 6. Main
# =========================
if __name__ == "__main__":

    study = optuna.create_study(

        directions=[
            "maximize",
            "minimize"
        ],

        study_name=(
            "mnist_accuracy_vs_"
            "eco2ai_emissions"
        ),

        storage=(
            "sqlite:///"
            "mnist_optuna_eco2ai.db"
        ),

        load_if_exists=True
    )

    print(
        f"\nExisting trials: "
        f"{len(study.trials)}"
    )

    # -------------------------
    # Run until 50 trials
    # -------------------------
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
    print(
        "\n\n==================================="
    )

    print(
        "Pareto-optimal trials"
    )

    print(
        "==================================="
    )

    for trial in study.best_trials:

        print(
            "\n-----------------------------------"
        )

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

    print(
        f"\nResults saved to:\n"
        f"{results_file}"
    )