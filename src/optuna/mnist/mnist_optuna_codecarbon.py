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

from codecarbon import EmissionsTracker

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

codecarbon_dir = LOGS_DIR / "codecarbon"
codecarbon_dir.mkdir(parents=True, exist_ok=True)

results_file = csv_dir / "results_optuna_mnist_codecarbon.csv"

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

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                28 * 28,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
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
    model = SimpleNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # -------------------------
    # CodeCarbon tracker
    # -------------------------
    tracker = EmissionsTracker(
        project_name="mnist_optuna",
        output_dir=str(codecarbon_dir),
        log_level="error"
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

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

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

            correct += pred.eq(target).sum().item()

    accuracy = correct / len(test_dataset)

    # -------------------------
    # Stop tracking
    # -------------------------
    emissions = tracker.stop()

    end_time = time.time()

    execution_time = end_time - start_time

    # -------------------------
    # Energy fallback
    # -------------------------
    if emissions is None:

        emissions = 0.0

    # approximate energy estimate
    # using Apple Silicon ~20W
    estimated_energy_kwh = (
        20.0 * execution_time
    ) / 3_600_000

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
        "epochs": [epochs],
        "accuracy": [accuracy],
        "time_sec": [execution_time],
        "energy_kwh": [estimated_energy_kwh],
        "co2_kg": [emissions]
    }

    df = pd.DataFrame(row)

    df.to_csv(
        results_file,
        mode="a",
        header=not results_file.exists(),
        index=False
    )

    print(
        f"\nTrial {trial.number} completed:"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"CO2: {emissions:.10f} kg"
    )

    print(
        f"Time: {execution_time:.2f} sec\n"
    )

    # -------------------------
    # Multi-objective return
    # -------------------------
    return accuracy, emissions

# =========================
# 6. Create study
# =========================
study = optuna.create_study(
    directions=[
        "maximize",
        "minimize"
    ],
    study_name="mnist_accuracy_vs_emissions",
    storage="sqlite:///mnist_optuna.db",
    load_if_exists=True
)

print(
    f"\nExisting trials: "
    f"{len(study.trials)}"
)

# =========================
# 7. Run until 50 trials
# =========================
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
# 8. Pareto results
# =========================
print("\n===== PARETO TRIALS =====")

for trial in study.best_trials:

    print("--------------------------------")

    print(f"Trial: {trial.number}")

    print(
        f"Accuracy: "
        f"{trial.values[0]:.4f}"
    )

    print(
        f"CO2: "
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