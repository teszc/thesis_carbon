import os
import re
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from carbontracker.tracker import CarbonTracker
from torchvision import datasets, transforms

from src.utils.config import (
    DEVICE,
    MNIST_DIR,
    RESULTS_DIR,
    LOGS_DIR
)

# =========================
# 1. Setup paths
# =========================
csv_dir = RESULTS_DIR / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)

tracker_logs_dir = LOGS_DIR / "carbontracker"
tracker_logs_dir.mkdir(parents=True, exist_ok=True)

results_file = csv_dir / "results_mnist_ct.csv"

# =========================
# 2. Print device info
# =========================
print(f"Using device: {DEVICE}")

# =========================
# 3. Load MNIST
# =========================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root=MNIST_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=MNIST_DIR,
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)

# =========================
# 4. Define model
# =========================
class SimpleNN(nn.Module):

    def __init__(self):
        super(SimpleNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

model = SimpleNN().to(DEVICE)

# =========================
# 5. Training setup
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# 6. CarbonTracker setup
# =========================
epochs = 3

tracker = CarbonTracker(
    epochs=epochs,
    monitor_epochs=epochs,
    log_dir=str(tracker_logs_dir),
    verbose=2
)

start_time = time.time()

# =========================
# 7. Train model
# =========================
for epoch in range(epochs):

    tracker.epoch_start()

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

    tracker.epoch_end()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

tracker.stop()

end_time = time.time()

execution_time = end_time - start_time

# =========================
# 8. Evaluate model
# =========================
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

# =========================
# 9. Parse CarbonTracker logs
# =========================
FALLBACK_TDP_W = 10.0
FALLBACK_INTENSITY_G_KWH = 287.53

co2_kg = None
energy_kwh = None

log_files = sorted(
    [
        f for f in os.listdir(tracker_logs_dir)
        if f.endswith(".log")
    ],
    key=lambda f: os.path.getmtime(
        os.path.join(tracker_logs_dir, f)
    )
)

if log_files:

    latest_log = os.path.join(
        tracker_logs_dir,
        log_files[-1]
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

        energy_kwh = float(energy_match.group(1))
        co2_g = float(co2_match.group(1))

        if energy_kwh > 0 and co2_g > 0:

            co2_kg = co2_g / 1000

            print("[tracker] Used measured values.")

        else:

            print("[tracker] Measured values were 0.")
            print("[tracker] Using fallback estimate.")

# =========================
# 10. Fallback estimation
# =========================
if co2_kg is None or co2_kg == 0.0:

    energy_kwh = (
        FALLBACK_TDP_W * execution_time
    ) / 3_600_000

    co2_kg = (
        energy_kwh * FALLBACK_INTENSITY_G_KWH
    ) / 1000

    print(f"[tracker] TDP estimate: {FALLBACK_TDP_W}W")
    print(f"[tracker] Carbon intensity: {FALLBACK_INTENSITY_G_KWH}")

# =========================
# 11. Print results
# =========================
print("\n===== RESULTS =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Execution Time: {execution_time:.4f} seconds")
print(f"Energy: {energy_kwh} kWh")
print(f"CO2 Emissions: {co2_kg} kg")

# =========================
# 12. Save results
# =========================
results_data = {
    "model": ["SimpleNN"],
    "dataset": ["MNIST"],
    "device": [str(DEVICE)],
    "epochs": [epochs],
    "batch_size": [64],
    "learning_rate": [0.001],
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