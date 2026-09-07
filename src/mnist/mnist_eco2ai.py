import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import time

import eco2ai
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

eco2ai_logs_dir = LOGS_DIR / "eco2ai"
eco2ai_logs_dir.mkdir(parents=True, exist_ok=True)

results_file = csv_dir / "results_mnist_eco2ai.csv"

eco2ai_log_file = eco2ai_logs_dir / "eco2ai_log_mnist.csv"

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

epochs = 3

# =========================
# 6. Start eco2ai tracking
# =========================
tracker = eco2ai.Tracker(
    project_name="MNIST_SimpleNN",
    experiment_description="3-epoch training on MNIST",
    file_name=str(eco2ai_log_file),
    alpha_2_code="IT"
)

tracker.start()

start_time = time.time()

# =========================
# 7. Train model
# =========================
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

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

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
# 9. Stop tracking
# =========================
tracker.stop()

end_time = time.time()

execution_time = end_time - start_time

# =========================
# 10. Read eco2ai logs
# =========================
TDP_W = 20.0
INTENSITY_G = 233.0

log_df = pd.read_csv(eco2ai_log_file)

last = log_df.iloc[-1]

energy_kwh = float(
    last["power_consumption(kWh)"]
)

co2_kg = float(
    last["CO2_emissions(kg)"]
)

# =========================
# 11. Fallback estimation
# =========================
if energy_kwh == 0.0 or co2_kg == 0.0:

    print(
        "[eco2ai] Zero reading detected."
    )

    print(
        "[eco2ai] Using fallback estimate."
    )

    energy_kwh = (
        TDP_W * execution_time
    ) / 3_600_000

    co2_kg = (
        energy_kwh * INTENSITY_G
    ) / 1000

# =========================
# 12. Print results
# =========================
print("\n===== RESULTS =====")

print(f"Accuracy: {accuracy:.4f}")

print(
    f"Execution Time: "
    f"{execution_time:.4f} seconds"
)

print(f"Energy: {energy_kwh:.8f} kWh")

print(
    f"CO2 Emissions: "
    f"{co2_kg:.8f} kg"
)

# =========================
# 13. Save results
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