import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from src.utils.config import (
    DEVICE,
    MNIST_DIR,
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

        current_time = time.time()

        dt = current_time - self.last_time

        self.energy_joules += (
            self.power_watts * dt
        )

        self.last_time = current_time

    def stop(self):

        self.step()

    def get_energy_kwh(self):

        return (
            self.energy_joules / 3_600_000
        )

    def get_co2(self, intensity=233.0):

        return (
            self.get_energy_kwh() * intensity
        ) / 1000


# =========================
# 2. Setup paths
# =========================
csv_dir = RESULTS_DIR / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)

results_file = csv_dir / "results_mnist_cumulator.csv"

# =========================
# 3. Print device info
# =========================
print(f"Using device: {DEVICE}")

# =========================
# 4. Load MNIST
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
# 5. Define model
# =========================
class SimpleNN(nn.Module):

    def __init__(self):

        super().__init__()

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
# 6. Training setup
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 3

# =========================
# 7. Start cumulator
# =========================
cumulator = EnergyCumulator(
    power_watts=20.0
)

cumulator.start()

start_time = time.time()

# =========================
# 8. Train model
# =========================
for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for data, target in train_loader:

        cumulator.step()

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
        f"Epoch [{epoch+1}/{epochs}] "
        f"- Loss: {avg_loss:.4f}"
    )

# =========================
# 9. Evaluate model
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
# 10. Stop cumulator
# =========================
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

# =========================
# 11. Print results
# =========================
print("\n===== RESULTS =====")

print(f"Accuracy: {accuracy:.4f}")

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