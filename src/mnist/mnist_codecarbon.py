import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from codecarbon import EmissionsTracker
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

logs_dir = LOGS_DIR / "codecarbon"
logs_dir.mkdir(parents=True, exist_ok=True)

results_file = csv_dir / "results_mnist_cc.csv"

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
# 6. Start CodeCarbon
# =========================
tracker = EmissionsTracker(
    output_dir=str(logs_dir),
    output_file="mnist_codecarbon_emissions.csv",
    log_level="warning"
)

tracker.start()

start_time = time.time()

# =========================
# 7. Train model
# =========================
epochs = 3

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
# 9. Stop emissions tracking
# =========================
emissions = tracker.stop()

end_time = time.time()

execution_time = end_time - start_time

# =========================
# 10. Print results
# =========================
print("\n===== RESULTS =====")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Execution Time: {execution_time:.2f} seconds")
print(f"CO2 Emissions: {emissions:.6f} kg")

# =========================
# 11. Save results
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
    "co2_kg": [emissions]
}

df = pd.DataFrame(results_data)

df.to_csv(
    results_file,
    mode="a",
    header=not results_file.exists(),
    index=False
)

print(f"\nResults saved to: {results_file}")