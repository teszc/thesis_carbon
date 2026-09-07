# ==========================================
# mnist_evaluation.py
# Final Evaluation Script for Thesis
# ==========================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize

from codecarbon import EmissionsTracker

# ==========================================
# DEVICE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LOAD CSV
# ==========================================
df = pd.read_csv("mnist_optuna_unified.csv")

# ==========================================
# COMPUTE AER
# ==========================================
epsilon = 1e-12

df["AER"] = df["accuracy"] / ((df["co2_kg"] * 1000) + epsilon)

# ==========================================
# SELECT CONFIGS
# ==========================================
baseline = df.iloc[0]
best_accuracy = df.loc[df["accuracy"].idxmax()]
best_aer = df.loc[df["AER"].idxmax()]
greenest = df.loc[df["co2_kg"].idxmin()]

configs = [
    ("baseline", baseline),
    ("best_accuracy", best_accuracy),
    ("best_aer", best_aer),
    ("greenest", greenest)
]

# ==========================================
# DATASET
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# ==========================================
# MODEL
# ==========================================
class SimpleNN(nn.Module):

    def __init__(self):

        super(SimpleNN, self).__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# RESULTS
# ==========================================
results = []

# ==========================================
# EVALUATION LOOP
# ==========================================
for config_name, row in configs:

    print(f"\nRunning: {config_name}")

    batch_size = int(row["batch_size"])
    learning_rate = float(row["learning_rate"])
    epochs = int(row["epochs"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )

    model = SimpleNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ======================================
    # TRACKER
    # ======================================
    tracker = EmissionsTracker(
        project_name=f"mnist_eval_{config_name}"
    )

    tracker.start()

    # ======================================
    # TRAINING
    # ======================================
    model.train()

    for epoch in range(epochs):

        for data, target in train_loader:

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

    # ======================================
    # EVALUATION
    # ======================================
    model.eval()

    correct = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            probs = torch.softmax(output, dim=1)

            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / len(test_dataset)

    precision = precision_score(
        all_targets,
        all_preds,
        average="macro"
    )

    recall = recall_score(
        all_targets,
        all_preds,
        average="macro"
    )

    f1 = f1_score(
        all_targets,
        all_preds,
        average="macro"
    )

    targets_bin = label_binarize(
        all_targets,
        classes=list(range(10))
    )

    auc = roc_auc_score(
        targets_bin,
        np.array(all_probs),
        multi_class="ovr",
        average="macro"
    )

    emissions = tracker.stop()

    # ======================================
    # ECO METRICS
    # ======================================
    co2_g = emissions * 1000

    accuracy_per_co2 = accuracy / (co2_g + epsilon)
    f1_per_co2 = f1 / (co2_g + epsilon)
    auc_per_co2 = auc / (co2_g + epsilon)

    # ======================================
    # SAVE RESULTS
    # ======================================
    results.append({
        "config": config_name,
        "tracker": row["tracker"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "co2_kg": emissions,
        "accuracy_per_co2": accuracy_per_co2,
        "f1_per_co2": f1_per_co2,
        "auc_per_co2": auc_per_co2
    })

# ==========================================
# SAVE CSV
# ==========================================
results_df = pd.DataFrame(results)

results_df.to_csv(
    "mnist_final_evaluation.csv",
    index=False
)

print("\nSaved: mnist_final_evaluation.csv")