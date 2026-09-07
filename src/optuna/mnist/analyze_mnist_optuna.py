import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# =====================================================
# PATHS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[3]

RESULTS_DIR = ROOT_DIR / "results" / "csv"
ANALYSIS_DIR = ROOT_DIR / "analysis" / "mnist"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# INPUT FILES
# =====================================================

FILES = {
    "CodeCarbon": RESULTS_DIR / "results_optuna_mnist_codecarbon.csv",
    "CarbonTracker": RESULTS_DIR / "results_optuna_mnist_carbontracker.csv",
    "Eco2AI": RESULTS_DIR / "results_optuna_mnist_eco2ai.csv",
    "Cumulator": RESULTS_DIR / "results_optuna_mnist_cumulator.csv"
}

# =====================================================
# LOAD DATA
# =====================================================

dfs = []

for tool, path in FILES.items():

    if not path.exists():
        print(f"Missing: {path}")
        continue

    df = pd.read_csv(path)

    df["tool"] = tool

    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(combined_df)} trials.")

# =====================================================
# CLEAN DATA
# =====================================================

combined_df = combined_df.dropna(
    subset=[
        "accuracy",
        "energy_kwh",
        "co2_kg",
        "time_sec"
    ]
)

combined_df = combined_df[
    combined_df["energy_kwh"] > 0
]

combined_df = combined_df[
    combined_df["co2_kg"] > 0
]

combined_df = combined_df.reset_index(drop=True)

# =====================================================
# UNIT CONVERSIONS
# =====================================================

combined_df["accuracy_pct"] = combined_df["accuracy"] * 100

combined_df["energy_wh"] = combined_df["energy_kwh"] * 1000

combined_df["co2_g"] = combined_df["co2_kg"] * 1000

# =====================================================
# RELATIVE ACCURACY
# =====================================================

best_accuracy = combined_df["accuracy_pct"].max()

combined_df["relative_accuracy"] = (
    combined_df["accuracy_pct"] / best_accuracy
)

# =====================================================
# AER METRICS
# =====================================================

combined_df["aer_energy"] = (
    combined_df["accuracy_pct"] /
    combined_df["energy_wh"]
)

combined_df["aer_co2"] = (
    combined_df["accuracy_pct"] /
    combined_df["co2_g"]
)

combined_df["aer_time"] = (
    combined_df["accuracy_pct"] /
    combined_df["time_sec"]
)

combined_df["aer_relative"] = (
    combined_df["relative_accuracy"] /
    combined_df["energy_wh"]
)

# =====================================================
# SAVE FULL DATASET
# =====================================================

combined_csv = ANALYSIS_DIR / "mnist_unified.csv"

combined_df.to_csv(combined_csv, index=False)

print(f"Saved {combined_csv}")

# =====================================================
# SUMMARY TABLE
# =====================================================

summary = (
    combined_df
    .groupby("tool")
    .agg(

        trials=("trial", "count"),

        mean_accuracy=("accuracy_pct", "mean"),
        max_accuracy=("accuracy_pct", "max"),

        mean_energy=("energy_wh", "mean"),
        min_energy=("energy_wh", "min"),
        max_energy=("energy_wh", "max"),

        mean_co2=("co2_g", "mean"),

        mean_time=("time_sec", "mean"),

        mean_aer=("aer_energy", "mean"),
        median_aer=("aer_energy", "median"),
        max_aer=("aer_energy", "max"),
        std_aer=("aer_energy", "std")

    )
)

summary = summary.round(4)

summary_path = ANALYSIS_DIR / "mnist_summary.csv"

summary.to_csv(summary_path)

print(f"Saved {summary_path}")

# =====================================================
# BEST TRIAL OF EACH TOOL
# =====================================================

best_trials = (
    combined_df
    .sort_values("aer_energy", ascending=False)
    .groupby("tool")
    .first()
)

best_trials_path = ANALYSIS_DIR / "mnist_best_trials.csv"

best_trials.to_csv(best_trials_path)

print(f"Saved {best_trials_path}")

# =====================================================
# FIGURE 1
# Accuracy vs Energy
# =====================================================

plt.figure(figsize=(10,6))

for tool in combined_df["tool"].unique():

    subset = combined_df[combined_df["tool"] == tool]

    plt.scatter(
        subset["energy_wh"],
        subset["accuracy_pct"],
        s=60,
        alpha=0.7,
        label=tool
    )

plt.xlabel("Energy Consumption (Wh)")
plt.ylabel("Accuracy (%)")
plt.title("MNIST: Accuracy vs Energy")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR / "accuracy_vs_energy.png",
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 2
# Accuracy vs CO2
# =====================================================

plt.figure(figsize=(10,6))

for tool in combined_df["tool"].unique():

    subset = combined_df[combined_df["tool"] == tool]

    plt.scatter(
        subset["co2_g"],
        subset["accuracy_pct"],
        s=60,
        alpha=0.7,
        label=tool
    )

plt.xlabel("CO₂ Emissions (g)")
plt.ylabel("Accuracy (%)")
plt.title("MNIST: Accuracy vs CO₂")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR / "accuracy_vs_co2.png",
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 3
# Energy Distribution
# =====================================================

plt.figure(figsize=(8,6))

combined_df.boxplot(
    column="energy_wh",
    by="tool"
)

plt.suptitle("")
plt.title("Energy Consumption Distribution")
plt.ylabel("Energy (Wh)")

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR / "energy_boxplot.png",
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 4
# CO2 Distribution
# =====================================================

plt.figure(figsize=(8,6))

combined_df.boxplot(
    column="co2_g",
    by="tool"
)

plt.suptitle("")
plt.title("CO₂ Emission Distribution")
plt.ylabel("CO₂ (g)")

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR / "co2_boxplot.png",
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 5
# AER Distribution
# =====================================================

plt.figure(figsize=(8,6))

combined_df.boxplot(
    column="aer_energy",
    by="tool"
)

plt.suptitle("")
plt.title("Accuracy-Efficiency Ratio Distribution")
plt.ylabel("Accuracy (%) / Wh")

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR / "aer_boxplot.png",
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 6
# Mean AER
# =====================================================

mean_aer = (
    combined_df
    .groupby("tool")["aer_energy"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8,6))

mean_aer.plot(kind="bar")

plt.ylabel("Mean AER (Accuracy % / Wh)")
plt.title("Mean Accuracy-Efficiency Ratio")

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR / "mean_aer.png",
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 7
# Best AER
# =====================================================

best_aer = (
    combined_df
    .groupby("tool")["aer_energy"]
    .max()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8,6))

best_aer.plot(kind="bar")

plt.ylabel("Best AER (Accuracy % / Wh)")
plt.title("Best Accuracy-Efficiency Ratio")

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR / "best_aer.png",
    dpi=300
)

plt.close()

# =====================================================
# FIGURE 8
# Relative Accuracy vs Energy
# =====================================================

plt.figure(figsize=(10,6))

for tool in combined_df["tool"].unique():

    subset = combined_df[combined_df["tool"] == tool]

    plt.scatter(
        subset["energy_wh"],
        subset["relative_accuracy"],
        alpha=0.7,
        s=60,
        label=tool
    )

plt.xlabel("Energy Consumption (Wh)")
plt.ylabel("Relative Accuracy")
plt.title("Relative Accuracy vs Energy")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    ANALYSIS_DIR / "relative_accuracy_vs_energy.png",
    dpi=300
)

plt.close()