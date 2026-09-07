import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. File paths
# =====================================================
FILES = {

    # =========================
    # MNIST
    # =========================
    "mnist_codecarbon": "results_optuna_mnist.csv",
    "mnist_eco2ai": "results_optuna_mnist_eco2ai.csv",
    "mnist_cumulator": "results_optuna_mnist_cumulator.csv",

    # =========================
    # DnCNN
    # =========================
    "dncnn_codecarbon": "results_optuna_dncnn.csv",
    "dncnn_eco2ai": "results_optuna_dncnn_eco2ai.csv",
    "dncnn_cumulator": "results_optuna_dncnn_cumulator.csv"
}

# =====================================================
# 2. Constants
# =====================================================
EPSILON = 1e-10

# =====================================================
# 3. Helper Functions
# =====================================================
def normalize(series):

    return (
        (series - series.min())
        /
        (series.max() - series.min() + EPSILON)
    )

# =====================================================
# AER computation
# =====================================================
def compute_aer(df, performance_col="performance_normalized"):

    baseline_idx = df["co2_kg"].idxmin()

    baseline_perf = df.loc[
        baseline_idx,
        performance_col
    ]

    baseline_co2 = df.loc[
        baseline_idx,
        "co2_kg"
    ]

    df = df.copy()

    df["AER"] = (
        (df[performance_col] - baseline_perf)
        /
        (df["co2_kg"] - baseline_co2 + EPSILON)
    )

    return df

# =====================================================
# TRUE Pareto Front
# =====================================================
def pareto_front(df):

    pareto = []

    for i, row in df.iterrows():

        dominated = False

        for j, other in df.iterrows():

            if i == j:
                continue

            better_or_equal_perf = (
                other["performance_normalized"]
                >= row["performance_normalized"]
            )

            lower_or_equal_co2 = (
                other["co2_kg"]
                <= row["co2_kg"]
            )

            strictly_better = (
                other["performance_normalized"]
                > row["performance_normalized"]
                or
                other["co2_kg"]
                < row["co2_kg"]
            )

            if (
                better_or_equal_perf
                and
                lower_or_equal_co2
                and
                strictly_better
            ):

                dominated = True
                break

        if not dominated:
            pareto.append(row)

    return pd.DataFrame(pareto)

# =====================================================
# 4. Load CSV files
# =====================================================
all_dfs = []

print("\n====================================")
print("Loading CSV files")
print("====================================")

for key, file in FILES.items():

    if os.path.exists(file):

        print(f"Loaded: {file}")

        df = pd.read_csv(file)

        model, tracker = key.split("_", 1)

        df["model"] = model.upper()
        df["tracker"] = tracker

        # ---------------------------------
        # Standardize performance
        # ---------------------------------
        if "accuracy" in df.columns:
            df["performance"] = df["accuracy"]

        elif "psnr" in df.columns:
            df["performance"] = df["psnr"]

        else:
            print(f"WARNING: No performance column in {file}")
            continue

        # ---------------------------------
        # Add missing energy column
        # ---------------------------------
        if "energy_kwh" not in df.columns:
            df["energy_kwh"] = np.nan

        all_dfs.append(df)

    else:
        print(f"Missing file: {file}")

# =====================================================
# 5. Merge datasets
# =====================================================
results_df = pd.concat(
    all_dfs,
    ignore_index=True
)

print("\nMerged dataset shape:")
print(results_df.shape)

# =====================================================
# 6. Normalize performance
# =====================================================
results_df["performance_normalized"] = (
    results_df
    .groupby("model")["performance"]
    .transform(normalize)
)

# =====================================================
# 7. Compute AER
# =====================================================
final_dfs = []

for (model, tracker), group in results_df.groupby([
    "model",
    "tracker"
]):

    group = compute_aer(group)

    final_dfs.append(group)

results_df = pd.concat(
    final_dfs,
    ignore_index=True
)

# =====================================================
# 8. Compute TRUE Pareto Fronts
# =====================================================
pareto_dfs = []

for (model, tracker), group in results_df.groupby([
    "model",
    "tracker"
]):

    pareto = pareto_front(group)

    pareto["pareto_optimal"] = True

    pareto_dfs.append(pareto)

pareto_df = pd.concat(
    pareto_dfs,
    ignore_index=True
)

pareto_df.to_csv(
    "pareto_optimal_trials.csv",
    index=False
)

print("\nSaved Pareto-optimal trials.")

# =====================================================
# 9. Save unified dataset
# =====================================================
results_df.to_csv(
    "all_results.csv",
    index=False
)

print("\nSaved unified dataset: all_results.csv")

# =====================================================
# 10. Summary statistics
# =====================================================
summary_table = (
    results_df
    .groupby(["model", "tracker"])
    .agg({
        "performance": ["mean", "max"],
        "co2_kg": ["mean", "min"],
        "energy_kwh": ["mean", "min"],
        "AER": ["mean", "max"]
    })
)

print("\n====================================")
print("SUMMARY TABLE")
print("====================================")

print(summary_table)

summary_table.to_csv(
    "summary_statistics.csv"
)

# =====================================================
# 11. Best trials table
# =====================================================
best_trials = (
    results_df
    .sort_values("AER", ascending=False)
    .groupby(["model", "tracker"])
    .head(3)
)

print("\n====================================")
print("BEST TRIALS")
print("====================================")

print(best_trials[[
    "model",
    "tracker",
    "trial",
    "performance",
    "co2_kg",
    "AER"
]])

best_trials.to_csv(
    "best_trials.csv",
    index=False
)

# =====================================================
# 12. Performance vs CO2 Plot
# =====================================================
plt.figure(figsize=(10, 6))

for tracker in results_df["tracker"].unique():

    subset = results_df[
        results_df["tracker"] == tracker
    ]

    plt.scatter(
        subset["co2_kg"],
        subset["performance_normalized"],
        label=tracker,
        alpha=0.7
    )

plt.xlabel("CO2 Emissions (kg)")
plt.ylabel("Normalized Performance")
plt.title("Performance vs CO2 Emissions")

plt.legend()

plt.grid(True)

plt.savefig(
    "plot_performance_vs_co2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# 13. Runtime vs CO2 Plot
# =====================================================
plt.figure(figsize=(10, 6))

for tracker in results_df["tracker"].unique():

    subset = results_df[
        results_df["tracker"] == tracker
    ]

    plt.scatter(
        subset["time_sec"],
        subset["co2_kg"],
        label=tracker,
        alpha=0.7
    )

plt.xlabel("Runtime (seconds)")
plt.ylabel("CO2 Emissions (kg)")
plt.title("Runtime vs CO2 Emissions")

plt.legend()

plt.grid(True)

plt.savefig(
    "plot_runtime_vs_co2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# 14. AER Distribution Plot
# =====================================================
plt.figure(figsize=(10, 6))

for tracker in results_df["tracker"].unique():

    subset = results_df[
        results_df["tracker"] == tracker
    ]

    plt.hist(
        subset["AER"],
        bins=20,
        alpha=0.5,
        label=tracker
    )

plt.xlabel("AER")
plt.ylabel("Frequency")
plt.title("AER Distribution")

plt.legend()

plt.grid(True)

plt.savefig(
    "plot_aer_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# 15. TRUE Pareto Front Plot
# =====================================================
plt.figure(figsize=(10, 6))

for tracker in results_df["tracker"].unique():

    subset = results_df[
        results_df["tracker"] == tracker
    ]

    pareto_subset = pareto_df[
        pareto_df["tracker"] == tracker
    ]

    # ---------------------------------
    # All trials
    # ---------------------------------
    plt.scatter(
        subset["co2_kg"],
        subset["performance_normalized"],
        alpha=0.3
    )

    # ---------------------------------
    # Pareto-optimal front
    # ---------------------------------
    pareto_subset = pareto_subset.sort_values("co2_kg")

    plt.plot(
        pareto_subset["co2_kg"],
        pareto_subset["performance_normalized"],
        marker='o',
        linewidth=3,
        label=f"{tracker} Pareto"
    )

plt.xlabel("CO2 Emissions (kg)")
plt.ylabel("Normalized Performance")

plt.title("TRUE Pareto-optimal Tradeoff Fronts")

plt.legend()

plt.grid(True)

plt.savefig(
    "plot_true_pareto_front.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# 16. Correlation analysis
# =====================================================
correlation_table = (
    results_df[[
        "performance_normalized",
        "co2_kg",
        "energy_kwh",
        "time_sec",
        "AER"
    ]]
    .corr()
)

print("\n====================================")
print("CORRELATION TABLE")
print("====================================")

print(correlation_table)

correlation_table.to_csv(
    "correlation_table.csv"
)

# =====================================================
# 17. Best configuration per tracker
# =====================================================
print("\n====================================")
print("BEST CONFIGURATION PER TRACKER")
print("====================================")

best_configs = (
    results_df
    .sort_values("AER", ascending=False)
    .groupby(["model", "tracker"])
    .first()
)

print(best_configs[[
    "performance",
    "co2_kg",
    "energy_kwh",
    "AER"
]])

best_configs.to_csv(
    "best_configurations.csv"
)

# =====================================================
# 18. Overall statistics
# =====================================================
print("\n====================================")
print("OVERALL STATISTICS")
print("====================================")

print(f"Total trials: {len(results_df)}")

print(f"Average CO2: {results_df['co2_kg'].mean()}")

print(f"Average energy: {results_df['energy_kwh'].mean()}")

print(f"Average AER: {results_df['AER'].mean()}")

# =====================================================
# 19. Save Pareto summary
# =====================================================
pareto_summary = (
    pareto_df
    .groupby(["model", "tracker"])
    .size()
    .reset_index(name="pareto_trial_count")
)

pareto_summary.to_csv(
    "pareto_summary.csv",
    index=False
)

print("\nSaved Pareto summary.")

# =====================================================
# 20. Done
# =====================================================
print("\n====================================")
print("Analysis complete.")
print("====================================")

print("\nGenerated CSV files:")

print("- all_results.csv")
print("- summary_statistics.csv")
print("- best_trials.csv")
print("- pareto_optimal_trials.csv")
print("- pareto_summary.csv")
print("- correlation_table.csv")
print("- best_configurations.csv")

print("\nGenerated plots:")

print("- plot_performance_vs_co2.png")
print("- plot_runtime_vs_co2.png")
print("- plot_aer_distribution.png")
print("- plot_true_pareto_front.png")