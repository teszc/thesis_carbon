import pandas as pd
import os

# =====================================================
# 1. Load files
# =====================================================
codecarbon_file = "results_optuna_mnist_codecarbon.csv"

eco2ai_file = "results_optuna_mnist_eco2ai.csv"

cumulator_file = "results_optuna_mnist_cumulator.csv"

# =====================================================
# 2. Read CSVs
# =====================================================
df_codecarbon = pd.read_csv(codecarbon_file)

df_eco2ai = pd.read_csv(eco2ai_file)

df_cumulator = pd.read_csv(cumulator_file)

# =====================================================
# 3. Add tracker names
# =====================================================
df_codecarbon["tracker"] = "codecarbon"

df_eco2ai["tracker"] = "eco2ai"

df_cumulator["tracker"] = "cumulator"

# =====================================================
# 4. Add missing columns
# =====================================================

# ---------------------------------
# CodeCarbon has no energy column
# ---------------------------------
if "energy_kwh" not in df_codecarbon.columns:

    df_codecarbon["energy_kwh"] = None

# ---------------------------------
# CodeCarbon may not have hidden_size
# ---------------------------------
if "hidden_size" not in df_codecarbon.columns:

    df_codecarbon["hidden_size"] = None

# =====================================================
# 5. Standardize column order
# =====================================================
columns = [

    "tracker",

    "trial",

    "batch_size",

    "learning_rate",

    "hidden_size",

    "epochs",

    "accuracy",

    "time_sec",

    "energy_kwh",

    "co2_kg"
]

df_codecarbon = df_codecarbon[columns]

df_eco2ai = df_eco2ai[columns]

df_cumulator = df_cumulator[columns]

# =====================================================
# 6. Merge all datasets
# =====================================================
mnist_unified = pd.concat(

    [
        df_codecarbon,
        df_eco2ai,
        df_cumulator
    ],

    ignore_index=True
)

# =====================================================
# 7. Save unified file
# =====================================================
output_file = "mnist_optuna_unified.csv"

mnist_unified.to_csv(

    output_file,

    index=False
)

# =====================================================
# 8. Print summary
# =====================================================
print("\n====================================")
print("MNIST unified Optuna dataset created")
print("====================================")

print(f"\nSaved file: {output_file}")

print("\nDataset shape:")

print(mnist_unified.shape)

print("\nColumns:")

print(mnist_unified.columns.tolist())

print("\nFirst rows:")

print(mnist_unified.head())