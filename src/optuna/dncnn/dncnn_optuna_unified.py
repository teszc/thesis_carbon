import pandas as pd
import os

# =====================================================
# 1. Input files
# =====================================================
codecarbon_file = "results_optuna_dncnn_codecarbon.csv"
eco2ai_file = "results_optuna_dncnn_eco2ai.csv"
cumulator_file = "results_optuna_dncnn_cumulator.csv"

# =====================================================
# 2. Load CSVs
# =====================================================
df_codecarbon = pd.read_csv(codecarbon_file)
df_eco2ai = pd.read_csv(eco2ai_file)
df_cumulator = pd.read_csv(cumulator_file)

# =====================================================
# 3. Add tracker labels
# =====================================================
df_codecarbon["tracker"] = "codecarbon"
df_eco2ai["tracker"] = "eco2ai"
df_cumulator["tracker"] = "cumulator"

# =====================================================
# 4. Ensure consistent columns
# =====================================================

def fix_columns(df):

    # energy missing in codecarbon
    if "energy_kwh" not in df.columns:
        df["energy_kwh"] = None

    # ensure column order consistency
    cols = [
        "tracker",
        "trial",
        "batch_size",
        "learning_rate",
        "epochs",
        "channels",
        "noise_factor",
        "psnr",
        "time_sec",
        "energy_kwh",
        "co2_kg"
    ]

    return df[cols]

df_codecarbon = fix_columns(df_codecarbon)
df_eco2ai = fix_columns(df_eco2ai)
df_cumulator = fix_columns(df_cumulator)

# =====================================================
# 5. Merge datasets
# =====================================================
dncnn_unified = pd.concat(
    [
        df_codecarbon,
        df_eco2ai,
        df_cumulator
    ],
    ignore_index=True
)

# =====================================================
# 6. Save unified dataset
# =====================================================
output_file = "dncnn_optuna_unified.csv"

dncnn_unified.to_csv(
    output_file,
    index=False
)

# =====================================================
# 7. Summary
# =====================================================
print("\n====================================")
print("DnCNN unified Optuna dataset created")
print("====================================")

print(f"\nSaved file: {output_file}")

print("\nShape:")

print(dncnn_unified.shape)

print("\nColumns:")

print(dncnn_unified.columns.tolist())

print("\nPreview:")

print(dncnn_unified.head())