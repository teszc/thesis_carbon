import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. Configuration
# =========================
RESULTS_DIR = Path("results/csv")
PLOTS_DIR   = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# CSV file paths for each tool — MNIST
MNIST_FILES = {
    "CodeCarbon":    RESULTS_DIR / "results_mnist_cc.csv",
    "CarbonTracker": RESULTS_DIR / "results_mnist_ct.csv",
    "eco2ai":        RESULTS_DIR / "results_mnist_eco2ai.csv",
    "Cumulator":     RESULTS_DIR / "results_mnist_cumulator.csv",
}

# CSV file paths for each tool — DnCNN
DNCNN_FILES = {
    "CodeCarbon":    RESULTS_DIR / "results_dncnn_cc.csv",
    "CarbonTracker": RESULTS_DIR / "results_dncnn_ct.csv",
    "eco2ai":        RESULTS_DIR / "results_dncnn_eco2ai.csv",
    "Cumulator":     RESULTS_DIR / "results_dncnn_cumulator.csv",
}

# Colors per tool — consistent across all plots
TOOL_COLORS = {
    "CodeCarbon":    "#E63946",
    "CarbonTracker": "#457B9D",
    "eco2ai":        "#2A9D8F",
    "Cumulator":     "#E9C46A",
}

# Tools with unreliable CO2 on Apple Silicon
CO2_UNRELIABLE = ["eco2ai"]

# =========================
# 2. Helper: load & average
# =========================
def load_and_average(file_map: dict, required_cols: list) -> pd.DataFrame:
    """
    Load CSVs for each tool, keep required columns,
    convert co2_kg to co2_mg, and average all numeric runs.
    Returns a DataFrame with one row per tool.
    """
    rows = []

    for tool, path in file_map.items():
        if not path.exists():
            print(f"  [WARNING] File not found, skipping: {path}")
            continue

        df = pd.read_csv(path)

        available = [c for c in required_cols if c in df.columns]
        df = df[available].copy()

        if "co2_kg" in df.columns:
            df["co2_mg"] = df["co2_kg"] * 1e6

        avg = df.mean(numeric_only=True).to_dict()
        avg["tool"] = tool
        rows.append(avg)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("tool")


# =========================
# 3. Helper: bar chart
# =========================
def bar_chart(ax, data: pd.Series, title: str, ylabel: str,
              colors: dict, fmt: str = "{:.4f}",
              unreliable: list = None, lower_better: bool = False):

    tools  = [t for t in data.index if not pd.isna(data[t])]
    values = [data[t] for t in tools]

    if not tools:
        ax.text(0.5, 0.5, "No data available",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9, color="gray")
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    bars = ax.bar(
        tools,
        values,
        color=[colors.get(t, "#AAAAAA") for t in tools],
        edgecolor="white",
        linewidth=0.8,
        width=0.5
    )

    for bar, val, tool in zip(bars, values, tools):
        if unreliable and tool in unreliable:
            bar.set_hatch("//")
            bar.set_alpha(0.5)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.12,
                "⚠ unreliable",
                ha="center", va="bottom",
                fontsize=6, color="gray", style="italic"
            )

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            fmt.format(val),
            ha="center", va="bottom", fontsize=8
        )

    direction = "↓ lower is better" if lower_better else "↑ higher is better"
    ax.text(0.98, 0.97, direction,
            transform=ax.transAxes, fontsize=7,
            color="steelblue", ha="right", va="top",
            style="italic")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(range(len(tools)))
    ax.set_xticklabels(tools, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


# =========================
# 4. Helper: scatter pareto
# =========================
def scatter_pareto(ax, df: pd.DataFrame,
                   x_col: str, y_col: str,
                   xlabel: str, ylabel: str,
                   title: str, colors: dict,
                   unreliable: list = None):

    for tool, row in df.iterrows():
        if x_col not in row or y_col not in row:
            continue
        if pd.isna(row[x_col]) or pd.isna(row[y_col]):
            continue

        is_unreliable = unreliable and tool in unreliable

        ax.scatter(
            row[x_col], row[y_col],
            color=colors.get(tool, "#AAAAAA"),
            s=140,
            zorder=5,
            label=tool,
            marker="^" if is_unreliable else "o",
            alpha=0.5 if is_unreliable else 1.0,
            edgecolors="gray" if is_unreliable else colors.get(tool, "#AAAAAA"),
            linewidths=1.5 if is_unreliable else 0
        )

        ax.annotate(
            f"{tool}{'*' if is_unreliable else ''}",
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="gray" if is_unreliable else "black"
        )

    if unreliable:
        ax.text(
            0.02, 0.04,
            "* CO₂ unreliable (TDP estimate,\n  cannot distinguish trials)",
            transform=ax.transAxes, fontsize=6,
            color="gray", style="italic",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7)
        )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3, linestyle="--")


# =========================
# 5. Load data
# =========================
print("Loading MNIST results...")
MNIST_COLS = ["accuracy", "time_sec", "co2_kg"]
mnist_df   = load_and_average(MNIST_FILES, MNIST_COLS)
print(mnist_df)

print("\nLoading DnCNN results...")
DNCNN_COLS = ["accuracy", "time_sec", "co2_kg", "noise_std"]
dncnn_df   = load_and_average(DNCNN_FILES, DNCNN_COLS)
print(dncnn_df)

# =========================
# 6. MNIST plots — 2×2 grid
# =========================
print("\nGenerating MNIST plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "MNIST — SimpleNN: Cross-Tool Comparison",
    fontsize=14, fontweight="bold"
)

# Panel 1: Accuracy by tool
bar_chart(
    ax           = axes[0, 0],
    data         = mnist_df["accuracy"],
    title        = "Accuracy by Tool",
    ylabel       = "Accuracy",
    colors       = TOOL_COLORS,
    fmt          = "{:.4f}",
    lower_better = False
)

# Panel 2: CO₂ by tool (mg)
bar_chart(
    ax           = axes[0, 1],
    data         = mnist_df["co2_mg"],
    title        = "CO₂ Emissions by Tool",
    ylabel       = "CO₂ (mg)",
    colors       = TOOL_COLORS,
    fmt          = "{:.4f}",
    unreliable   = CO2_UNRELIABLE,
    lower_better = True
)

# Panel 3: Runtime by tool
bar_chart(
    ax           = axes[1, 0],
    data         = mnist_df["time_sec"],
    title        = "Runtime by Tool",
    ylabel       = "Time (seconds)",
    colors       = TOOL_COLORS,
    fmt          = "{:.2f}",
    lower_better = True
)

# Panel 4: Accuracy vs CO₂ scatter (mini Pareto)
scatter_pareto(
    ax         = axes[1, 1],
    df         = mnist_df,
    x_col      = "co2_mg",
    y_col      = "accuracy",
    xlabel     = "CO₂ (mg)",
    ylabel     = "Accuracy",
    title      = "Accuracy vs CO₂\n(mini Pareto — upper-left is best)",
    colors     = TOOL_COLORS,
    unreliable = CO2_UNRELIABLE
)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "plots_mnist_comparison.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/plots/plots_mnist_comparison.png")

# =========================
# 7. DnCNN plots — 2×2 grid
# =========================
print("\nGenerating DnCNN plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "DnCNN — TinyImageNet: Cross-Tool Comparison",
    fontsize=14, fontweight="bold"
)

psnr_col   = "accuracy"   # change to "psnr" if your DnCNN CSV uses that name
psnr_label = "PSNR (dB)"

# Panel 1: PSNR by tool
bar_chart(
    ax           = axes[0, 0],
    data         = dncnn_df[psnr_col],
    title        = "PSNR by Tool",
    ylabel       = psnr_label,
    colors       = TOOL_COLORS,
    fmt          = "{:.4f}",
    lower_better = False
)

# Panel 2: CO₂ by tool (mg)
bar_chart(
    ax           = axes[0, 1],
    data         = dncnn_df["co2_mg"],
    title        = "CO₂ Emissions by Tool",
    ylabel       = "CO₂ (mg)",
    colors       = TOOL_COLORS,
    fmt          = "{:.4f}",
    unreliable   = CO2_UNRELIABLE,
    lower_better = True
)

# Panel 3: Runtime by tool
bar_chart(
    ax           = axes[1, 0],
    data         = dncnn_df["time_sec"],
    title        = "Runtime by Tool",
    ylabel       = "Time (seconds)",
    colors       = TOOL_COLORS,
    fmt          = "{:.2f}",
    lower_better = True
)

# Panel 4: PSNR vs CO₂ scatter (mini Pareto)
scatter_pareto(
    ax         = axes[1, 1],
    df         = dncnn_df,
    x_col      = "co2_mg",
    y_col      = psnr_col,
    xlabel     = "CO₂ (mg)",
    ylabel     = psnr_label,
    title      = "PSNR vs CO₂\n(mini Pareto — upper-left is best)",
    colors     = TOOL_COLORS,
    unreliable = CO2_UNRELIABLE
)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "plots_dncnn_comparison.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/plots/plots_dncnn_comparison.png")