from pathlib import Path
import torch

# =========================
# Root project directory
# thesis-carbon/
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]

# =========================
# Data directories
# =========================

# thesis-carbon/data/
DATA_DIR = ROOT_DIR / "data"

# MNIST dataset location
MNIST_DIR = DATA_DIR / "mnist"

# thesis-carbon/datasets/
DATASETS_DIR = ROOT_DIR / "datasets"

# thesis-carbon/datasets/tiny-imagenet-200/
TINY_IMAGENET_DIR = (
    DATASETS_DIR / "tiny-imagenet-200"
)

# =========================
# Results & logs
# =========================

# thesis-carbon/results/
RESULTS_DIR = ROOT_DIR / "results"

# thesis-carbon/logs/
LOGS_DIR = ROOT_DIR / "logs"

# Create folders automatically
RESULTS_DIR.mkdir(exist_ok=True)

(RESULTS_DIR / "csv").mkdir(
    parents=True,
    exist_ok=True
)

(RESULTS_DIR / "plots").mkdir(
    parents=True,
    exist_ok=True
)

(RESULTS_DIR / "optuna_dbs").mkdir(
    parents=True,
    exist_ok=True
)

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================
# Device configuration
# =========================
if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")

# =========================
# Debug info
# =========================
print(f"Using device: {DEVICE}")
print(f"Project root: {ROOT_DIR}")
print(f"TinyImageNet path: {TINY_IMAGENET_DIR}")