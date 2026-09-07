## Quickstart — DnCNN

```python
from pathlib import Path
from greenoptim import GreenOptimizer
from greenoptim.models import DnCNN
from greenoptim.trackers import CodeCarbonTracker

TINY_IMAGENET_VAL = Path("data/tiny-imagenet-200/val")

optimizer = GreenOptimizer(
    model_class     = DnCNN,
    task            = "dncnn",
    tracker         = CodeCarbonTracker(),
    train_dataset   = None,          # built per trial
    test_dataset    = None,          # built per trial
    dncnn_data_path = TINY_IMAGENET_VAL,
    n_trials        = 20,
    results_file    = "results/csv/results_optuna_dncnn_cc.csv",
    study_name      = "dncnn_greenoptim_cc",
    storage         = "sqlite:///dncnn_cc.db"
)

optimizer.run()
optimizer.print_pareto()
```

Note: DnCNN passes `train_dataset=None` and `test_dataset=None`
because the dataset is rebuilt each trial (noise_factor is a
hyperparameter). Pass `dncnn_data_path` instead.