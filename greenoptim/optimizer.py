import time
from pathlib import Path

import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .trackers.base         import BaseTracker
from .trackers.carbontracker import CarbonTrackerTracker
from .trackers.cumulator     import CumulatorTracker
from .trackers.eco2ai        import Eco2AITracker
from .datasets               import NoisyImageDataset, DNCNN_TRANSFORM


class GreenOptimizer:
    """
    Unified interface combining Optuna hyperparameter optimization
    with carbon/energy tracking for PyTorch models.

    Supports:
      - task="mnist"  → SimpleNN classification, accuracy metric
      - task="dncnn"  → DnCNN denoising, PSNR metric

    Any tracker implementing BaseTracker is accepted.
    Switching trackers requires only changing the tracker argument.

    Usage (MNIST):
        from greenoptim import GreenOptimizer
        from greenoptim.trackers import CodeCarbonTracker
        from greenoptim.models import SimpleNN

        optimizer = GreenOptimizer(
            model_class   = SimpleNN,
            task          = "mnist",
            tracker       = CodeCarbonTracker(),
            train_dataset = train_dataset,
            test_dataset  = test_dataset,
            n_trials      = 50,
            results_file  = "results/csv/results_optuna_mnist_cc.csv",
            study_name    = "mnist_greenoptim_cc",
            storage       = "sqlite:///mnist_cc.db"
        )
        optimizer.run()
        optimizer.print_pareto()

    Usage (DnCNN):
        from greenoptim import GreenOptimizer
        from greenoptim.trackers import CarbonTrackerTracker
        from greenoptim.models import DnCNN

        optimizer = GreenOptimizer(
            model_class      = DnCNN,
            task             = "dncnn",
            tracker          = CarbonTrackerTracker(),
            train_dataset    = None,   # built per trial (noise_factor varies)
            test_dataset     = None,
            dncnn_data_path  = TINY_IMAGENET_DIR / "val",
            n_trials         = 20,
            results_file     = "results/csv/results_optuna_dncnn_ct.csv",
            study_name       = "dncnn_greenoptim_ct",
            storage          = "sqlite:///dncnn_ct.db"
        )
        optimizer.run()
        optimizer.print_pareto()
    """

    def __init__(
        self,
        model_class,
        task: str,
        tracker: BaseTracker,
        train_dataset         = None,
        test_dataset          = None,
        dncnn_data_path       = None,
        n_trials: int         = 50,
        results_file: str     = "results/csv/greenoptim_results.csv",
        study_name: str       = "greenoptim_study",
        storage: str          = "sqlite:///greenoptim.db",
        device                = None
    ):
        self.model_class      = model_class
        self.task             = task.lower()
        self.tracker          = tracker
        self.train_dataset    = train_dataset
        self.test_dataset     = test_dataset
        self.dncnn_data_path  = dncnn_data_path
        self.n_trials         = n_trials
        self.results_file     = Path(results_file)
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self.study_name       = study_name
        self.storage          = storage
        self.study            = None

        self.device = device or (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        print(f"[GreenOptimizer] Device  : {self.device}")
        print(f"[GreenOptimizer] Task    : {self.task}")
        print(f"[GreenOptimizer] Tracker : {self.tracker.name}")
        print(f"[GreenOptimizer] Trials  : {self.n_trials}")

    # =========================
    # Hyperparameter sampling
    # =========================
    def _sample_hyperparams(self, trial) -> dict:
        """
        Sample trial hyperparameters.
        Shared params: batch_size, learning_rate, epochs.
        Task-specific params added below.
        """
        params = {}

        if self.task == "mnist":
            params["batch_size"] = trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]
            )
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True
            )
            params["epochs"] = trial.suggest_int(
                "epochs", 2, 6
            )
            params["hidden_size"] = trial.suggest_categorical(
                "hidden_size", [64, 128, 256]
            )

        elif self.task == "dncnn":
            params["batch_size"] = trial.suggest_categorical(
                "batch_size", [8, 16, 32]
            )
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 1e-4, 1e-2, log=True
            )
            params["epochs"] = trial.suggest_int(
                "epochs", 1, 4
            )
            params["channels"] = trial.suggest_categorical(
                "channels", [32, 64, 128]
            )
            params["noise_factor"] = trial.suggest_float(
                "noise_factor", 0.05, 0.30
            )

        return params

    # =========================
    # Dataset builder (DnCNN)
    # =========================
    def _build_dncnn_loaders(
        self,
        params: dict
    ) -> tuple:
        """
        DnCNN datasets must be rebuilt each trial because
        noise_factor is a hyperparameter that changes per trial.
        MNIST datasets are fixed and passed in at init time.
        """
        train_dataset = NoisyImageDataset(
            root_dir     = self.dncnn_data_path,
            noise_factor = params["noise_factor"],
            transform    = DNCNN_TRANSFORM
        )
        test_dataset = NoisyImageDataset(
            root_dir     = self.dncnn_data_path,
            noise_factor = params["noise_factor"],
            transform    = DNCNN_TRANSFORM
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size = params["batch_size"],
            shuffle    = True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size = params["batch_size"],
            shuffle    = False
        )
        return train_loader, test_loader

    # =========================
    # Model builder
    # =========================
    def _build_model(self, params: dict) -> nn.Module:
        if self.task == "mnist":
            return self.model_class(
                hidden_size=params.get("hidden_size", 128)
            ).to(self.device)

        elif self.task == "dncnn":
            return self.model_class(
                channels=params.get("channels", 64)
            ).to(self.device)

        raise ValueError(f"Unknown task: {self.task}")

    # =========================
    # Training loop
    # =========================
    def _train(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        epochs: int,
        trial_number: int
    ) -> None:

        is_cumulator     = isinstance(self.tracker, CumulatorTracker)
        is_carbontracker = isinstance(self.tracker, CarbonTrackerTracker)

        for epoch in range(epochs):

            if is_carbontracker:
                self.tracker._tracker.epoch_start()

            model.train()
            running_loss = 0.0

            for batch in train_loader:

                if is_cumulator:
                    self.tracker.step()

                if self.task == "mnist":
                    data, target = batch
                    data   = data.to(self.device)
                    target = target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss   = criterion(output, target)

                elif self.task == "dncnn":
                    noisy, clean = batch
                    noisy = noisy.to(self.device)
                    clean = clean.to(self.device)
                    optimizer.zero_grad()
                    output = model(noisy)
                    loss   = criterion(output, clean)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(
                f"Trial {trial_number} | "
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Loss: {avg_loss:.6f}"
            )

            if is_carbontracker:
                self.tracker._tracker.epoch_end()

    # =========================
    # Evaluation
    # =========================
    def _evaluate(
        self,
        model,
        test_loader,
        criterion
    ) -> float:

        model.eval()

        if self.task == "mnist":
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data   = data.to(self.device)
                    target = target.to(self.device)
                    output = model(data)
                    pred   = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
            return correct / len(self.test_dataset)

        elif self.task == "dncnn":
            total_psnr = 0.0
            count      = 0
            with torch.no_grad():
                for noisy, clean in test_loader:
                    noisy  = noisy.to(self.device)
                    clean  = clean.to(self.device)
                    output = model(noisy)
                    mse    = criterion(output, clean)
                    psnr   = 10 * torch.log10(1.0 / mse)
                    total_psnr += psnr.item()
                    count += 1
            return total_psnr / count

        raise ValueError(f"Unknown task: {self.task}")

    # =========================
    # Objective function
    # =========================
    def _objective(self, trial) -> tuple:

        params = self._sample_hyperparams(trial)

        # --- Build data loaders ---
        if self.task == "mnist":
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=params["batch_size"],
                shuffle=True
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=1000,
                shuffle=False
            )

        elif self.task == "dncnn":
            # Rebuilt per trial — noise_factor changes each trial
            train_loader, test_loader = self._build_dncnn_loaders(
                params
            )

        # --- Build model ---
        model     = self._build_model(params)
        criterion = (
            nn.CrossEntropyLoss()
            if self.task == "mnist"
            else nn.MSELoss()
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=params["learning_rate"]
        )

        # --- Tracker setup ---
        if isinstance(self.tracker, Eco2AITracker):
            self.tracker.set_trial(trial.number)

        if isinstance(self.tracker, CarbonTrackerTracker):
            self.tracker.build(params["epochs"])

        self.tracker.start()
        start_time = time.time()

        # --- Train ---
        self._train(
            model        = model,
            train_loader = train_loader,
            criterion    = criterion,
            optimizer    = optimizer,
            epochs       = params["epochs"],
            trial_number = trial.number
        )

        # --- Stop tracker ---
        execution_time = time.time() - start_time
        self.tracker.stop(execution_time=execution_time)

        # --- Evaluate ---
        performance = self._evaluate(
            model       = model,
            test_loader = test_loader,
            criterion   = criterion
        )

        co2_kg     = self.tracker.get_co2_kg()
        energy_kwh = self.tracker.get_energy_kwh()

        # --- Build result row ---
        row = {
            "trial":         trial.number,
            "tracker":       self.tracker.name,
            "task":          self.task,
            "device":        str(self.device),
            "batch_size":    params["batch_size"],
            "learning_rate": params["learning_rate"],
            "epochs":        params["epochs"],
            "time_sec":      execution_time,
            "energy_kwh":    energy_kwh,
            "co2_kg":        co2_kg
        }

        # Task-specific columns
        if self.task == "mnist":
            row["hidden_size"] = params.get("hidden_size")
            row["accuracy"]    = performance

        elif self.task == "dncnn":
            row["channels"]    = params.get("channels")
            row["noise_factor"] = params.get("noise_factor")
            row["psnr"]        = performance

        # --- Save ---
        df = pd.DataFrame([row])
        df.to_csv(
            self.results_file,
            mode   = "a",
            header = not self.results_file.exists(),
            index  = False
        )

        # --- Print summary ---
        perf_label = (
            "Accuracy" if self.task == "mnist"
            else "PSNR"
        )
        print(f"\n[Trial {trial.number}]")
        print(f"  {perf_label:<12}: {performance:.4f}")
        print(f"  CO2         : {co2_kg:.10f} kg")
        print(f"  Energy      : {energy_kwh:.8f} kWh")
        print(f"  Time        : {execution_time:.2f} sec\n")

        return performance, co2_kg

    # =========================
    # Run
    # =========================
    def run(self) -> None:
        self.study = optuna.create_study(
            directions     = ["maximize", "minimize"],
            study_name     = self.study_name,
            storage        = self.storage,
            load_if_exists = True
        )

        existing  = len(self.study.trials)
        remaining = self.n_trials - existing

        print(f"\n[GreenOptimizer] Existing trials: {existing}")

        if remaining <= 0:
            print("[GreenOptimizer] Target already reached.")
            return

        print(
            f"[GreenOptimizer] Running "
            f"{remaining} trials...\n"
        )

        self.study.optimize(
            self._objective,
            n_trials=remaining
        )

    # =========================
    # Print Pareto
    # =========================
    def print_pareto(self) -> None:
        if self.study is None:
            print("[GreenOptimizer] No study found. Run first.")
            return

        perf_label = (
            "Accuracy" if self.task == "mnist"
            else "PSNR"
        )

        print("\n===== PARETO FRONT =====")

        for trial in self.study.best_trials:
            print("----------------------------")
            print(f"Trial       : {trial.number}")
            print(
                f"{perf_label:<12}: "
                f"{trial.values[0]:.4f}"
            )
            print(
                f"CO2         : "
                f"{trial.values[1]:.10f} kg"
            )
            print(f"Params      : {trial.params}")

        print(f"\nTotal trials: {len(self.study.trials)}")
        print(f"Results     : {self.results_file}")