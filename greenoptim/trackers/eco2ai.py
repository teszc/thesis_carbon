import eco2ai
import pandas as pd
from pathlib import Path
from .base import BaseTracker


class Eco2AITracker(BaseTracker):
    """
    Wraps eco2ai Tracker.
    Reads CO2/energy from eco2ai's own CSV log after stop().
    Falls back to TDP estimate if eco2ai returns zero —
    common on Apple Silicon where GPU power is not measurable
    via NVML. eco2ai CO2 values on Apple Silicon are near-identical
    across trials and should be treated as indicative only.
    """

    def __init__(
        self,
        log_dir: str      = "logs/eco2ai",
        country_code: str = "IT"
    ):
        self._log_dir     = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._country     = country_code
        self._tracker     = None
        self._log_file    = None
        self._co2_kg      = 0.0
        self._energy_kwh  = 0.0
        self._trial_id    = 0

    def set_trial(self, trial_id: int) -> None:
        """Called by GreenOptimizer before each trial."""
        self._trial_id = trial_id
        self._log_file = (
            self._log_dir /
            f"eco2ai_trial_{trial_id}.csv"
        )

    def start(self) -> None:
        self._tracker = eco2ai.Tracker(
            project_name             = "GreenOptim",
            experiment_description   = f"Trial_{self._trial_id}",
            file_name                = str(self._log_file),
            alpha_2_code             = self._country
        )
        self._tracker.start()

    def stop(self, execution_time: float = 0.0) -> None:
        self._tracker.stop()

        try:
            log_df = pd.read_csv(self._log_file)
            last   = log_df.iloc[-1]

            self._energy_kwh = float(
                last["power_consumption(kWh)"]
            )
            self._co2_kg = float(
                last["CO2_emissions(kg)"]
            )

        except Exception as e:
            print(f"[Eco2AITracker] Could not parse log: {e}")
            self._energy_kwh = 0.0
            self._co2_kg     = 0.0

        if self._energy_kwh == 0.0 or self._co2_kg == 0.0:
            print(
                "[Eco2AITracker] "
                "Zero reading — using TDP fallback."
            )
            self._energy_kwh, self._co2_kg = (
                self.fallback_estimate(execution_time)
            )

    def get_co2_kg(self) -> float:
        return self._co2_kg

    def get_energy_kwh(self) -> float:
        return self._energy_kwh