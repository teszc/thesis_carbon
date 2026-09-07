import os
import re
from pathlib import Path
from carbontracker.tracker import CarbonTracker as _CarbonTracker
from .base import BaseTracker


class CarbonTrackerTracker(BaseTracker):
    """
    Wraps CarbonTracker.
    Uses epoch_start()/epoch_end() internally.
    Parses log file after training to extract CO2/energy.
    Falls back to TDP estimate if log returns zero
    (common on Apple Silicon for short runs).
    """

    def __init__(
        self,
        log_dir: str = "logs/carbontracker",
        epochs: int  = 3
    ):
        self._log_dir    = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._epochs     = epochs
        self._tracker    = None
        self._co2_kg     = 0.0
        self._energy_kwh = 0.0

    def build(self, epochs: int) -> "_CarbonTracker":
        """
        Build a new CarbonTracker instance for a trial.
        Called by GreenOptimizer before each trial's training loop.
        """
        self._epochs  = epochs
        self._tracker = _CarbonTracker(
            epochs        = epochs,
            monitor_epochs = epochs,
            log_dir        = str(self._log_dir),
            verbose        = 0
        )
        return self._tracker

    def start(self) -> None:
        # CarbonTracker uses epoch_start/epoch_end
        # so start() is a no-op here
        pass

    def stop(self, execution_time: float = 0.0) -> None:
        self._tracker.stop()
        self._parse_log(execution_time)

    def _parse_log(self, execution_time: float) -> None:
        log_files = sorted(
            [
                f for f in os.listdir(self._log_dir)
                if f.endswith(".log")
            ],
            key=lambda f: os.path.getmtime(
                self._log_dir / f
            )
        )

        if not log_files:
            print(
                "[CarbonTrackerTracker] "
                "No log file found — using TDP fallback."
            )
            self._energy_kwh, self._co2_kg = (
                self.fallback_estimate(execution_time)
            )
            return

        latest = self._log_dir / log_files[-1]

        with open(latest, "r") as f:
            content = f.read()

        intensity = self.INTENSITY_G
        intensity_match = re.search(
            r"Average carbon intensity.*?([\d.]+)\s*gCO2eq/kWh",
            content
        )
        if intensity_match:
            intensity = float(intensity_match.group(1))

        energy_match = re.search(
            r"Actual consumption[\s\S]*?Energy:\s*([\d.eE+\-]+)\s*kWh",
            content
        )
        co2_match = re.search(
            r"Actual consumption[\s\S]*?CO2eq:\s*([\d.eE+\-]+)\s*g",
            content
        )

        if energy_match and co2_match:
            energy = float(energy_match.group(1))
            co2_g  = float(co2_match.group(1))

            if energy > 0 and co2_g > 0:
                self._energy_kwh = energy
                self._co2_kg     = co2_g / 1000
                return

        print(
            "[CarbonTrackerTracker] "
            "Zero values in log — using TDP fallback."
        )
        self._energy_kwh, self._co2_kg = (
            self.fallback_estimate(execution_time)
        )

    def get_co2_kg(self) -> float:
        return self._co2_kg

    def get_energy_kwh(self) -> float:
        return self._energy_kwh