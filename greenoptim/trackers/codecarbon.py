from codecarbon import EmissionsTracker
from .base import BaseTracker


class CodeCarbonTracker(BaseTracker):
    """
    Wraps CodeCarbon EmissionsTracker.
    CO2 is returned directly by tracker.stop().
    Energy is estimated from Apple Silicon TDP since
    NVML cannot read Apple GPU power directly.
    """

    def __init__(
        self,
        output_dir: str = "logs/codecarbon",
        project_name: str = "greenoptim"
    ):
        self._output_dir  = output_dir
        self._project     = project_name
        self._tracker     = None
        self._co2_kg      = 0.0
        self._energy_kwh  = 0.0
        self._exec_time   = 0.0

    def start(self) -> None:
        self._tracker = EmissionsTracker(
            project_name = self._project,
            output_dir   = self._output_dir,
            log_level    = "error"
        )
        self._tracker.start()

    def stop(self, execution_time: float = 0.0) -> None:
        emissions = self._tracker.stop()
        self._exec_time = execution_time

        self._co2_kg = (
            emissions if emissions is not None
            else 0.0
        )

        # Energy: approximate from TDP since NVML
        # cannot read Apple Silicon GPU power
        self._energy_kwh = (
            self.TDP_W * execution_time
        ) / 3_600_000

        if self._co2_kg == 0.0:
            print(
                "[CodeCarbonTracker] "
                "Zero reading — using TDP fallback."
            )
            _, self._co2_kg = self.fallback_estimate(
                execution_time
            )

    def get_co2_kg(self) -> float:
        return self._co2_kg

    def get_energy_kwh(self) -> float:
        return self._energy_kwh