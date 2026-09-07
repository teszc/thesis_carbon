import time
from .base import BaseTracker


class CumulatorTracker(BaseTracker):
    """
    Custom TDP-based energy tracker that accumulates
    energy by sampling wall-clock time at each training step.
    More granular than a simple start/stop estimate because
    it accounts for variable step durations.
    No external dependencies required.
    """

    def __init__(
        self,
        power_watts: float  = 20.0,
        intensity_g: float  = 233.0
    ):
        self.TDP_W       = power_watts
        self.INTENSITY_G = intensity_g
        self._energy_j   = 0.0
        self._last_time  = None
        self._running    = False

    def start(self) -> None:
        self._energy_j  = 0.0
        self._last_time = time.time()
        self._running   = True

    def step(self) -> None:
        """
        Call once per training batch to accumulate energy.
        More accurate than a single start/stop measurement
        as it captures time between batches.
        """
        if not self._running:
            return
        now = time.time()
        dt  = now - self._last_time
        self._energy_j += self.TDP_W * dt
        self._last_time = now

    def stop(self, execution_time: float = 0.0) -> None:
        if self._running:
            self.step()
            self._running = False

    def get_energy_kwh(self) -> float:
        return self._energy_j / 3_600_000

    def get_co2_kg(self) -> float:
        return (
            self.get_energy_kwh() * self.INTENSITY_G
        ) / 1000