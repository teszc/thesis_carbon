from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """
    Abstract base class for all carbon/energy trackers.
    Every tracker must implement start(), stop(),
    get_co2_kg(), and get_energy_kwh().
    The GreenOptimizer talks only to this interface —
    it does not care which tracker is underneath.
    """

    # Fallback constants for Apple Silicon TDP estimation
    TDP_W       = 20.0    # Apple M-series TDP in watts
    INTENSITY_G = 233.0   # Italy grid intensity gCO2eq/kWh

    @abstractmethod
    def start(self) -> None:
        """Start tracking emissions for one trial."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop tracking and finalise measurements."""
        ...

    @abstractmethod
    def get_co2_kg(self) -> float:
        """Return CO2 emissions in kg for the last tracked period."""
        ...

    @abstractmethod
    def get_energy_kwh(self) -> float:
        """Return energy consumption in kWh for the last tracked period."""
        ...

    def fallback_estimate(
        self,
        execution_time: float
    ) -> tuple[float, float]:
        """
        TDP-based fallback when the tracker returns zero.
        Used on Apple Silicon where hardware power cannot
        be read directly by NVML-based tools.
        Returns (energy_kwh, co2_kg).
        """
        energy_kwh = (self.TDP_W * execution_time) / 3_600_000
        co2_kg     = (energy_kwh * self.INTENSITY_G) / 1000
        return energy_kwh, co2_kg

    @property
    def name(self) -> str:
        """Human-readable tracker name for CSV and logging."""
        return self.__class__.__name__