"""Green AI carbon footprint and energy consumption tracking for OmniForge workloads."""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

import psutil

from omniforge.core.constants import CARBON_INTENSITY_BY_COUNTRY
from omniforge.core.logging import get_logger
from omniforge.core.types import CarbonReport
from omniforge.core.utils import is_package_available

logger = get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


class CarbonTracker:
    """Context manager measuring compute energy (kWh) and emissions (kgCO2eq).

    Automatically integrates with ``codecarbon`` when installed and enabled;
    otherwise estimates energy from CPU utilization, memory, and power assumptions.
    GPU power is included only when a reader supplies average power attributable to
    the operation. Device availability and TDP alone do not establish GPU use.
    """

    def __init__(
        self,
        operation_name: str = "omniforge_operation",
        *,
        country_iso: str = "DEU",
        grid_intensity_g_per_kwh: float | None = None,
        pue: float = 1.25,
        cpu_tdp_watts: float = 65.0,
        gpu_tdp_watts: float = 250.0,
        gpu_power_reader: Callable[[], float] | None = None,
        use_codecarbon: bool = False,
        enabled: bool = True,
    ) -> None:
        self.operation_name = operation_name
        self.country_iso = country_iso.upper()
        self.grid_intensity = (
            float(grid_intensity_g_per_kwh)
            if grid_intensity_g_per_kwh is not None
            else CARBON_INTENSITY_BY_COUNTRY.get(self.country_iso, 475.0)
        )
        self.pue = max(1.0, float(pue))
        self.cpu_tdp_watts = float(cpu_tdp_watts)
        self.gpu_tdp_watts = float(gpu_tdp_watts)
        self._gpu_power_reader = gpu_power_reader
        self.use_codecarbon = use_codecarbon and is_package_available("codecarbon")
        self.enabled = enabled

        self._start_wall: float = 0.0
        self._start_cpu_time: float = 0.0
        self._started_at: str = ""
        self._process = psutil.Process()
        self._cc_tracker: Any = None
        self.report: CarbonReport | None = None

    def start(self) -> None:
        """Begin measuring energy and carbon emissions."""
        if not self.enabled:
            return
        self._started_at = datetime.now(UTC).isoformat()
        self._start_wall = time.perf_counter()
        cpu_times = self._process.cpu_times()
        self._start_cpu_time = float(cpu_times.user + cpu_times.system)
        if self.use_codecarbon:
            try:
                from codecarbon import OfflineEmissionsTracker

                self._cc_tracker = OfflineEmissionsTracker(
                    project_name=self.operation_name,
                    country_iso_code=self.country_iso,
                    save_to_file=False,
                    tracking_mode="process",
                    log_level="error",
                )
                self._cc_tracker.start()
            except Exception as exc:
                logger.warning("codecarbon_fallback", error_type=type(exc).__name__)
                self._cc_tracker = None

    def stop(self) -> CarbonReport:
        """Stop measurement and produce a :class:`CarbonReport`."""
        finished_at = datetime.now(UTC).isoformat()
        if not self.enabled:
            self.report = CarbonReport(
                operation_name=self.operation_name,
                tracker_backend="disabled",
                metadata={"energy_source": "unmeasured", "emissions_source": "unmeasured"},
                country_iso=self.country_iso,
                grid_intensity_g_per_kwh=self.grid_intensity,
                pue=self.pue,
                started_at=self._started_at or finished_at,
                finished_at=finished_at,
            )
            return self.report

        duration_s = max(time.perf_counter() - self._start_wall, 1e-6)
        cpu_times = self._process.cpu_times()
        cpu_time_delta = max(float(cpu_times.user + cpu_times.system) - self._start_cpu_time, 0.0)
        logical_cores = max(psutil.cpu_count(logical=True) or 1, 1)
        avg_cpu_util = min(100.0, (cpu_time_delta / (duration_s * logical_cores)) * 100.0)

        rss_gb = self._process.memory_info().rss / (1024**3)
        # Industry rule of thumb: ~0.3725 W per GB of active DRAM
        ram_power_w = max(0.2, rss_gb * 0.3725)
        # Active CPU power scaled between 15% idle floor and 100% TDP
        cpu_power_w = self.cpu_tdp_watts * (0.15 + 0.85 * (avg_cpu_util / 100.0))

        gpu_power_w = 0.0
        gpu_attribution = "unavailable"
        if self._gpu_power_reader is not None:
            try:
                reading = float(self._gpu_power_reader())
                if math.isfinite(reading) and reading >= 0:
                    gpu_power_w = reading
                    gpu_attribution = "workload_telemetry"
            except Exception as exc:
                logger.warning("gpu_power_unavailable", error_type=type(exc).__name__)

        total_power_w = (cpu_power_w + ram_power_w + gpu_power_w) * self.pue
        energy_kwh = (total_power_w * (duration_s / 3600.0)) / 1000.0
        emissions_g = energy_kwh * self.grid_intensity
        emissions_kg = emissions_g / 1000.0
        backend = "omniforge_hardware_estimator"
        energy_source = emissions_source = "modeled_estimate"

        if self._cc_tracker is not None:
            try:
                cc_emissions_kg = self._cc_tracker.stop()
                cc_data = getattr(self._cc_tracker, "final_emissions_data", None)
                cc_energy = getattr(cc_data, "energy_consumed", None)
                if cc_emissions_kg is not None and math.isfinite(cc_emissions_kg) and cc_emissions_kg >= 0:
                    emissions_kg = float(cc_emissions_kg)
                    emissions_g = emissions_kg * 1000.0
                    backend = emissions_source = "codecarbon"
                if cc_energy is not None and math.isfinite(cc_energy) and cc_energy >= 0:
                    energy_kwh = float(cc_energy)
                    backend = energy_source = "codecarbon"
            except Exception as exc:
                logger.warning("codecarbon_stop_failed", error_type=type(exc).__name__)

        # Green score rewards lower emissions per run (exponentially decaying scale)
        green_score = round(100.0 * float(pow(2.718281828, -emissions_g / 50.0)), 2)

        self.report = CarbonReport(
            operation_name=self.operation_name,
            tracker_backend=backend,
            metadata={
                "energy_source": energy_source,
                "emissions_source": emissions_source,
                "gpu_attribution": gpu_attribution,
                "modeled_cpu_tdp_watts": self.cpu_tdp_watts,
                "modeled_ram_watts_per_gib": 0.3725,
            },
            duration_seconds=round(duration_s, 6),
            energy_kwh=round(energy_kwh, 9),
            emissions_kg_co2eq=round(emissions_kg, 9),
            emissions_g_co2eq=round(emissions_g, 6),
            country_iso=self.country_iso,
            grid_intensity_g_per_kwh=self.grid_intensity,
            pue=self.pue,
            avg_cpu_utilization_pct=round(avg_cpu_util, 2),
            cpu_power_watts=round(cpu_power_w, 3),
            ram_power_watts=round(ram_power_w, 3),
            gpu_power_watts=round(gpu_power_w, 3),
            green_score=green_score,
            started_at=self._started_at,
            finished_at=finished_at,
        )
        return self.report

    def __enter__(self) -> CarbonTracker:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


def track_carbon(
    operation_name: str | None = None,
    *,
    country_iso: str = "DEU",
) -> Callable[[F], F]:
    """Decorator to automatically measure and log carbon emissions of a function."""

    def decorator(func: F) -> F:
        op_name = operation_name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with CarbonTracker(op_name, country_iso=country_iso) as tracker:
                result = func(*args, **kwargs)
            if tracker.report is not None:
                logger.info(
                    "carbon_footprint_recorded",
                    operation=op_name,
                    emissions_g_co2eq=tracker.report.emissions_g_co2eq,
                    energy_kwh=tracker.report.energy_kwh,
                    green_score=tracker.report.green_score,
                )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
