"""Drift models for baseline latency evolution over time.

Supports linear, sinusoidal, step, regime-shift, and heteroscedastic
drift. Each drift function is parameterized and toggleable via config.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DriftParams:
    """Parameters for drift model.

    Attributes:
        drift_type: One of "none", "linear", "sinusoidal", "step", "regime_shift".
        a_us_per_step: Linear drift slope (us per time index).
        amplitude_us: Sinusoidal amplitude in us.
        period: Sinusoidal period in time indices.
        step_time: Time index at which step/regime shift occurs.
        step_magnitude_us: Step drift magnitude in us.
        hetero_scale: Scale factor for heteroscedastic noise growth.
    """
    drift_type: str = "none"
    a_us_per_step: float = 0.0
    amplitude_us: float = 0.0
    period: float = 20.0
    step_time: int = 50
    step_magnitude_us: float = 0.0
    hetero_scale: float = 0.0

    def __post_init__(self):
        valid_types = ("none", "linear", "sinusoidal", "step", "regime_shift")
        if self.drift_type not in valid_types:
            raise ValueError(
                f"Invalid drift_type '{self.drift_type}'. Must be one of {valid_types}"
            )


def mu_drift_us(t_index: int, params: DriftParams) -> float:
    """Compute baseline drift at a given time index.

    Returns the deterministic drift component in microseconds.
    Multiple drift types can be combined by chaining calls, but
    the primary API uses a single type per experiment for clarity.

    Args:
        t_index: Integer time index.
        params: Drift parameters.

    Returns:
        Drift value in microseconds.
    """
    if params.drift_type == "none":
        return 0.0

    elif params.drift_type == "linear":
        return params.a_us_per_step * t_index

    elif params.drift_type == "sinusoidal":
        if params.period <= 0:
            return 0.0
        return params.amplitude_us * np.sin(2.0 * np.pi * t_index / params.period)

    elif params.drift_type == "step":
        if t_index >= params.step_time:
            return params.step_magnitude_us
        return 0.0

    elif params.drift_type == "regime_shift":
        # Before shift: linear with half slope; after: step + reverse linear
        if t_index < params.step_time:
            return params.a_us_per_step * 0.5 * t_index
        else:
            return (
                params.step_magnitude_us
                + params.a_us_per_step * 0.5 * (t_index - params.step_time)
            )

    return 0.0


def drift_noise_std(t_index: int, params: DriftParams, base_std: float) -> float:
    """Compute heteroscedastic noise standard deviation.

    If hetero_scale > 0, noise variance grows with time index.

    Args:
        t_index: Integer time index.
        params: Drift parameters.
        base_std: Base standard deviation in us.

    Returns:
        Adjusted standard deviation in us.
    """
    if params.hetero_scale <= 0:
        return base_std
    # Variance grows linearly: std grows as sqrt(1 + scale * t)
    return base_std * np.sqrt(1.0 + params.hetero_scale * t_index)


def parse_drift_config(drift_cfg: Dict[str, Any]) -> DriftParams:
    """Parse drift parameters from a config dict."""
    return DriftParams(
        drift_type=drift_cfg.get("type", "none"),
        a_us_per_step=float(drift_cfg.get("a_us_per_step", 0.0)),
        amplitude_us=float(drift_cfg.get("amplitude_us", 0.0)),
        period=float(drift_cfg.get("period", 20.0)),
        step_time=int(drift_cfg.get("step_time", 50)),
        step_magnitude_us=float(drift_cfg.get("step_magnitude_us", 0.0)),
        hetero_scale=float(drift_cfg.get("hetero_scale", 0.0)),
    )
