"""Canonical unit system for SIT.

All latency values in the SIT pipeline use a single canonical unit (microseconds).
This module provides conversion utilities and assertions to enforce unit consistency.
No implicit conversions are permitted anywhere in the codebase.
"""

from enum import Enum
from typing import Union

import numpy as np


class LatencyUnit(str, Enum):
    """Supported latency units."""
    US = "us"   # microseconds (canonical)
    NS = "ns"   # nanoseconds
    MS = "ms"   # milliseconds


# The single canonical unit used throughout SIT
CANONICAL_LATENCY_UNIT: str = "us"

# Conversion factors TO microseconds
_TO_US = {
    LatencyUnit.US: 1.0,
    LatencyUnit.NS: 1e-3,
    LatencyUnit.MS: 1e3,
}


def convert_latency(
    x: Union[float, np.ndarray],
    from_unit: Union[str, LatencyUnit],
    to_unit: Union[str, LatencyUnit],
) -> Union[float, np.ndarray]:
    """Convert latency values between units.

    Supports scalar and numpy array inputs. Conversion is exact via
    intermediate canonical unit (microseconds).

    Args:
        x: Latency value(s) to convert.
        from_unit: Source unit ("us", "ns", or "ms").
        to_unit: Target unit ("us", "ns", or "ms").

    Returns:
        Converted value(s) in target unit.

    Raises:
        ValueError: If either unit is not recognized.
    """
    from_unit = LatencyUnit(from_unit)
    to_unit = LatencyUnit(to_unit)

    if from_unit == to_unit:
        return x

    # Convert to canonical (us), then to target
    factor = _TO_US[from_unit] / _TO_US[to_unit]
    return x * factor


def assert_latency_unit(unit: str) -> None:
    """Assert that a unit string is the canonical latency unit.

    Args:
        unit: Unit string to check.

    Raises:
        ValueError: If unit does not match CANONICAL_LATENCY_UNIT.
    """
    if unit != CANONICAL_LATENCY_UNIT:
        raise ValueError(
            f"Expected canonical latency unit '{CANONICAL_LATENCY_UNIT}', "
            f"got '{unit}'. All latency values must use '{CANONICAL_LATENCY_UNIT}'."
        )


def validate_latency_unit(unit: str) -> LatencyUnit:
    """Validate and return a LatencyUnit from a string.

    Args:
        unit: Unit string to validate.

    Returns:
        The corresponding LatencyUnit enum member.

    Raises:
        ValueError: If unit is not a valid LatencyUnit.
    """
    try:
        return LatencyUnit(unit)
    except ValueError:
        valid = [u.value for u in LatencyUnit]
        raise ValueError(
            f"Invalid latency unit '{unit}'. Valid units: {valid}"
        )
