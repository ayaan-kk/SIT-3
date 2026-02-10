"""Configuration loading, validation, and normalization.

Loads YAML configs, applies defaults, validates types, and produces
a canonical dict suitable for deterministic hashing.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from sit.core.hashing import hash_config
from sit.core.units import CANONICAL_LATENCY_UNIT, assert_latency_unit


# Default configuration values
DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "strict_mode": True,
    "latency_unit": CANONICAL_LATENCY_UNIT,
    "slo_us": 500000,
    "n_trials": 10,
    "output_dir": "data",
    "export_format": "parquet",
    "schedulers": ["random"],
    "regimes": {"mode": "synthetic", "n_regimes": 2},
    "logging": {"level": "INFO"},
}

# Required top-level keys and their expected types
_REQUIRED_KEYS = {
    "seed": int,
    "strict_mode": bool,
    "latency_unit": str,
    "slo_us": (int, float),
    "n_trials": int,
    "output_dir": str,
    "export_format": str,
    "schedulers": list,
    "regimes": dict,
    "logging": dict,
}


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config file, apply defaults, validate, and normalize.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional dict of key-value overrides applied after loading.

    Returns:
        Normalized, validated configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config validation fails.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

    config = _apply_defaults(raw)

    if overrides:
        config.update(overrides)

    _validate_config(config)
    config = _normalize_config(config)

    return config


def _apply_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Merge raw config with defaults (raw takes precedence)."""
    merged = {}
    for key, default_val in DEFAULTS.items():
        if key in raw:
            merged[key] = raw[key]
        else:
            merged[key] = default_val
    # Preserve any extra keys from the raw config
    for key in raw:
        if key not in merged:
            merged[key] = raw[key]
    return merged


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate config types and constraints.

    Raises:
        ValueError: On any validation failure.
    """
    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")
        val = config[key]
        if not isinstance(val, expected_type):
            raise ValueError(
                f"Config key '{key}' expected type {expected_type}, "
                f"got {type(val).__name__}: {val!r}"
            )

    # Latency unit must match canonical
    assert_latency_unit(config["latency_unit"])

    # Seed must be non-negative
    if config["seed"] < 0:
        raise ValueError(f"Seed must be non-negative, got {config['seed']}")

    # n_trials must be positive
    if config["n_trials"] < 1:
        raise ValueError(f"n_trials must be >= 1, got {config['n_trials']}")

    # slo_us must be positive
    if config["slo_us"] <= 0:
        raise ValueError(f"slo_us must be positive, got {config['slo_us']}")

    # export_format must be supported
    if config["export_format"] not in ("parquet", "csv"):
        raise ValueError(
            f"export_format must be 'parquet' or 'csv', got '{config['export_format']}'"
        )


def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config for stable hashing.

    Ensures numeric types are consistent and slo_us is float.
    """
    config["seed"] = int(config["seed"])
    config["slo_us"] = float(config["slo_us"])
    config["n_trials"] = int(config["n_trials"])
    return config


def config_hash(config: Dict[str, Any]) -> str:
    """Compute the canonical hash of a configuration dict.

    This is a convenience wrapper around hashing.hash_config.
    """
    return hash_config(config)
