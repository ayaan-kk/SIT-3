"""Data I/O for SIT datasets.

Handles reading and writing of trial, decision, and artifact DataFrames
in both Parquet and CSV formats. Parquet is preferred for reproducibility
(binary-exact round-trip).
"""

import os
from pathlib import Path

import pandas as pd

from sit.core.logging import get_logger

logger = get_logger("data.io")


def write_dataframe(df: pd.DataFrame, path: str, fmt: str = "parquet") -> str:
    """Write a DataFrame to disk in the specified format.

    Args:
        df: DataFrame to write.
        path: Output file path (extension will be enforced).
        fmt: Format string, either 'parquet' or 'csv'.

    Returns:
        The actual path written to.

    Raises:
        ValueError: If format is not supported.
    """
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"Unsupported format '{fmt}'. Use 'parquet' or 'csv'.")

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(path, index=False, engine="pyarrow")
    else:
        df.to_csv(path, index=False)

    logger.info("Wrote %d rows to %s", len(df), path)
    return path


def read_dataframe(path: str) -> pd.DataFrame:
    """Read a DataFrame from disk, inferring format from extension.

    Args:
        path: Path to the data file (.parquet or .csv).

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If format cannot be determined.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = file_path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Cannot determine format from extension '{ext}'")

    logger.info("Read %d rows from %s", len(df), path)
    return df
