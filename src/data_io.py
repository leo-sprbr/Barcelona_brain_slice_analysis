"""
src/data_io.py
==============
Functions for reading and writing the HDF5 (.h5) data files produced
by the SMR → H5 conversion script.

This module knows *nothing* about analysis — it only handles I/O.
If you need to support a different file format in the future, this is
the only file you need to change.

Expected .h5 file layout (written by convert_smr_to_h5.py)
------------------------------------------------------------
    /metadata          (group)
        attrs:
            original_file    (str)
            fs               (float)   sampling rate in Hz
            units            (str)     physical units of the signal
            duration_seconds (float)
    /raw_data           (dataset, shape: [n_samples, n_channels])

Usage
-----
    from src.data_io import load_h5, list_h5_files, save_results

    data, meta = load_h5("data/processed/slice_01.h5")
    # data.shape → (n_samples, n_channels)
    # meta       → dict with fs, units, duration_seconds, original_file

    h5_files = list_h5_files("data/processed/")
"""

import os
import glob
import numpy as np
import h5py
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_h5(h5_path: str) -> tuple[np.ndarray, dict]:
    """
    Load a converted .h5 file and return the raw data array and metadata.

    Parameters
    ----------
    h5_path : str
        Path to the .h5 file.

    Returns
    -------
    data : np.ndarray, shape (n_samples, n_channels)
        Raw LFP data. Rows = time, columns = channels.
    meta : dict
        Metadata dict with keys:
            fs               (float) — sampling rate in Hz
            units            (str)   — physical unit string
            duration_seconds (float) — recording duration
            original_file    (str)   — original .smr filename
            n_channels       (int)   — number of channels
            n_samples        (int)   — number of samples
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # --- Raw data ---
        data = f["raw_data"][:]   # shape: (n_samples, n_channels)

        # --- Metadata ---
        meta_grp = f["metadata"]
        meta = {
            "fs":               float(meta_grp.attrs["fs"]),
            "units":            str(meta_grp.attrs["units"]),
            "duration_seconds": float(meta_grp.attrs["duration_seconds"]),
            "original_file":    str(meta_grp.attrs["original_file"]),
        }

    meta["n_samples"], meta["n_channels"] = data.shape

    return data, meta


def list_h5_files(directory: str) -> list[str]:
    """
    Return a sorted list of all .h5 file paths in a directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing .h5 files.

    Returns
    -------
    list of str
        Sorted list of absolute file paths.
    """
    pattern = os.path.join(directory, "*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[data_io] Warning: no .h5 files found in {directory}")
    return files


def get_slice_id(h5_path: str) -> str:
    """
    Derive a short slice identifier from the file name.

    Example: "data/processed/slice_01_baseline.h5" → "slice_01_baseline"

    Parameters
    ----------
    h5_path : str
        Path to the .h5 file.

    Returns
    -------
    str
        Filename without extension.
    """
    return os.path.splitext(os.path.basename(h5_path))[0]


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a results DataFrame to a CSV file.
    Creates the parent directory if it does not exist.

    Parameters
    ----------
    df : pd.DataFrame
        Results table to save.
    output_path : str
        Full path including filename, e.g. "results/up_down_metrics.csv"
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[data_io] Saved results → {output_path}")


def load_results(csv_path: str) -> pd.DataFrame:
    """
    Load a previously saved results CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Extension points
# ---------------------------------------------------------------------------
# To add support for a new data format (e.g. .nwb, .mat):
#   1. Add a new load_<format>() function here that returns
#      (data: np.ndarray, meta: dict) in the same format as load_h5().
#   2. Update Experiment.__init__() in src/core/experiment.py to
#      dispatch to the correct loader based on file extension.
#
# To load additional datasets stored inside the .h5 file
# (e.g. stimulus timing, electrode positions):
#   Add a dedicated load_<thing>() function that opens the same .h5
#   file and reads the extra group/dataset.
