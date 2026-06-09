"""
scripts/convert_smr_to_h5.py
=============================
Convert all .smr files in data/raw/ to .h5 format in data/processed/.

This is your original conversion script, wrapped to read paths from
config.yaml and process all files in the raw directory automatically.

Usage
-----
    python scripts/convert_smr_to_h5.py
    python scripts/convert_smr_to_h5.py --config path/to/config.yaml
    python scripts/convert_smr_to_h5.py --file data/raw/specific_slice.smr
"""

import sys
import os
import argparse
import warnings

import neo
import h5py
import numpy as np

# Add project root to path so src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Core conversion function (your original script, unchanged)
# ---------------------------------------------------------------------------

def convert_smr_to_h5(smr_path: str, h5_dir: str) -> str | None:
    """
    Convert a single Spike2 .smr file to a portable .h5 file.

    Parameters
    ----------
    smr_path : str
        Path to the source .smr file.
    h5_dir : str
        Directory to write the output .h5 file.

    Returns
    -------
    str or None
        Path to the written .h5 file, or None if conversion failed.
    """
    file_name = os.path.basename(smr_path)
    h5_name   = file_name.replace(".smr", ".h5")
    h5_path   = os.path.join(h5_dir, h5_name)

    print(f"Converting {file_name} → {h5_name}...")

    try:
        reader = neo.io.Spike2IO(filename=smr_path)
        block  = reader.read_block()

        with h5py.File(h5_path, "w") as f:
            if not block.segments:
                print(f"  Warning: No segments found in {file_name}")
                return None

            seg = block.segments[0]

            # Metadata group
            meta = f.create_group("metadata")
            meta.attrs["original_file"] = file_name

            if not seg.analogsignals:
                print(f"  Warning: No analog signals found in {file_name}")
                return None

            # We assume the first analogsignal contains the multi-channel data
            sig   = seg.analogsignals[0]
            fs    = float(sig.sampling_rate)
            units = str(sig.units.dimensionality)

            meta.attrs["fs"]               = fs
            meta.attrs["units"]            = units
            meta.attrs["duration_seconds"] = float(sig.duration)

            # Raw data: shape (n_samples, n_channels)
            data_shape = sig.shape
            print(f"  Writing dataset shape {data_shape} at {fs} Hz...")

            f.create_dataset(
                "raw_data",
                data=sig.magnitude,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )

            print(f"  ✓ Saved to {h5_path}")
            return h5_path

    except Exception as e:
        print(f"  ✗ Error converting {file_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def batch_convert(raw_dir: str, processed_dir: str) -> list[str]:
    """
    Convert all .smr files in raw_dir to .h5 files in processed_dir.

    Parameters
    ----------
    raw_dir : str
    processed_dir : str

    Returns
    -------
    list of str
        Paths of successfully converted files.
    """
    os.makedirs(processed_dir, exist_ok=True)

    smr_files = sorted([
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".smr")
    ])

    if not smr_files:
        print(f"No .smr files found in {raw_dir}")
        return []

    print(f"Found {len(smr_files)} .smr files to convert.\n")
    converted = []

    for smr_path in smr_files:
        result = convert_smr_to_h5(smr_path, processed_dir)
        if result:
            converted.append(result)

    print(f"\nDone: {len(converted)}/{len(smr_files)} files converted successfully.")
    return converted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert .smr files to .h5 format."
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (default: project root config.yaml)"
    )
    parser.add_argument(
        "--file", default=None,
        help="Convert a single .smr file instead of the full raw directory."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir       = cfg["paths"]["raw_data_dir"]
    processed_dir = cfg["paths"]["processed_dir"]

    if args.file:
        os.makedirs(processed_dir, exist_ok=True)
        convert_smr_to_h5(args.file, processed_dir)
    else:
        batch_convert(raw_dir, processed_dir)


if __name__ == "__main__":
    main()
