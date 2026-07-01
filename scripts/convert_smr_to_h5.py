"""
scripts/fix_and_convert_smr.py
===============================
Converts .smr files that fail with:

    RuntimeWarning: overflow encountered in scalar multiply
    IndexError: index 7 is out of bounds for axis 0 with size 7

ROOT CAUSE
----------
In neo/rawio/spike2rawio.py, the block-walking loop computes the next
disk block offset as:

    ind = block_info["succ_block"] * self._diskblock

`succ_block` is stored as int32. On the last block, its value is a
sentinel that can overflow int32 when multiplied. NumPy >= 2.0 raises
a RuntimeWarning AND returns a garbage value, which is then used as a
memmap index → IndexError.

THE FIX
-------
We patch just the two offending lines in _parse_header in-memory:

    ind = int(chan_info["firstblock"]) * self._diskblock   # safe start
    ind = int(block_info["succ_block"]) * self._diskblock  # safe next

int() promotes the int32 scalar to a Python int before the multiply.
Python ints never overflow. We also add a bounds check so that a
sentinel that lands outside the file is simply ignored (the loop
counter already stops at the right number of blocks).

No file is ever modified. This is a pure runtime patch.

Usage
-----
    python scripts/fix_and_convert_smr.py --file data/raw/recording.smr
    python scripts/fix_and_convert_smr.py         # all files in raw_dir
"""

import sys
import os
import argparse
import warnings

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# The surgical patch — replaces only the two int32-unsafe lines
# ---------------------------------------------------------------------------

def apply_patch():
    """
    Monkey-patch Spike2RawIO._parse_header to cast succ_block to Python
    int before the multiply, preventing NumPy int32 overflow.

    Must be called before any Spike2RawIO instance is created.
    """
    import neo.rawio.spike2rawio as _mod

    _original_parse_header = _mod.Spike2RawIO._parse_header

    def _safe_parse_header(self):
        # Run the original method first — it populates everything EXCEPT
        # _all_data_blocks correctly (that's where the overflow happens).
        # We capture the error, then re-do only the block-walking loop safely.
        try:
            _original_parse_header(self)
            return   # no error → no patch needed for this file
        except (IndexError, OverflowError):
            pass     # fall through to the safe re-walk below

        # At this point the header info and channel_infos are populated
        # correctly (the error only happens in the block-walking section).
        # Re-run just that section with int64-safe arithmetic.
        from neo.rawio.spike2rawio import blockHeaderDesciption

        memmap_len = len(self._memmap)

        self._all_data_blocks    = {}
        self._by_seg_data_blocks = {}

        for chan_id, chan_info in enumerate(self._channel_infos):
            data_blocks = []

            # Cast to Python int immediately — no int32 overflow possible
            ind = int(chan_info["firstblock"]) * self._diskblock

            for b in range(chan_info["blocks"]):
                # Bounds guard: stop if the sentinel pushed us out of file
                if ind < 0 or ind + 20 > memmap_len:
                    break

                block_info = self._memmap[ind: ind + 20].view(blockHeaderDesciption)[0]
                data_blocks.append((
                    ind,
                    block_info["items"],
                    0,
                    block_info["start_time"],
                    block_info["end_time"],
                ))

                # THE FIX: int() before multiply → Python int → no overflow
                ind = int(block_info["succ_block"]) * self._diskblock

            data_blocks = np.array(
                data_blocks,
                dtype=[
                    ("pos",        "int64"),  # int64 for safety
                    ("size",       "int32"),
                    ("cumsum",     "int32"),
                    ("start_time", "int32"),
                    ("end_time",   "int32"),
                ],
            )
            if len(data_blocks):
                data_blocks["pos"] += 20   # skip 20-byte block header

            self._all_data_blocks[chan_id]    = data_blocks
            self._by_seg_data_blocks[chan_id] = []

        # Re-run everything after the block walk by calling the parent
        # continuation — BaseRawIO._parse_header calls _parse_header then
        # _post_parse_header (or equivalent). In this version of Neo the
        # remaining logic (gap detection, segment building, signal grouping)
        # is in the same method after the block walk. We need to trigger it.
        # The cleanest way: call _post_spike2_parse if it exists, otherwise
        # the data we've put in _all_data_blocks is used directly by
        # _get_analogsignal_chunk and friends at read time.
        #
        # Run the rest of the original method starting from the gap-detection
        # section by re-calling with a flag so we don't loop infinitely.
        self._patch_applied = True
        _original_parse_header(self)

    # Guard: skip re-entry on the second call triggered above
    _true_original = _original_parse_header

    def _guarded_parse_header(self):
        if getattr(self, "_patch_applied", False):
            # Second call — jump straight past the block walk to gap detection
            # by running a trimmed version that skips the walk entirely
            _safe_block_walk_only(self)
            return
        _safe_parse_header(self)

    def _safe_block_walk_only(self):
        """Re-entry path: only gap detection and segment building."""
        # The block data is already in self._all_data_blocks from the
        # safe walk. Call the original — it will redo the walk (safely
        # this time because _patch_applied bypasses the error path) but
        # that's cheap and correct since we've already set _all_data_blocks.
        # Actually the simplest safe re-entry is just to do nothing here
        # and let BaseRawIO use _all_data_blocks as-is.
        pass

    _mod.Spike2RawIO._parse_header = _guarded_parse_header
    print("  [patch] Spike2RawIO overflow fix applied.")


# ---------------------------------------------------------------------------
# Simpler, more reliable approach: patch just the two lines directly
# ---------------------------------------------------------------------------

def apply_patch_direct():
    """
    Direct patch: rewrite just the block-walking loop inside _parse_header
    using inspect + exec to replace the two unsafe lines in-place.

    This is more reliable than try/except re-entry because it doesn't
    depend on partial state from a failed call.
    """
    import inspect
    import textwrap
    import neo.rawio.spike2rawio as _mod

    src = inspect.getsource(_mod.Spike2RawIO._parse_header)

    # Remove the common indentation so exec() sees valid Python
    src = textwrap.dedent(src)

    # The two lines to fix. Root cause: succ_block is int32, sentinel value overflows
    # NumPy int32 arithmetic. Fix: cast to int64 and clamp to valid file range.
    old_ind_start = 'ind = chan_info["firstblock"] * self._diskblock'
    # Cast to int64 before multiply — int32 sentinel values overflow with NumPy 2.x
    new_ind_start = 'ind = int(np.int64(chan_info["firstblock"]) * np.int64(self._diskblock))'

    old_ind_next  = 'ind = block_info["succ_block"] * self._diskblock'
    # np.clip clamps sentinel to valid file range; loop counter ensures it's never used
    new_ind_next  = ('ind = int(np.clip('
                     'np.int64(block_info["succ_block"]) * np.int64(self._diskblock),'
                     ' 0, max(0, len(self._memmap) - 1)))')

    if old_ind_start not in src or old_ind_next not in src:
        print("  [patch] WARNING: could not find target lines in _parse_header source.")
        print("          Your Neo version may differ. Trying without patch.")
        return False

    patched_src = src.replace(old_ind_start, new_ind_start, 1)
    patched_src = patched_src.replace(old_ind_next,  new_ind_next,  1)

    # Compile and bind into the class
    ns = {}
    exec(compile(patched_src, "<patched_spike2rawio>", "exec"), _mod.__dict__, ns)
    _mod.Spike2RawIO._parse_header = ns["_parse_header"]

    print("  [patch] Spike2RawIO._parse_header patched successfully.")
    return True


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _best_signal(analogsignals):
    """
    From a list of Neo AnalogSignal objects, return the one most likely
    to be the multi-channel LFP (the one with the most channels).
    Prints a summary of all groups found so you can verify.
    """
    print(f"  Found {len(analogsignals)} analog signal group(s):")
    for i, sig in enumerate(analogsignals):
        print(f"    [{i}] shape={sig.shape}  "
              f"fs={float(sig.sampling_rate):.1f} Hz  "
              f"name='{sig.name}'  "
              f"units={sig.units.dimensionality}")
    return max(analogsignals,
               key=lambda s: s.shape[1] if s.ndim > 1 else 1)


def _load_via_sonpy(smr_path: str):
    """
    Direct fallback using sonpy, bypassing Neo's CedIO entirely.

    sonpy 1.9.x exposes SonFile at the top level (sonpy.SonFile),
    NOT at sonpy.lib.SonFile (which is what older Neo CedIO expects).
    This function handles both APIs.
    """
    import sonpy

    # Detect API version: 1.9.x uses sonpy.SonFile directly,
    # older versions used sonpy.lib.SonFile
    if hasattr(sonpy, 'SonFile'):
        SonFile   = sonpy.SonFile
        get_dtype = lambda ch_type: str(ch_type)
        adc_type  = 'DataType.Adc'
    elif hasattr(sonpy, 'lib'):
        SonFile   = sonpy.lib.SonFile
        get_dtype = lambda ch_type: ch_type
        adc_type  = sonpy.lib.DataType.Adc
    else:
        raise ImportError(
            f"Cannot find SonFile in sonpy. "
            f"Available: {[a for a in dir(sonpy) if not a.startswith('_')]}"
        )

    f      = SonFile(str(smr_path), True)   # True = read-only
    n_chan = f.MaxChannels()
    t_base = f.GetTimeBase()                 # seconds per clock tick

    print(f"  sonpy: {n_chan} channels, time base = {t_base:.2e} s/tick")

    # Find all ADC (waveform) channels
    waveform_channels = []
    for ch in range(n_chan):
        ch_type = f.ChannelType(ch)
        # Handle both API versions
        is_adc = (str(ch_type) == adc_type or
                  (hasattr(sonpy, 'lib') and ch_type == sonpy.lib.DataType.Adc))
        if is_adc:
            waveform_channels.append(ch)

    if not waveform_channels:
        # If no Adc type found, try reading all non-empty channels
        print("  No Adc channels found — trying all non-empty channels")
        for ch in range(n_chan):
            try:
                n_max = f.ChannelMaxTime(ch)
                if n_max > 0:
                    waveform_channels.append(ch)
            except Exception:
                pass

    if not waveform_channels:
        raise ValueError("No readable waveform channels found via sonpy.")

    print(f"  Reading {len(waveform_channels)} waveform channel(s): {waveform_channels}")

    arrays = []
    ch0_divide = f.ChannelDivide(waveform_channels[0])
    ch0_fs     = 1.0 / (t_base * ch0_divide)

    for ch in waveform_channels:
        divide = f.ChannelDivide(ch)
        n_max  = int(f.ChannelMaxTime(ch) // divide) + 1
        try:
            arr = f.ReadFloats(ch, n_max, 0)
            arrays.append(np.array(arr, dtype=np.float32))
        except Exception as e:
            print(f"    Channel {ch} failed ({e}), skipping")

    if not arrays:
        raise ValueError("Could not read any channel data via sonpy.")

    min_len = min(len(a) for a in arrays)
    data    = np.column_stack([a[:min_len] for a in arrays])
    dur     = float(min_len / ch0_fs)

    print(f"  sonpy read OK — shape {data.shape} @ {ch0_fs:.1f} Hz")
    return data, float(ch0_fs), "mV", dur


def convert_smr_to_h5(smr_path: str, h5_dir: str) -> str | None:
    """
    Convert one .smr file to .h5.

    Tries three backends in order:
      1. Patched Spike2IO  (old SON1 .smr files)
      2. Neo CedIO         (newer SON2 files, requires sonpy)
      3. sonpy directly    (bypasses Neo CedIO, handles sonpy API changes)
    """
    import neo

    file_name = os.path.basename(smr_path)
    h5_name   = file_name.rsplit(".", 1)[0] + ".h5"
    h5_path   = os.path.join(h5_dir, h5_name)

    print(f"\nConverting {file_name} → {h5_name} ...")

    data, fs, units, dur = None, None, None, None

    # --- Attempt 1: patched Spike2IO ---
    try:
        reader = neo.io.Spike2IO(filename=smr_path)
        block  = reader.read_block()
        sigs   = block.segments[0].analogsignals
        if not sigs:
            raise ValueError("No analog signals found.")
        sig   = _best_signal(sigs)
        data  = sig.magnitude
        fs    = float(sig.sampling_rate)
        units = str(sig.units.dimensionality)
        dur   = float(sig.duration)
        print(f"  ✓ Spike2IO read OK — using shape {data.shape} @ {fs} Hz")

    except Exception as e1:
        print(f"  ✗ Spike2IO failed: {e1}")

        # --- Attempt 2: Neo CedIO ---
        try:
            reader = neo.io.CedIO(filename=smr_path)
            block  = reader.read_block()
            sigs   = block.segments[0].analogsignals
            if not sigs:
                raise ValueError("No analog signals found.")
            sig   = _best_signal(sigs)
            data  = sig.magnitude
            fs    = float(sig.sampling_rate)
            units = str(sig.units.dimensionality)
            dur   = float(sig.duration)
            print(f"  ✓ CedIO read OK — using shape {data.shape} @ {fs} Hz")

        except Exception as e2:
            print(f"  ✗ CedIO failed: {e2}")

            # --- Attempt 3: sonpy directly (bypasses Neo's CedIO) ---
            try:
                data, fs, units, dur = _load_via_sonpy(smr_path)
                print(f"  ✓ sonpy direct read OK")
            except Exception as e3:
                print(f"  ✗ sonpy direct failed: {e3}")
                print(f"  ✗ All backends failed for {file_name}.")
                return None

    # --- Write .h5 ---
    os.makedirs(h5_dir, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["original_file"]    = file_name
        meta.attrs["fs"]               = fs
        meta.attrs["units"]            = units
        meta.attrs["duration_seconds"] = dur

        print(f"  Writing {data.shape} ...")
        f.create_dataset(
            "raw_data",
            data=data,
            compression="gzip",
            compression_opts=4,
            chunks=True,
        )

    print(f"  ✓ Saved → {h5_path}")
    return h5_path



def batch_convert(raw_dir: str, processed_dir: str) -> list[str]:
    smr_files = sorted([
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith((".smr", ".smrx"))
    ])
    if not smr_files:
        print(f"No .smr/.smrx files found in {raw_dir}")
        return []

    print(f"Found {len(smr_files)} file(s).")
    converted = []
    for path in smr_files:
        result = convert_smr_to_h5(path, processed_dir)
        if result:
            converted.append(result)

    print(f"\nDone: {len(converted)}/{len(smr_files)} converted.")
    return converted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert .smr files that fail with the Neo int32 overflow bug."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--file",   default=None,
                        help="Convert a single .smr file.")
    args = parser.parse_args()

    # Apply the source-level patch BEFORE importing neo.io.Spike2IO
    patch_ok = apply_patch_direct()
    if not patch_ok:
        print("  Proceeding without patch (may still work if Neo was updated).")

    cfg           = load_config(args.config)
    raw_dir       = cfg["paths"]["raw_data_dir"]
    processed_dir = cfg["paths"]["processed_dir"]

    if args.file:
        os.makedirs(processed_dir, exist_ok=True)
        convert_smr_to_h5(args.file, processed_dir)
    else:
        batch_convert(raw_dir, processed_dir)


if __name__ == "__main__":
    main()