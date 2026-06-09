"""
scripts/define_trials.py
========================
Interactive tool for defining trial windows within a raw .h5 recording.

WHY THIS EXISTS
---------------
A single recording file may contain multiple stimulation trials, each with
baseline / stimulation / post-stimulation periods, plus large noise artifacts
(electrode settling, mechanical disturbances) at arbitrary time points.

This script lets you:
  1. Visually inspect the full raw LFP.
  2. Define one or more trial windows (start / end time in seconds).
  3. Assign conditions (baseline / stimulation / post_stimulation) to
     sub-windows within each trial OR treat each trial as a single condition.
  4. Reject transient noise artifacts within each window (z-score threshold).
  5. Save each clean trial window as its own .h5 file, ready for the
     analysis pipeline.

TWO USAGE MODES
---------------
A) Interactive matplotlib GUI (recommended when working locally):
       python scripts/define_trials.py --slice data/processed/recording.h5 --interactive

B) Non-interactive / scripted (reproducible, good for re-running):
       python scripts/define_trials.py --slice data/processed/recording.h5 --trials trials.json

   where trials.json looks like:
   [
     {"label": "trial_01", "start_s": 300, "end_s": 900,
      "conditions": [
        {"name": "baseline",        "start_s": 300, "end_s": 500},
        {"name": "stimulation",     "start_s": 500, "end_s": 700},
        {"name": "post_stimulation","start_s": 700, "end_s": 900}
      ]},
     ...
   ]

OUTPUT
------
For each trial × condition window, one .h5 file is written to data/processed/:
    <original_stem>_<trial_label>_<condition>.h5

These files have the same layout as the original (raw_data, metadata) plus
two extra metadata attributes:
    trial_window_start_s   — start of this window in the original recording
    trial_window_end_s     — end of this window
    artifact_fraction      — fraction of samples replaced by interpolation

The rest of the pipeline (Experiment, batch_process_slices.py, etc.) reads
these files identically to any other .h5 file — no changes needed there.
"""

import sys
import os
import json
import argparse
import numpy as np
import h5py

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data_io import load_h5, get_slice_id
from src.trial_io import (
    detect_artifacts,
    interpolate_artifacts,
    save_trial_h5,
    TrialDefinition,
    ConditionWindow,
)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(h5_path: str, cfg: dict) -> list[TrialDefinition]:
    """
    Launch an interactive matplotlib figure for defining trial windows.

    Controls
    --------
    - Click and drag on the LFP trace to define a trial window.
    - Press ENTER in the terminal to confirm each window and assign conditions.
    - Press 'q' or close the figure when done.

    Parameters
    ----------
    h5_path : str
    cfg : dict

    Returns
    -------
    list of TrialDefinition
    """
    import matplotlib
    matplotlib.use("TkAgg" if _tk_available() else "Qt5Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    data, meta = load_h5(h5_path)
    fs  = meta["fs"]
    lfp = data.mean(axis=1)            # channel average for display
    t   = np.arange(len(lfp)) / fs
    conditions = cfg["conditions"]

    print("\n" + "=" * 60)
    print("  TRIAL DEFINITION — Interactive Mode")
    print("=" * 60)
    print(f"  Recording : {os.path.basename(h5_path)}")
    print(f"  Duration  : {meta['duration_seconds']:.1f} s")
    print(f"  Conditions: {conditions}")
    print()
    print("  HOW TO USE:")
    print("  1. Click and drag on the plot to select a trial window.")
    print("  2. Release the mouse — the window will be highlighted.")
    print("  3. Switch to the terminal and follow the prompts.")
    print("  4. Repeat for each trial. Close the figure when done.")
    print("=" * 60 + "\n")

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, lfp, color="steelblue", lw=0.4, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP (a.u.)")
    ax.set_title("Click and drag to select a trial window. Close when done.")

    trials        = []
    current_span  = {"t_start": None, "t_end": None}
    highlight     = [None]   # mutable container for the span patch

    def _on_select(t_start, t_end):
        # Remove previous highlight
        if highlight[0] is not None:
            highlight[0].remove()

        highlight[0] = ax.axvspan(t_start, t_end, alpha=0.25,
                                  color="orange", label="Selected window")
        ax.set_title(f"Selected: {t_start:.1f} – {t_end:.1f} s  "
                     f"(duration {t_end - t_start:.1f} s)\n"
                     f"Switch to terminal to confirm.")
        fig.canvas.draw_idle()

        current_span["t_start"] = t_start
        current_span["t_end"]   = t_end

        # Prompt the user in a non-blocking way via input thread
        import threading
        threading.Thread(
            target=_prompt_user,
            args=(t_start, t_end, trials, conditions, ax, fig, highlight),
            daemon=True,
        ).start()

    span_selector = SpanSelector(
        ax, _on_select, direction="horizontal",
        useblit=True,
        props=dict(alpha=0.3, facecolor="orange"),
        interactive=True,
        drag_from_anywhere=True,
    )

    plt.tight_layout()
    plt.show()   # blocks until the figure is closed

    return trials


def _prompt_user(t_start, t_end, trials, conditions, ax, fig, highlight):
    """Run in a background thread: prompt the user to confirm or skip a window."""
    import matplotlib.pyplot as plt

    print(f"\n  Selected window: {t_start:.1f} – {t_end:.1f} s")
    label = input("  Trial label (e.g. trial_01) or ENTER to skip: ").strip()
    if not label:
        if highlight[0] is not None:
            highlight[0].set_alpha(0.05)
            fig.canvas.draw_idle()
        return

    # Ask for condition sub-windows
    print(f"\n  Conditions available: {conditions}")
    print("  Define a time window for each condition within this trial.")
    print("  Press ENTER to skip a condition.")
    cond_windows = []

    for cond in conditions:
        raw = input(f"    {cond} — start end (seconds, space-separated): ").strip()
        if not raw:
            continue
        try:
            parts = raw.split()
            cs, ce = float(parts[0]), float(parts[1])
            if cs < t_start or ce > t_end:
                print(f"    Warning: {cs}–{ce} is outside the trial window. Clipping.")
                cs, ce = max(cs, t_start), min(ce, t_end)
            cond_windows.append(ConditionWindow(name=cond, start_s=cs, end_s=ce))
        except (ValueError, IndexError):
            print(f"    Could not parse '{raw}', skipping {cond}.")

    trial = TrialDefinition(
        label=label, start_s=t_start, end_s=t_end,
        conditions=cond_windows
    )
    trials.append(trial)

    # Mark this trial in the figure
    color = plt.cm.tab10(len(trials) / 10)
    ax.axvspan(t_start, t_end, alpha=0.15, color=color)
    ax.text((t_start + t_end) / 2, ax.get_ylim()[1] * 0.9, label,
            ha="center", fontsize=8, color=color)
    if highlight[0] is not None:
        highlight[0].set_alpha(0.0)
    fig.canvas.draw_idle()

    print(f"  ✓ Trial '{label}' saved ({len(cond_windows)} conditions).\n")
    print("  Select another window or close the figure when done.\n")


def _tk_available() -> bool:
    try:
        import tkinter
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Non-interactive (JSON) mode
# ---------------------------------------------------------------------------

def run_from_json(h5_path: str, json_path: str, cfg: dict) -> list[TrialDefinition]:
    """
    Load trial definitions from a JSON file.

    Parameters
    ----------
    h5_path : str
    json_path : str
    cfg : dict

    Returns
    -------
    list of TrialDefinition
    """
    with open(json_path) as f:
        raw = json.load(f)

    trials = []
    for entry in raw:
        cond_windows = [
            ConditionWindow(
                name=c["name"],
                start_s=float(c["start_s"]),
                end_s=float(c["end_s"]),
            )
            for c in entry.get("conditions", [])
        ]
        trials.append(TrialDefinition(
            label=entry["label"],
            start_s=float(entry["start_s"]),
            end_s=float(entry["end_s"]),
            conditions=cond_windows,
        ))

    print(f"Loaded {len(trials)} trial(s) from {json_path}")
    return trials


# ---------------------------------------------------------------------------
# Processing & saving
# ---------------------------------------------------------------------------

def process_and_save(
    h5_path: str,
    trials: list[TrialDefinition],
    cfg: dict,
    output_dir: str = None,
) -> list[str]:
    """
    For each trial × condition window:
      1. Slice the raw data.
      2. Save a new .h5 file.

    Parameters
    ----------
    h5_path : str
    trials : list of TrialDefinition
    cfg : dict
    output_dir : str, optional
        Where to save the output files (default: cfg["paths"]["processed_dir"]).

    Returns
    -------
    list of str
        Paths of all written .h5 files.
    """
    data, meta = load_h5(h5_path)
    fs         = meta["fs"]
    stem       = get_slice_id(h5_path)
    out_dir    = output_dir or cfg["paths"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    written    = []

    for trial in trials:
        print(f"\nProcessing trial '{trial.label}' "
              f"({trial.start_s:.1f}–{trial.end_s:.1f} s)...")

        # Determine which condition windows to process
        windows_to_process = trial.conditions if trial.conditions else [
            ConditionWindow(name="recording",
                            start_s=trial.start_s, end_s=trial.end_s)
        ]

        for cw in windows_to_process:
            lo  = int(round(cw.start_s * fs))
            hi  = int(round(cw.end_s   * fs))
            lo  = max(0, lo)
            hi  = min(data.shape[0], hi)

            window_data = data[lo:hi, :].copy()   # (n_samples, n_channels)

            # --- Save ---
            out_name = f"{stem}_{trial.label}_{cw.name}.h5"
            out_path = os.path.join(out_dir, out_name)

            extra_meta = {
                "trial_label":           trial.label,
                "trial_window_start_s":  cw.start_s,
                "trial_window_end_s":    cw.end_s,
                "artifact_fraction":     0.0,
                "source_file":           os.path.basename(h5_path),
            }

            save_trial_h5(
                path=out_path,
                data=window_data,
                original_meta=meta,
                extra_meta=extra_meta,
            )
            written.append(out_path)
            print(f"  ✓ Saved → {out_name}")

    # Also save the trial definitions as JSON alongside the data
    json_out = os.path.join(out_dir, f"{stem}_trial_definitions.json")
    _save_trial_json(trials, json_out)
    print(f"\nTrial definitions saved → {os.path.basename(json_out)}")
    print(f"\nDone. {len(written)} file(s) written to {out_dir}/")
    return written


def _save_trial_json(trials: list[TrialDefinition], path: str) -> None:
    """Serialize trial definitions to JSON for reproducibility."""
    data = [
        {
            "label":   t.label,
            "start_s": t.start_s,
            "end_s":   t.end_s,
            "conditions": [
                {"name": c.name, "start_s": c.start_s, "end_s": c.end_s}
                for c in t.conditions
            ],
        }
        for t in trials
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Define trial windows and apply artifact rejection."
    )
    parser.add_argument("--slice", required=True,
                        help="Path to the .h5 recording file.")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive matplotlib window picker.")
    parser.add_argument("--trials",
                        help="Path to a JSON file with pre-defined trial windows.")
    parser.add_argument("--output-dir",
                        help="Where to write output .h5 files "
                             "(default: processed_dir from config.yaml).")
    parser.add_argument("--config",
                        help="Path to config.yaml (default: project root).")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.interactive:
        trials = run_interactive(args.slice, cfg)
    elif args.trials:
        trials = run_from_json(args.slice, args.trials, cfg)
    else:
        parser.error("Provide --interactive or --trials <json_file>.")
        return

    if not trials:
        print("No trials defined. Nothing to save.")
        return

    process_and_save(args.slice, trials, cfg, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
