"""
src/visualization.py
====================
All plotting functions for the pipeline. Each function takes data
(arrays, DataFrames, dicts) and an optional Axes object, and returns
the Figure and Axes it drew on.

Design rules
------------
- Every function is self-contained: it accepts data directly, not an
  Experiment object, so you can call any plot from a notebook without
  needing to run the full pipeline.
- Functions that accept an `ax` parameter will draw on the provided
  Axes (useful for multi-panel figures). If ax=None, a new Figure is
  created and returned.
- No global state: every function accepts a `cfg` dict for styling
  parameters (colors, DPI, format).

Usage
-----
    from src.visualization import (
        plot_raw_lfp,
        plot_pc1_distribution,
        plot_states_overlay,
        plot_aligned_firing_rates,
        plot_duration_boxplot,
        plot_so_frequency,
        plot_delta_duration,
        save_figure,
    )
"""

import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_style(cfg: dict) -> None:
    """Apply matplotlib style from config."""
    style = cfg.get("figures", {}).get("style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("default")


def _get_colors(conditions: list, cfg: dict) -> dict:
    """Return a {condition: color} dict using the configured colormap."""
    import matplotlib.cm as cm
    cmap_name = cfg.get("figures", {}).get("colormap", "tab10")
    cmap = cm.get_cmap(cmap_name)
    return {cond: cmap(i / max(len(conditions) - 1, 1))
            for i, cond in enumerate(conditions)}


def save_figure(fig: plt.Figure, filename: str, cfg: dict) -> str:
    """
    Save a figure to the configured figures directory.

    Directory layout
    ----------------
        <cfg.paths.figures_dir>/<t_start>-<t_end>/<filename>_t<t_start>-<t_end>.<fmt>

    cfg["paths"]["figures_dir"] is the BASE — the test script is expected
    to have set this to include the recording name, e.g.
        results/figures/20260414_SLICE3_CONTROL_Nathalia

    The <t_start>-<t_end> time-window subfolder is added here automatically
    so each analysis window gets its own folder.
    """
    # Base figures folder — respect whatever the caller has set,
    # so nested recording folders (set in the test script) work.
    base_dir = cfg["paths"]["figures_dir"]

    # Time-window subfolder
    t_start = cfg["visualization"]["t_start"]
    t_end   = cfg["visualization"]["t_end"]
    fig_dir = os.path.join(base_dir, f"{t_start}-{t_end}")
    os.makedirs(fig_dir, exist_ok=True)

    # Append the time window to the filename too, so identical filenames
    # from different windows are still distinguishable if flattened later
    fmt  = cfg.get("figures", {}).get("format", "png")
    dpi  = cfg.get("figures", {}).get("dpi", 300)
    path = os.path.join(fig_dir, f"{filename}_t{t_start}-{t_end}.{fmt}")

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure → {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Raw LFP overview
# ---------------------------------------------------------------------------

def plot_raw_lfp(
    lfp: np.ndarray,
    fs: float,
    cfg: dict,
    t_range_s: tuple = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the raw (or channel-averaged) LFP signal.

    Parameters
    ----------
    lfp : np.ndarray, shape (n_samples,)
    fs : float
        Original sampling rate (Hz).
    cfg : dict
    t_range_s : (start_s, end_s) or None
        Zoom into a time window. None = plot the full recording.
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)
    t = np.arange(len(lfp)) / fs

    if t_range_s is not None:
        mask = (t >= t_range_s[0]) & (t <= t_range_s[1])
        t, lfp = t[mask], lfp[mask]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))
    else:
        fig = ax.get_figure()

    ax.plot(t, lfp, color="steelblue", lw=0.5, alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP (a.u.)")
    ax.set_title("Raw LFP")

    return fig, ax

def plot_raw_lfp_time_window(
    lfp: np.ndarray,
    fs: float,
    cfg: dict,
    t_range_s: tuple = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the raw LFP signal.

    Parameters
    ----------
    lfp : np.ndarray, shape (n_samples, n_channels)
    fs : float
        Original sampling rate (Hz).
    cfg : dict
    t_range_s : (start_s, end_s) or None
        Zoom into a time window. None = plot the full recording.
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)

    n_samples = lfp.shape[0]
    t = np.arange(n_samples) / fs

    if t_range_s is not None:
        mask = (t >= t_range_s[0]) & (t <= t_range_s[1])
        t = t[mask]
        lfp = lfp[mask]
    
    # 3. Determine if we are plotting single or multiple channels
    is_multichannel = lfp.ndim > 1 and lfp.shape[1] > 1
    n_channels = lfp.shape[1] if is_multichannel else 1

    # 4. Create the Figure and Subplots
    if is_multichannel:
        # Create a vertically stacked layout sharing the same exact X-axis
        # Adjust height dynamically based on the number of tracks (2 inches per channel)
        fig, axes = plt.subplots(
            nrows=n_channels, 
            ncols=1, 
            figsize=(14, 2 * n_channels), 
            sharex=True
        )
        # Ensure 'axes' is always iterable even if a 1-row slice somehow occurs
        if n_channels == 1:
            axes = [axes]
    else:
        # Fallback for single channel (uses provided ax, or generates a clean single plot)
        if ax is None:
            fig, ax_single = plt.subplots(figsize=(14, 3))
        else:
            fig = ax.get_figure()
            ax_single = ax
        axes = [ax_single]

    t_start = cfg['visualization']['t_start']
    t_end = cfg['visualization']['t_end']

    # 5. Plot the data onto the axes
    if is_multichannel:
        # Map out which channels were selected in the config to label them correctly
        channel_list = cfg.get("preprocessing", {}).get("channel", list(range(n_channels)))
        if channel_list is None:  # In case it was null/None in yaml
            channel_list = list(range(n_channels))

        for i in range(n_channels):
            axes[i].plot(t, lfp[:, i], color="steelblue", lw=0.5, alpha=0.9)
            axes[i].set_ylabel(f"Ch {channel_list[i]+1}\n(a.u.)")
            axes[i].grid(True, alpha=0.2)
            
        axes[0].set_title(f"Raw Multichannel LFP Rows")
        axes[-1].set_xlabel("Time (s)")
        
    else:
        # Standard Single-Channel workflow
        axes[0].plot(t, lfp, color="steelblue", lw=0.5, alpha=0.9)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("LFP (a.u.)")
        
        channel = cfg.get("preprocessing", {}).get("channel", "unknown")
        axes[0].set_title(f"Raw LFP (Channel {channel})")
    
    # 6. Rewrite the X-axis tick labels manually
    # Find the current tick positions (which are between 0 and 60)
    current_ticks = axes[-1].get_xticks()
    
    # Calculate what the real time text should say for each position
    new_labels = [f"{int(tick + t_start)}" for tick in current_ticks]
    
    # Apply the new text labels to the bottom axis
    axes[-1].set_xticks(current_ticks)
    axes[-1].set_xticklabels(new_labels)

    plt.tight_layout()

    # Save figure
    save_figure(fig, "raw_lfp_all", cfg)
    print("The plot has been saved to:", os.path.join(cfg["paths"]["figures_dir"], "raw_lfp_all.png"))

    return fig, axes

# ---------------------------------------------------------------------------
# 2. Feature signals (SO, gamma, logMUA)
# ---------------------------------------------------------------------------

def plot_features(
    features: np.ndarray,
    t_ds: np.ndarray,
    cfg: dict,
    t_range_s: tuple = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot the three z-scored feature signals (SO, gamma env, logMUA)
    as stacked subplots.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, 3)
    t_ds : np.ndarray
    cfg : dict
    t_range_s : (start_s, end_s) or None

    Returns
    -------
    fig, axes (array of 3 Axes)
    """
    _apply_style(cfg)
    labels = ["SO deflection (z)", "Gamma envelope (z)", "logMUA (z)"]
    colors = ["royalblue", "darkorange", "forestgreen"]

    if t_range_s is not None:
        mask     = (t_ds >= t_range_s[0]) & (t_ds <= t_range_s[1])
        t_ds     = t_ds[mask]
        features = features[mask]

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t_ds, features[:, i], color=color, lw=0.8)
        ax.set_ylabel(label, fontsize=9)
        ax.axhline(0, color="gray", lw=0.5, ls="--")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Z-scored feature signals", y=1.01)
    fig.tight_layout()

    return fig, axes

def plot_multichannel_feature_group(
    feature_matrix: np.ndarray,
    t_ds: np.ndarray,
    feature_name: str,
    cfg: dict,
    channel_order: list[int] = None,
) -> plt.Figure:
    """
    Plots all channels for a single feature type in stacked subplots.

    Parameters
    ----------
    feature_matrix : shape (n_downsampled, n_channels)
    channel_order : list of int, optional
        1-based channel numbers giving the display order. If None, defaults
        to natural order 1..N. Use src.utils.get_electrode_display_order(N)
        to get the physical electrode ordering.
    """
    from src.utils import get_electrode_display_order

    n_channels = feature_matrix.shape[1]
    t_start    = cfg['visualization']['t_start']

    # Resolve display order
    if channel_order is None:
        channel_order = list(range(1, n_channels + 1))   # natural order

    fig, axes = plt.subplots(
        nrows=n_channels, ncols=1,
        figsize=(14, 2 * n_channels), sharex=True,
    )
    if n_channels == 1:
        axes = [axes]

    # Plot channels in the requested order
    for row_idx, ch_1based in enumerate(channel_order):
        col_idx = ch_1based - 1   # convert to 0-based column index
        axes[row_idx].plot(
            t_ds, feature_matrix[:, col_idx],
            color="crimson" if "logMUA" in feature_name else "steelblue",
            lw=0.6,
        )
        axes[row_idx].set_ylabel(f"Ch {ch_1based}\n(z)", fontsize=8)
        axes[row_idx].axhline(0, color="gray", lw=0.5, ls="--")
        axes[row_idx].grid(True, alpha=0.15)

    axes[0].set_title(f"All Channels — {feature_name}", fontsize=14, fontweight='bold')
    axes[-1].set_xlabel("Time (s)")

    current_ticks = axes[-1].get_xticks()
    new_labels = [f"{int(tick + t_start)}" for tick in current_ticks]
    axes[-1].set_xticks(current_ticks)
    axes[-1].set_xticklabels(new_labels)

    plt.tight_layout()
    save_filename = f"multichannel_{feature_name.lower().replace(' ', '_')}"
    save_figure(fig, save_filename, cfg)
    print("The plot has been saved to:", os.path.join(cfg["paths"]["figures_dir"], save_filename))

    return fig
# ---------------------------------------------------------------------------
# 3. PC1 distribution (bimodal check)
# ---------------------------------------------------------------------------

def plot_pc1_distribution(
    pc1: np.ndarray,
    threshold: float,
    cfg: dict,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the distribution of PC1 values with the detection threshold.

    This is a key diagnostic: the distribution should be bimodal, with
    the threshold sitting in the valley between the two modes.

    Parameters
    ----------
    pc1 : np.ndarray
    threshold : float
    cfg : dict
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.hist(pc1, bins=100, density=True, alpha=0.5, color="slategray",
            label="PC1 distribution")

    # KDE overlay
    x = np.linspace(pc1.min(), pc1.max(), 500)
    kde = gaussian_kde(pc1, bw_method="scott")
    ax.plot(x, kde(x), color="black", lw=1.5, label="KDE")

    # Threshold line
    ax.axvline(threshold, color="crimson", lw=2, ls="--",
               label=f"Threshold = {threshold:.3f}")

    # Shade Up/Down regions
    ax.axvspan(threshold, pc1.max(), alpha=0.08, color="firebrick", label="Up state")
    ax.axvspan(pc1.min(), threshold, alpha=0.08, color="navy",      label="Down state")

    ax.set_xlabel("PC1 score")
    ax.set_ylabel("Density")
    ax.set_title("PC1 bimodal distribution")
    ax.legend(fontsize=8)

    return fig, ax


# ---------------------------------------------------------------------------
# 4. States overlaid on LFP / PC1
# ---------------------------------------------------------------------------

def plot_states_overlay(
    t_ds: np.ndarray,
    pc1: np.ndarray,
    states: np.ndarray,
    threshold: float,
    cfg: dict,
    t_range_s: tuple = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot PC1 over time with Up/Down states indicated by background shading.

    Parameters
    ----------
    t_ds : np.ndarray
    pc1 : np.ndarray
    states : np.ndarray — binary (1=Up, 0=Down)
    threshold : float
    cfg : dict
    t_range_s : (start_s, end_s) or None
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)

    if t_range_s is not None:
        mask   = (t_ds >= t_range_s[0]) & (t_ds <= t_range_s[1])
        t_ds   = t_ds[mask]
        pc1    = pc1[mask]
        states = states[mask]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))
    else:
        fig = ax.get_figure()

    ax.plot(t_ds, pc1, color="gray", lw=0.7, alpha=0.8, label="PC1")
    ax.axhline(threshold, color="crimson", lw=1, ls="--", label="Threshold")

    # Shade Up-state regions
    _shade_states(ax, t_ds, states, up_color="mistyrose", down_color="aliceblue")

    # Legend patches
    up_patch   = mpatches.Patch(color="mistyrose", alpha=0.6, label="Up state")
    down_patch = mpatches.Patch(color="aliceblue", alpha=0.6, label="Down state")
    ax.legend(handles=[up_patch, down_patch], fontsize=8, loc="upper right")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PC1")
    ax.set_title("Up/Down state detection")

    return fig, ax

def plot_multichannel_states_overlay(
    states_data: dict,
    t_ds: np.ndarray,
    cfg: dict,
    channel_order: list[int] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot PC1 over time with Up/Down states indicated by background shading.

    Parameters
    ----------
    states_data : dict
        {"pc1": (n_samples, n_channels), "states": (n_samples, n_channels),
         "thresholds": (n_channels,), "epochs": [...]}
    t_ds : np.ndarray
    cfg : dict
    channel_order : list of int, optional
        1-based channel numbers giving the display (row) order.
        If None, defaults to the config's channel list or natural 1..N.
    """
    _apply_style(cfg)

    pc1_matrix    = states_data["pc1"]
    states_matrix = states_data["states"]
    thresholds    = states_data["thresholds"]

    n_samples, n_channels = pc1_matrix.shape
    t_start = cfg['visualization']['t_start']

    # Resolve display order
    if channel_order is None:
        channel_order = list(range(1, n_channels + 1))

    fig, axes = plt.subplots(
        nrows=len(channel_order),
        ncols=1,
        figsize=(14, 2 * len(channel_order)),
        sharex=True,
    )
    if len(channel_order) == 1:
        axes = [axes]

    for row_idx, ch_1based in enumerate(channel_order):
        col_idx = ch_1based - 1
        ax = axes[row_idx]

        if col_idx < 0 or col_idx >= n_channels:
            ax.text(0.5, 0.5, f"Ch {ch_1based} not in data",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=9)
            ax.set_ylabel(f"Ch {ch_1based}\nPC1", fontsize=8)
            continue

        ch_pc1    = pc1_matrix[:, col_idx]
        ch_states = states_matrix[:, col_idx]
        ch_thresh = thresholds[col_idx]

        ax.plot(t_ds, ch_pc1, color="gray", lw=0.7, alpha=0.8)

        if not np.isnan(ch_thresh):
            ax.axhline(ch_thresh, color="crimson", lw=1, ls="--")
            _shade_states(ax, t_ds, ch_states, up_color="#FF9999", down_color="#A8DADC")
        else:
            ax.set_facecolor("#F9F9F9")
            ax.text(0.98, 0.90, "No Threshold: Non-Bimodal Signal",
                    color="darkred", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, fontweight="bold",
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        ax.set_ylabel(f"Ch {ch_1based}\nPC1", fontsize=8)
        ax.grid(True, alpha=0.15)

    axes[0].set_title("All Channels — Up/Down State Detection Overlay",
                      fontsize=14, fontweight='bold')
    axes[-1].set_xlabel("Time (s)")

    up_patch   = mpatches.Patch(color="#FF9999", alpha=0.6, label="Up state")
    down_patch = mpatches.Patch(color="#A8DADC", alpha=0.6, label="Down state")
    axes[0].legend(handles=[up_patch, down_patch], fontsize=8, loc="upper right")

    current_ticks = axes[-1].get_xticks()
    new_labels = [f"{int(tick + t_start)}" for tick in current_ticks]
    axes[-1].set_xticks(current_ticks)
    axes[-1].set_xticklabels(new_labels)

    plt.tight_layout()
    save_figure(fig, "multichannel_state_overlay", cfg)
    return fig, axes

def _shade_states(
    ax: plt.Axes,
    t: np.ndarray,
    states: np.ndarray,
    up_color: str = "mistyrose",
    down_color: str = "aliceblue",
) -> None:
    """Fill background spans for Up (1) and Down (0) state regions."""
    in_run   = False
    run_val  = None
    run_start = None

    for i, (ti, si) in enumerate(zip(t, states)):
        if not in_run:
            in_run    = True
            run_val   = si
            run_start = ti
        elif si != run_val or i == len(states) - 1:
            color = up_color if run_val == 1 else down_color
            ax.axvspan(run_start, ti, alpha=0.4, color=color, linewidth=0)
            run_val   = si
            run_start = ti


# ---------------------------------------------------------------------------
# 5. Transition-aligned firing rate traces
# ---------------------------------------------------------------------------

def plot_aligned_firing_rates(
    up_traces: np.ndarray,
    down_traces: np.ndarray,
    t_win: np.ndarray,
    condition: str,
    cfg: dict,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot mean ± SEM of firing rate traces aligned to Up and Down onsets.

    Parameters
    ----------
    up_traces : np.ndarray, shape (n_up_events, n_window)
    down_traces : np.ndarray, shape (n_down_events, n_window)
    t_win : np.ndarray — time axis (s), 0 = onset
    condition : str
    cfg : dict
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    def _plot_mean_sem(traces, color, label):
        valid   = traces[~np.isnan(traces).any(axis=1)]
        if len(valid) == 0:
            return
        mean    = valid.mean(axis=0)
        sem     = valid.std(axis=0) / np.sqrt(len(valid))
        ax.plot(t_win, mean, color=color, lw=1.5, label=label)
        ax.fill_between(t_win, mean - sem, mean + sem,
                        alpha=0.25, color=color)

    _plot_mean_sem(up_traces,   "firebrick", f"Up onset (n={len(up_traces)})")
    _plot_mean_sem(down_traces, "steelblue", f"Down onset (n={len(down_traces)})")

    ax.axvline(0, color="black", lw=1, ls="--", label="Onset")
    ax.set_xlabel("Time relative to onset (s)")
    ax.set_ylabel("logMUA (a.u.)")
    ax.set_title(f"Transition-aligned firing rate — {condition}")
    ax.legend(fontsize=8)

    return fig, ax


# ---------------------------------------------------------------------------
# 6. Duration boxplots (per condition)
# ---------------------------------------------------------------------------

def plot_duration_boxplot(
    summary_df: pd.DataFrame,
    state: str,
    cfg: dict,
    stat_results: pd.DataFrame = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Boxplot of mean Up or Down state durations across slices, per condition.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Full results table (output of metrics.build_summary_table, concatenated
        across slices and conditions).
    state : str
        "up" or "down".
    cfg : dict
    stat_results : pd.DataFrame, optional
        Output of metrics.run_all_tests(). If provided, significance brackets
        are drawn above the boxes.
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)

    col = f"mean_{state}_s"
    conditions = cfg["conditions"]
    colors = _get_colors(conditions, cfg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    # Filter to conditions present in the data
    plot_df = summary_df[summary_df["condition"].isin(conditions)].copy()

    sns.boxplot(
        data=plot_df,
        x="condition", y=col,
        hue="condition",
        order=conditions,
        hue_order=conditions,
        palette={c: colors[c] for c in conditions},
        width=0.5,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="condition", y=col,
        order=conditions,
        color="black", size=4, alpha=0.6, jitter=True,
        ax=ax,
    )

    ax.set_xlabel("Condition")
    ax.set_ylabel(f"Mean {state.capitalize()} duration (s)")
    ax.set_title(f"{state.capitalize()} state duration")

    # Draw significance brackets if provided
    if stat_results is not None and not stat_results.empty:
        _draw_significance_brackets(ax, stat_results, conditions, plot_df, col)

    return fig, ax


def _draw_significance_brackets(
    ax, stat_results, conditions, data_df, col
):
    """Add significance brackets above boxplots."""
    y_max   = data_df[col].max()
    y_step  = y_max * 0.08
    y_start = y_max + y_step

    for _, row in stat_results.iterrows():
        if row.get("label", "ns") == "ns" or row.get("label") == "n/a":
            continue

        cond_a = row["condition_a"]
        cond_b = row["condition_b"]

        if cond_a not in conditions or cond_b not in conditions:
            continue

        x1 = conditions.index(cond_a)
        x2 = conditions.index(cond_b)
        y  = y_start

        ax.plot([x1, x1, x2, x2], [y, y + y_step * 0.4,
                                     y + y_step * 0.4, y],
                lw=1.2, color="black")
        ax.text((x1 + x2) / 2, y + y_step * 0.5, row["label"],
                ha="center", va="bottom", fontsize=10)
        y_start += y_step * 1.2


# ---------------------------------------------------------------------------
# 7. SO frequency bar plot
# ---------------------------------------------------------------------------

def plot_so_frequency(
    summary_df: pd.DataFrame,
    cfg: dict,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Bar plot of mean SO frequency (Hz) per condition, with SEM error bars.

    Parameters
    ----------
    summary_df : pd.DataFrame
    cfg : dict
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)
    conditions = cfg["conditions"]
    colors = _get_colors(conditions, cfg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    means = summary_df.groupby("condition")["so_frequency"].mean()
    sems  = summary_df.groupby("condition")["so_frequency"].sem()

    xs     = range(len(conditions))
    bar_colors = [colors[c] for c in conditions if c in means.index]
    bar_means  = [means.get(c, 0) for c in conditions]
    bar_sems   = [sems.get(c, 0)  for c in conditions]

    ax.bar(xs, bar_means, yerr=bar_sems, capsize=4,
           color=bar_colors, edgecolor="black", width=0.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel("SO frequency (Hz)")
    ax.set_title("Slow oscillation frequency")

    return fig, ax


# ---------------------------------------------------------------------------
# 8. Delta duration scatter / bar plot
# ---------------------------------------------------------------------------

def plot_delta_duration(
    delta_values: list[float],
    condition_label: str,
    state: str,
    cfg: dict,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot Δ duration (comparison − baseline) as individual points + mean ± SEM.

    Parameters
    ----------
    delta_values : list of float
        Δ duration values (one per slice).
    condition_label : str
        Label for the comparison condition (e.g. "stimulation").
    state : str
        "up" or "down".
    cfg : dict
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)

    vals = np.array([v for v in delta_values if not np.isnan(v)])

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))
    else:
        fig = ax.get_figure()

    # Individual points
    jitter  = np.random.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(jitter, vals, color="steelblue", s=30, alpha=0.7, zorder=3)

    # Mean ± SEM
    if len(vals) > 0:
        mean = vals.mean()
        sem  = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.errorbar(0, mean, yerr=sem, fmt="o", color="black",
                    capsize=6, markersize=8, zorder=4, label="Mean ± SEM")

    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_xlabel(f"{condition_label} vs baseline")
    ax.set_ylabel(f"Δ {state.capitalize()} duration (s)")
    ax.set_title(f"Change in {state} duration\n({condition_label})")

    return fig, ax


# ---------------------------------------------------------------------------
# 9. Trial overview (full recording with trial windows marked)
# ---------------------------------------------------------------------------

def plot_trial_overview(
    lfp: np.ndarray,
    fs: float,
    trials: list,
    cfg: dict,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the full raw LFP with trial windows and condition sub-windows
    overlaid as coloured spans. Use this to verify trial definitions
    before committing to the windowing step.

    Parameters
    ----------
    lfp : np.ndarray, shape (n_samples,)
        Full-session channel-averaged LFP.
    fs : float
        Original sampling rate (Hz).
    trials : list of TrialDefinition
        Output of define_trials.run_interactive() or run_from_json().
    cfg : dict

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)
    t = np.arange(len(lfp)) / fs

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, lfp, color="steelblue", lw=0.35, alpha=0.8)

    conditions = cfg["conditions"]
    colors     = _get_colors(conditions, cfg)
    trial_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(trials), 1)))

    for i, trial in enumerate(trials):
        t_color = trial_colors[i]

        # Full trial span (faint)
        ax.axvspan(trial.start_s, trial.end_s, alpha=0.08, color=t_color)

        # Trial label at top
        ax.text(
            (trial.start_s + trial.end_s) / 2,
            ax.get_ylim()[1],
            trial.label, ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=t_color,
        )

        # Condition sub-windows (stronger colour)
        for cw in trial.conditions:
            c_color = colors.get(cw.name, "gray")
            ax.axvspan(cw.start_s, cw.end_s, alpha=0.3, color=c_color)
            ax.text(
                (cw.start_s + cw.end_s) / 2,
                ax.get_ylim()[1] * 0.85,
                cw.name, ha="center", va="bottom",
                fontsize=7, color=c_color,
            )

    # Legend for conditions
    patches = [
        mpatches.Patch(color=colors[c], alpha=0.6, label=c)
        for c in conditions if c in colors
    ]
    if patches:
        ax.legend(handles=patches, fontsize=8, loc="upper right")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP (a.u.)")
    ax.set_title("Full recording — trial windows overview")

    return fig, ax


# ---------------------------------------------------------------------------
# 10. Artifact diagnostic (before / after rejection)
# ---------------------------------------------------------------------------

def plot_artifact_rejection(
    signal_raw: np.ndarray,
    signal_clean: np.ndarray,
    artifact_mask: np.ndarray,
    fs: float,
    cfg: dict,
    t_range_s: tuple = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one channel before and after artifact rejection, with artifact
    regions highlighted in red.

    Parameters
    ----------
    signal_raw : np.ndarray, shape (n_samples,)
        Original signal (before interpolation).
    signal_clean : np.ndarray, shape (n_samples,)
        Signal after artifact interpolation.
    artifact_mask : np.ndarray, shape (n_samples,), dtype bool
        True at artifact samples.
    fs : float
        Sampling rate (Hz).
    cfg : dict
    t_range_s : (start_s, end_s) or None
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)
    t = np.arange(len(signal_raw)) / fs

    if t_range_s is not None:
        mask_t        = (t >= t_range_s[0]) & (t <= t_range_s[1])
        t             = t[mask_t]
        signal_raw    = signal_raw[mask_t]
        signal_clean  = signal_clean[mask_t]
        artifact_mask = artifact_mask[mask_t]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))
    else:
        fig = ax.get_figure()

    ax.plot(t, signal_raw,   color="lightcoral", lw=0.5, alpha=0.7, label="Raw")
    ax.plot(t, signal_clean, color="steelblue",  lw=0.5, alpha=0.9, label="Clean")

    # Shade artifact regions
    _shade_states(ax, t, (~artifact_mask).astype(int),
                  up_color="white", down_color="mistyrose")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP (a.u.)")
    pct = artifact_mask.mean() * 100
    ax.set_title(f"Artifact rejection — {pct:.1f}% of samples flagged")
    ax.legend(fontsize=8, loc="upper right")

    return fig, ax

def plot_spectrogram(log_mua, times, freqs, Sxx_db, t_ds, trials, trial_start_s):
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 4), sharex=True)

    # Plot 1: logMUA (z-scored)
    ax1.plot(t_ds, log_mua, color='black', linewidth=0.5)
    ax1.set_ylabel('logMUA (z-scored)', fontsize=12)
    ax1.set_title('logMUA and Spectrogram across Trials', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spectrogram
    im = ax2.pcolormesh(
        times,
        freqs,
        Sxx_db,
        shading='auto',
        cmap='cool'
    )
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)

    # Shade stimulation periods across both plots
    colors_stim = {'baseline': None, 'stimulation': 'yellow', 'post_stimulation': 'orange'}

    for trial in trials:
        for cond in trial['conditions']:
            if cond['name'] in ['stimulation']:
                # Convert to relative time from trial_start_s
                start_rel = cond['start_s'] - trial_start_s
                end_rel = cond['end_s'] - trial_start_s
                
                color = colors_stim[cond['name']]
                alpha = 0.1
                
                ax1.axvspan(start_rel, end_rel, alpha=alpha, color=color, label=cond['name'])
                ax2.axvspan(start_rel, end_rel, alpha=alpha, color=color)

    # Add legend
    handles = [
        mpatches.Patch(facecolor='red', alpha=0.2, label='Stimulation'),
        mpatches.Patch(facecolor='orange', alpha=0.2, label='Post-stimulation')
    ]
    ax1.legend(handles=handles, loc='upper right')

    # Set x-axis to full trial duration
    ax2.set_xlim(0, t_ds[-1])

    plt.tight_layout()
    plt.show()

def plot_spectrogram_mua(log_mua, times, freqs, Sxx_db, t_ds, cfg):
    # --- FIX 1: Find the true end time of your actual data ---
    max_real_time = t_ds[-1]  # This will be exactly ~289.799s
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

    # Plot 1: logMUA (z-scored)
    print(f"t_ds: {t_ds}")
    ax1.plot(t_ds, log_mua, color='black', linewidth=0.5)
    ax1.set_ylabel('logMUA a.u.', fontsize=12)
    # ax1.set_title('logMUA and Spectrogram across Trials', fontsize=14)
    ax1.grid(False)
    ax1.set_xlim(t_ds[0], t_ds[-1]) # Force explicit axis boundary bounds

    ax1.get_yaxis().set_visible(False)

    for spine in ax1.spines.values():
        spine.set_visible(False)

    print(f"Spectrogram times: {times}")
    # Plot 2: Spectrogram
    im = ax2.pcolormesh(
        times,
        freqs,
        Sxx_db,
        shading='auto',
        cmap='turbo',
        # cmap='RdBu_r',
        vmin=-2.5,   # Keeps the baseline dark blue
        vmax=4.0
    )
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(r'$\log_{10}(P(\omega))$', fontsize=13)
    
    # Optional: Add an invisible placeholder box next to ax1 so their 
    # right edges match down to the exact pixel.
    divider_top = make_axes_locatable(ax1)
    cax_ghost = divider_top.append_axes("right", size="2%", pad=0.1)
    cax_ghost.axis('off') # Keep it hidden

    # Clean layout margins (keep this, it controls the space between subplots)
    fig.subplots_adjust(hspace=0.2)

    # --- SAVE AS TRANSPARENT IMAGE ---
    processed_dir = os.path.dirname(cfg["paths"]["processed_dir"])
    fig_dir = os.path.join(processed_dir, "results", "figures")
    t_start = cfg['visualization']['t_start']
    t_end = cfg['visualization']['t_end']

    # 2. Build the name of the dynamic subfolder (e.g., "3280-3340")
    time_folder = f"{t_start}-{t_end}"
    fig_dir = os.path.join(fig_dir, time_folder)

    filename = f"Spectrogram_MUA_{t_start}-{t_end}.png"
    os.makedirs(fig_dir, exist_ok=True)

    # Specify your path and filename. bbox_inches='tight' removes excess padding.
    output_path = os.path.join(fig_dir, filename)
    plt.savefig(
        output_path, 
        transparent=True,   # <-- Makes background transparent
        dpi=300,            # <-- High resolution for publication/presentations
        bbox_inches='tight' # <-- Safely crops edges without ruining alignment
    )
    print(f"💾 [Saved Transparent Plot]: {output_path}")

    # Display the plot on your screen
    plt.show()

def plot_delta_fr_boxplot(
    delta_data: dict[str, list[float]],
    state: str,
    cfg: dict,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Boxplot of Δ firing rate for all non-baseline conditions side by side,
    for one state (Up or Down).

    Parameters
    ----------
    delta_data : dict
        {condition_label: [delta_fr_values, ...]}
        e.g. {"stimulation": [0.1, -0.2, ...], "post_stimulation": [...]}
        Keys become the x-axis tick labels.
    state : str
        "up" or "down" — used in axis labels only.
    cfg : dict
    ax : plt.Axes, optional
    """
    _apply_style(cfg)
    conditions = cfg["conditions"]
    colors     = _get_colors(conditions, cfg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    labels      = list(delta_data.keys())
    data_arrays = [np.array([v for v in delta_data[l] if not np.isnan(v)])
                   for l in labels]
    positions   = list(range(len(labels)))

    # Boxplot
    bp = ax.boxplot(
        data_arrays,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )

    # Colour each box by its condition
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(colors.get(label, "steelblue"))
        patch.set_alpha(0.65)

    # Overlay individual paired points connected by lines
    for i, (arr, pos) in enumerate(zip(data_arrays, positions)):
        jitter = np.random.uniform(-0.06, 0.06, size=len(arr))
        ax.scatter(pos + jitter, arr,
                   color=colors.get(labels[i], "steelblue"),
                   s=28, zorder=4, alpha=0.85, edgecolors="white", linewidths=0.5)

    # Connect paired points across conditions with thin grey lines
    n_pairs = min(len(a) for a in data_arrays)
    if len(data_arrays) > 1 and n_pairs > 0:
        for i in range(n_pairs):
            y_vals = [arr[i] for arr in data_arrays if i < len(arr)]
            x_vals = list(range(len(y_vals)))
            ax.plot(x_vals, y_vals, color="gray", lw=0.7, alpha=0.4, zorder=2)

    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_ylabel(f"Δ {state.capitalize()} FR (logMUA a.u.)")
    ax.set_title(f"Δ {state.capitalize()} firing rate vs baseline")

    return fig, ax

def plot_up_state_raster(
    up_traces: np.ndarray,
    t_win: np.ndarray,
    cfg: dict,
    sort_by_duration: bool = True,
    vmin: float = None,
    vmax: float = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Raster plot of instantaneous firing rates aligned to each Up state onset.

    Each row is one Up state. The x-axis is time relative to onset (0 = onset).
    Colour encodes the logMUA instantaneous firing rate.

    Parameters
    ----------
    up_traces : np.ndarray, shape (n_up_states, n_window_samples)
        Output of firing_rates.compute_aligned_traces() — each row is one
        Up-state onset-aligned logMUA snippet.
    t_win : np.ndarray, shape (n_window_samples,)
        Time axis in seconds, centred on 0 (= Up state onset).
    cfg : dict
    sort_by_duration : bool
        If True, sort rows by peak firing rate within the post-onset window
        (makes spatial structure more visible). If False, keep detection order.
    vmin, vmax : float, optional
        Colourmap limits. Defaults to 5th / 95th percentile of the data
        (robust to outliers).
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)

    # Drop rows that are entirely NaN (onset too close to edge of recording)
    valid_mask  = ~np.isnan(up_traces).all(axis=1)
    traces      = up_traces[valid_mask].copy()
    n_states    = traces.shape[0]

    if n_states == 0:
        print("[plot_up_state_raster] No valid Up-state traces to plot.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()
        ax.text(0.5, 0.5, "No valid Up-state traces",
                ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    # Replace remaining NaNs within valid rows with local row median
    # (handles partial edge clips gracefully without distorting the colour scale)
    for i in range(n_states):
        row_nan = np.isnan(traces[i])
        if row_nan.any():
            median_val = np.nanmedian(traces[i])
            traces[i, row_nan] = median_val

    # Sort rows by mean firing rate in the first 200 ms after onset
    # → Up states with strongest early activation cluster at top
    if sort_by_duration:
        post_onset_mask = t_win >= 0
        sort_key        = traces[:, post_onset_mask].mean(axis=1)
        order           = np.argsort(sort_key)[::-1]   # descending
        traces          = traces[order]

    # Robust colour limits
    if vmin is None:
        vmin = np.nanpercentile(traces, 5)
    if vmax is None:
        vmax = np.nanpercentile(traces, 95)

    if ax is None:
        # Scale figure height with number of states (min 4, max 14 inches)
        height = float(np.clip(n_states * 0.18, 4, 14))
        fig, ax = plt.subplots(figsize=(10, height))
    else:
        fig = ax.get_figure()

    img = ax.imshow(
        traces,
        aspect="auto",
        origin="upper",
        extent=[t_win[0], t_win[-1], n_states + 0.5, 0.5],
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Onset line
    ax.axvline(0, color="cyan", lw=1.2, ls="--", alpha=0.85, label="Up onset")

    # Colourbar
    cbar = fig.colorbar(img, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("logMUA (a.u.)", fontsize=9)

    ax.set_xlabel("Time relative to Up onset (s)")
    ax.set_ylabel("Up state #")
    ax.set_title(f"Instantaneous firing rate — Up states (n={n_states})")
    ax.legend(fontsize=8, loc="upper left")

    return fig, ax


def plot_up_state_raster(
    up_traces: np.ndarray,
    t_win: np.ndarray,
    epochs: dict,
    log_mua_raw: np.ndarray,
    t_ds: np.ndarray,
    cfg: dict,
    sort_by_peak_fr: bool = True,
    pre_baseline_s: float = 0.3,
    vmin: float = None,
    vmax: float = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Raster plot of RELATIVE instantaneous firing rates aligned to each
    Up state PEAK (maximum logMUA), not onset.

    Relative FR = (logMUA(t) - mean_logMUA_pre) / |mean_logMUA_pre|
    where "pre" is the `pre_baseline_s` seconds immediately before
    the peak. This makes the colour axis a fractional change:
        0   = same as pre-peak baseline
       +1   = twice the baseline firing rate
       -0.5 = half the baseline firing rate

    Parameters
    ----------
    up_traces : np.ndarray, shape (n_up_states, n_window_samples)
        Onset-aligned logMUA snippets from firing_rates.compute_aligned_traces().
        Used only to identify valid states; we re-extract peak-aligned snippets
        from log_mua_raw directly.
    t_win : np.ndarray, shape (n_window_samples,)
        Time axis from compute_aligned_traces() — used to infer fs and window size.
    epochs : dict
        exp.epochs — {"up": [(start_s, end_s, duration_s), ...], ...}
        Used to find the peak within each Up epoch.
    log_mua_raw : np.ndarray, shape (n_ds_samples,)
        The raw (un-z-scored) logMUA at the downsampled rate. exp.log_mua_raw.
    t_ds : np.ndarray, shape (n_ds_samples,)
        Downsampled time vector. exp.t_ds.
    cfg : dict
    sort_by_peak_fr : bool
        If True, sort rows by peak relative FR (descending).
        If False, keep chronological (detection) order.
    pre_baseline_s : float
        Duration of the pre-peak window used to compute the baseline FR
        for normalisation (seconds). Default 0.3 s.
    vmin, vmax : float, optional
        Colourmap limits. Defaults to 5th/95th percentile of relative FR.
    ax : plt.Axes, optional

    Returns
    -------
    fig, ax
    """
    _apply_style(cfg)

    fs_ds      = 1.0 / float(t_ds[1] - t_ds[0])
    win_s      = float(t_win[-1])               # half-window in seconds
    win_samples = int(round(win_s * fs_ds))
    n_window    = 2 * win_samples + 1
    t_centred   = np.linspace(-win_s, win_s, n_window)

    pre_samples = max(1, int(round(pre_baseline_s * fs_ds)))
    eps         = 1e-12

    up_epoch_list = epochs["up"]
    traces_rel    = []    # relative FR snippets
    valid_indices = []    # which Up states made it in

    for i, (start_s, end_s, _) in enumerate(up_epoch_list):
        # Indices of this epoch in the downsampled signal
        lo_ep = int(round(start_s * fs_ds))
        hi_ep = int(round(end_s   * fs_ds))
        lo_ep = max(0, lo_ep)
        hi_ep = min(len(log_mua_raw), hi_ep)

        if hi_ep <= lo_ep:
            continue

        # Find the peak within this Up epoch
        epoch_sig  = log_mua_raw[lo_ep:hi_ep]
        peak_local = int(np.argmax(epoch_sig))
        peak_idx   = lo_ep + peak_local

        # Extract window centred on peak
        lo_win = peak_idx - win_samples
        hi_win = peak_idx + win_samples + 1

        if lo_win < 0 or hi_win > len(log_mua_raw):
            continue    # too close to edge — skip

        snippet = log_mua_raw[lo_win:hi_win].copy()

        # Pre-peak baseline: `pre_samples` immediately before the peak
        pre_start = win_samples - pre_samples   # index within snippet
        pre_mean  = snippet[pre_start:win_samples].mean()

        # Relative FR: fractional change from pre-peak baseline
        rel_snippet = (snippet - pre_mean) / (abs(pre_mean) + eps)

        traces_rel.append(rel_snippet)
        valid_indices.append(i)

    if len(traces_rel) == 0:
        print("[plot_up_state_raster] No valid Up-state traces to plot.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()
        ax.text(0.5, 0.5, "No valid Up-state traces",
                ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    traces = np.array(traces_rel)   # shape (n_valid, n_window)
    n_states = traces.shape[0]

    if sort_by_peak_fr:
        # Sort by peak relative FR (the centre column is the peak by definition,
        # but small jitter from the global vs local peak means we sort on
        # the max value in a ±50ms window around centre)
        half = max(1, int(round(0.05 * fs_ds)))
        centre = win_samples
        peak_col = traces[:, centre - half : centre + half + 1].max(axis=1)
        order    = np.argsort(peak_col)[::-1]
        traces   = traces[order]

    # Colour limits — relative FR, so 0 = baseline
    if vmin is None:
        vmin = np.nanpercentile(traces, 5)
    if vmax is None:
        vmax = np.nanpercentile(traces, 95)
    # Force the scale to be symmetric around a sensible range if both
    # sides are positive (can happen if pre-baseline estimate is low)
    if vmin > 0:
        vmin = -0.1

    if ax is None:
        height = float(np.clip(n_states * 0.2, 5, 16))
        fig, ax = plt.subplots(figsize=(10, height))
    else:
        fig = ax.get_figure()

    img = ax.imshow(
        traces,
        aspect="auto",
        origin="upper",
        extent=[t_centred[0], t_centred[-1], n_states + 0.5, 0.5],
        cmap="RdBu_r",       # diverging: blue=below baseline, red=above
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Peak line
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.9, label="Peak")

    # Colourbar
    cbar = fig.colorbar(img, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Relative FR\n(fraction of pre-peak baseline)", fontsize=9)

    # --- Discrete Y axis ---
    if n_states <= 30:
        # Show every state number
        ax.set_yticks(range(1, n_states + 1))
    else:
        # Show a tick every 10 states to avoid crowding
        step = max(1, round(n_states / 20) * 5)   # nearest multiple of 5
        ax.set_yticks(range(1, n_states + 1, step))

    ax.set_xlabel("Time relative to Up-state peak (s)")
    ax.set_ylabel("Up state #")
    ax.set_title(f"Relative firing rate — Up states (n={n_states})")
    ax.legend(fontsize=8, loc="upper left")

    return fig, ax


def plot_delta_duration_joint(
    delta_data: dict[str, tuple[list[float], list[float]]],
    cfg: dict,
) -> plt.Figure:
    """
    Joint scatter + 2D KDE cloud plot of Δup (y) vs Δdown (x) for multiple
    conditions overlaid on the same axes, with marginal 1D KDEs.

    Parameters
    ----------
    delta_data : dict
        {condition_label: (delta_up, delta_down)}
        e.g. {
            "stimulation":      ([0.1, -0.2, ...], [0.05, -0.1, ...]),
            "post_stimulation": ([0.2, -0.1, ...], [0.1,  0.0, ...]),
        }
        Values are lists of per-trial Δ durations in seconds.
    cfg : dict

    Returns
    -------
    plt.Figure
    """
    from scipy.stats import gaussian_kde

    _apply_style(cfg)

    all_conditions = cfg.get("conditions", list(delta_data.keys()))
    colors         = _get_colors(all_conditions, cfg)

    # ---------------------------------------------------------------------------
    # Grid layout: 2×2
    #   [top marginal  | empty         ]
    #   [main scatter  | right marginal]
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(7, 7))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios  = [4, 1.2],
        height_ratios = [1.2, 4],
        hspace=0.04,
        wspace=0.04,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Reference lines
    ax_main.axhline(0, color="gray", lw=0.9, ls="--", alpha=0.6, zorder=1)
    ax_main.axvline(0, color="gray", lw=0.9, ls="--", alpha=0.6, zorder=1)
    ax_top.axvline(0,  color="gray", lw=0.9, ls="--", alpha=0.6)
    ax_right.axhline(0, color="gray", lw=0.9, ls="--", alpha=0.6)

    # Collect all values to build a shared evaluation grid across conditions
    all_dd, all_du = [], []
    clean_data = {}   # {label: (du_array, dd_array)}

    for label, (delta_up, delta_down) in delta_data.items():
        pairs = [(u, d) for u, d in zip(delta_up, delta_down)
                 if not (np.isnan(u) or np.isnan(d))]
        if len(pairs) < 2:
            print(f"  [{label}] only {len(pairs)} valid pair(s) — skipping KDE.")
            continue
        du = np.array([p[0] for p in pairs])
        dd = np.array([p[1] for p in pairs])
        clean_data[label] = (du, dd)
        all_du.extend(du)
        all_dd.extend(dd)

    if not clean_data:
        ax_main.text(0.5, 0.5, "Not enough data",
                     ha="center", va="center", transform=ax_main.transAxes)
        return fig

    all_du = np.array(all_du)
    all_dd = np.array(all_dd)

    # Shared evaluation grid with padding
    pad_x = max((all_dd.max() - all_dd.min()) * 0.35, 0.05)
    pad_y = max((all_du.max() - all_du.min()) * 0.35, 0.05)
    x_grid = np.linspace(all_dd.min() - pad_x, all_dd.max() + pad_x, 200)
    y_grid = np.linspace(all_du.min() - pad_y, all_du.max() + pad_y, 200)
    XX, YY = np.meshgrid(x_grid, y_grid)

    # ---------------------------------------------------------------------------
    # Per-condition: 2D KDE cloud + scatter + marginal KDEs
    # ---------------------------------------------------------------------------
    for label, (du, dd) in clean_data.items():
        color = colors.get(label, "steelblue")
        n     = len(du)

        # 2D KDE cloud
        kde_2d = gaussian_kde(np.vstack([dd, du]), bw_method="scott")
        ZZ     = kde_2d(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

        ax_main.contourf(XX, YY, ZZ, levels=6, colors=[color], alpha=0.15)
        ax_main.contour( XX, YY, ZZ, levels=4, colors=[color],
                         alpha=0.7, linewidths=1.0)

        # Scatter points
        ax_main.scatter(dd, du, color=color, s=50, zorder=5,
                        edgecolors="white", linewidths=0.6, alpha=0.95,
                        label=f"{label.replace('_', ' ')} (n={n})")

        # Top marginal — 1D KDE of Δdown (x axis)
        kde_top = gaussian_kde(dd, bw_method="scott")
        y_top   = kde_top(x_grid)
        ax_top.fill_between(x_grid, y_top, alpha=0.2, color=color)
        ax_top.plot(x_grid, y_top, color=color, lw=1.5)

        # Right marginal — 1D KDE of Δup (y axis)
        kde_right = gaussian_kde(du, bw_method="scott")
        x_right   = kde_right(y_grid)
        ax_right.fill_betweenx(y_grid, x_right, alpha=0.2, color=color)
        ax_right.plot(x_right, y_grid, color=color, lw=1.5)

    # ---------------------------------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------------------------------
    ax_main.set_xlabel("Δ Down duration (s)")
    ax_main.set_ylabel("Δ Up duration (s)")
    ax_main.legend(fontsize=9, loc="upper left", framealpha=0.8)

    ax_top.set_ylabel("Density")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title("Δ duration joint distribution — stimulation vs baseline",
                     fontsize=11, pad=8)

    ax_right.set_xlabel("Density")
    ax_right.tick_params(labelleft=False)

    return fig


# ---------------------------------------------------------------------------
# 11. Per-channel logMUA stats heatmap + threshold
# ---------------------------------------------------------------------------

def plot_channel_stats(
    stats_df,
    threshold: float,
    high_channels: list,
    low_channels: list,
    cfg: dict,
    metric: str = "mean_logmua",
) -> tuple[plt.Figure, np.ndarray]:
    """
    Two-panel figure showing per-channel logMUA statistics and the
    KDE distribution used to find the high/low threshold.

    Panel A (left): bar chart of `metric` per channel (1-based labels),
                    bars coloured by group (high = red, low = blue).
    Panel B (right): KDE of `metric` values across channels with the
                     threshold marked.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of channel_analysis.compute_channel_stats().
        Must have columns: channel_1based, mean_logmua, var_logmua.
    threshold : float
        The KDE valley threshold separating high from low channels.
    high_channels : list of int
        0-based indices of high channels.
    low_channels : list of int
        0-based indices of low channels.
    cfg : dict
    metric : str
        "mean_logmua" or "var_logmua".
    """
    from scipy.stats import gaussian_kde as _gkde

    _apply_style(cfg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [3, 1.2]})
    ax_bar, ax_kde = axes

    values      = stats_df[metric].values
    ch_1based   = stats_df["channel_1based"].values    # display labels
    ch_0based   = stats_df["channel_0based"].values

    # Colours per bar
    bar_colors = [
        "#d62728" if c in high_channels else "#1f77b4"   # red / blue
        for c in ch_0based
    ]

    # --- Panel A: bar chart ---
    x_pos = np.arange(len(ch_1based))
    ax_bar.bar(x_pos, values, color=bar_colors, edgecolor="white",
               linewidth=0.4, width=0.8)
    ax_bar.axhline(threshold, color="black", lw=1.5, ls="--",
                   label=f"Threshold = {threshold:.3f}")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(ch_1based, fontsize=7, rotation=90)
    ax_bar.set_xlabel("Channel (1-based)")
    metric_label = "Mean logMUA" if metric == "mean_logmua" else "logMUA variance"
    ax_bar.set_ylabel(metric_label)
    ax_bar.set_title(f"Per-channel {metric_label}\n"
                     f"High (red): n={len(high_channels)}  "
                     f"Low (blue): n={len(low_channels)}")
    ax_bar.legend(fontsize=9)

    # --- Panel B: KDE with threshold ---
    kde   = _gkde(values, bw_method="scott")
    x_ev  = np.linspace(values.min() - abs(values.std()), 
                        values.max() + abs(values.std()), 400)
    dens  = kde(x_ev)

    ax_kde.fill_betweenx(x_ev, dens, alpha=0.25, color="gray")
    ax_kde.plot(dens, x_ev, color="black", lw=1.5)
    ax_kde.axhline(threshold, color="black", lw=1.5, ls="--",
                   label=f"Threshold\n{threshold:.3f}")

    # Shade high / low regions
    ax_kde.fill_between(
        [0, dens.max() * 1.1],
        threshold, x_ev.max(),
        alpha=0.12, color="#d62728", label="High"
    )
    ax_kde.fill_between(
        [0, dens.max() * 1.1],
        x_ev.min(), threshold,
        alpha=0.12, color="#1f77b4", label="Low"
    )

    ax_kde.set_xlabel("Density")
    ax_kde.set_ylabel(metric_label)
    ax_kde.set_title("Channel distribution")
    ax_kde.legend(fontsize=8)
    ax_kde.set_xlim(left=0)

    fig.suptitle("Channel classification — stimulation files", fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_group_comparison(
    summary_high: pd.DataFrame,
    summary_low: pd.DataFrame,
    cfg: dict,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Six-panel comparison figure between high and low channel groups,
    reusing the existing per-condition plotting functions.

    Panels (one column per channel group, rows = metrics):
        Row 1 : Up duration boxplot
        Row 2 : Down duration boxplot
        Row 3 : SO frequency bar
        Row 4 : Up FR delta (stim vs baseline)
        Row 5 : Down FR delta (stim vs baseline)
        Row 6 : Up FR delta (post vs baseline)    [if post_stimulation present]

    Actually rendered as a flexible grid: 3 metric rows × 2 group columns.

    Parameters
    ----------
    summary_high : pd.DataFrame
        summary_df for the high-channel group (all conditions, all trials).
    summary_low : pd.DataFrame
        summary_df for the low-channel group.
    cfg : dict

    Returns
    -------
    fig, axes  shape (3, 2)
    """
    _apply_style(cfg)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    group_labels = ["High channels", "Low channels"]
    summaries    = [summary_high, summary_low]

    for col, (label, sdf) in enumerate(zip(group_labels, summaries)):
        # Row 0: Up duration
        plot_duration_boxplot(sdf, "up",   cfg, ax=axes[0, col])
        axes[0, col].set_title(f"{label}\nUp duration")

        # Row 1: Down duration
        plot_duration_boxplot(sdf, "down", cfg, ax=axes[1, col])
        axes[1, col].set_title(f"{label}\nDown duration")

        # Row 2: SO frequency
        plot_so_frequency(sdf, cfg, ax=axes[2, col])
        axes[2, col].set_title(f"{label}\nSO frequency")

    fig.suptitle("High vs Low channel groups — Up/Down metrics", fontsize=13)
    plt.tight_layout()
    return fig, axes


def plot_multichannel_raw_lfp(
    lfp_all: np.ndarray,
    fs: float,
    cfg: dict,
    band: tuple = (0.1, 300),
    channel_order: list[int] = None,
) -> plt.Figure:
    """
    Plot all channels of the raw LFP after bandpass filtering (default 0.1–300 Hz).

    Parameters
    ----------
    lfp_all : np.ndarray, shape (n_samples, n_channels) or (n_samples,)
    fs : float
        Original sampling rate.
    cfg : dict
    band : (low_hz, high_hz)
        Bandpass edges. Default (0.1, 300).
    channel_order : list of int, optional
        1-based channel numbers giving the display (row) order.
        If None, defaults to natural 1..N ordering. Use
        src.utils.get_electrode_display_order(N) to get the physical order.
    """
    from src.preprocessing import bandpass_filter

    if lfp_all.ndim == 1:
        lfp_all = lfp_all[:, np.newaxis]

    n_samples, n_channels = lfp_all.shape
    t = np.arange(n_samples) / fs
    t_start = cfg["visualization"].get("t_start", 0)

    # Resolve display order
    if channel_order is None:
        channel_order = list(range(1, n_channels + 1))

    fig, axes = plt.subplots(
        nrows=len(channel_order), ncols=1,
        figsize=(14, 2 * len(channel_order)),
        sharex=True,
    )
    if len(channel_order) == 1:
        axes = [axes]

    # Iterate through the display order — row_idx = subplot position,
    # ch_1based = channel number to display, col_idx = data column
    for row_idx, ch_1based in enumerate(channel_order):
        col_idx = ch_1based - 1
        if col_idx < 0 or col_idx >= n_channels:
            # Requested a channel that doesn't exist in the data — skip
            axes[row_idx].text(0.5, 0.5, f"Ch {ch_1based} not in data",
                               ha="center", va="center",
                               transform=axes[row_idx].transAxes,
                               color="gray", fontsize=9)
            axes[row_idx].set_ylabel(f"Ch {ch_1based}", fontsize=8)
            continue

        filtered = bandpass_filter(
            lfp_all[:, col_idx].astype(np.float64), fs, band[0], band[1], order=4
        )
        axes[row_idx].plot(t, filtered, color="black", lw=0.4)
        axes[row_idx].set_ylabel(f"Ch {ch_1based}\n({band[0]}–{band[1]} Hz)", fontsize=8)
        axes[row_idx].axhline(0, color="gray", lw=0.5, ls="--")
        axes[row_idx].grid(True, alpha=0.15)

    axes[0].set_title(
        f"All Channels — Raw LFP bandpassed {band[0]}–{band[1]} Hz",
        fontsize=14, fontweight="bold",
    )
    axes[-1].set_xlabel("Time (s)")

    current_ticks = axes[-1].get_xticks()
    new_labels = [f"{int(tick + t_start)}" for tick in current_ticks]
    axes[-1].set_xticks(current_ticks)
    axes[-1].set_xticklabels(new_labels)

    plt.tight_layout()
    save_filename = f"multichannel_raw_lfp_{band[0]}-{band[1]}Hz"
    save_figure(fig, save_filename, cfg)
    return fig
# ---------------------------------------------------------------------------
# Extension points
# ---------------------------------------------------------------------------
# - To add a new plot type: add a function plot_<thing>(data, cfg, ax=None)
#   following the same (fig, ax) return convention.
#
# - To build a multi-panel summary figure: create a function make_summary_figure()
#   that calls individual plot_* functions with explicit ax= arguments.
#
# - To switch to an interactive backend (e.g. plotly):
#   Add a parallel set of plotly_* functions; swap them in notebooks.
