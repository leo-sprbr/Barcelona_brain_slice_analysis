import sys, os

# from src.core.metrics import compute_firing_rates
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from src.config import load_config
from src.data_io import load_h5_time_window, list_h5_file_single
from src.preprocessing import select_channel_raw, _extract_raw_log_mua, extract_features_all_channels
from src.visualization import plot_multichannel_states_overlay, plot_raw_lfp_time_window, plot_multichannel_feature_group, plot_pc1_distribution, plot_states_overlay, plot_spectrogram, plot_spectrogram_mua
from src.core.up_down import detect_states
from src.utils import spectrogram

def print_channel_metrics_summary(metrics_data: dict, states_data: dict, cfg: dict) -> None:
    """
    Accepts the pipeline metrics and state structures to print a beautifully
    formatted, human-readable statistical summary table for all channels.
    Safely unpacks data metrics regardless of whether they are formatted as
    standard Python dictionaries, Pandas Series, or DataFrames.
    """
    # 1. Grab the total number of processed channels
    n_processed_channels = len(metrics_data["summaries"])
    
    # 2. Grab the configured channel labels
    channel_list = cfg["preprocessing"].get("channel", None)
    
    if channel_list is None or channel_list == []:
        channel_list = list(range(1, n_processed_channels + 1))
        
    # Helper to safely extract a clean float out of potential Pandas objects
    def _to_float(item, key_name):
        if item is None:
            return 0.0
        
        # Extract the value depending on whether it's a DataFrame, Series, or dict
        try:
            if hasattr(item, "get"):
                val = item.get(key_name, 0.0)
            else:
                val = item[key_name]
        except (KeyError, IndexError, ValueError):
            return 0.0

        # If the extracted value itself is a Pandas Series, pull the first value
        if hasattr(val, "iloc"):
            if val.empty:
                return 0.0
            val = val.iloc[0]
            
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    # 3. Collect all lines of text in a tracking list
    lines = []
    lines.append("="*50)
    lines.append("         METRICS SUMMARY PER CHANNEL         ")
    lines.append("="*50)
    
    # 4. Iterate through every channel summary row stored in metrics_data
    for idx, summary in enumerate(metrics_data["summaries"]):
        real_ch = channel_list[idx] if idx < len(channel_list) else (idx + 1)
        
        lines.append(f"\n▶ Hardware Channel {real_ch}:")
        
        # Check if this channel threw a KDE bimodal error and was skipped
        if isinstance(summary, dict) and "status" in summary:
            lines.append(f"  ❌ Status: SKIPPED (KDE non-bimodal distribution)")
            continue
        elif np.isnan(states_data["thresholds"][idx]):
            lines.append(f"  ❌ Status: SKIPPED (KDE non-bimodal distribution)")
            continue
            
        # 5. Extract and format values cleanly using our adaptive unpacking tool
        n_up      = int(_to_float(summary, 'n_up'))
        n_down    = int(_to_float(summary, 'n_down'))
        mean_up   = _to_float(summary, 'mean_up_s')
        sem_up    = _to_float(summary, 'sem_up_s')
        mean_down = _to_float(summary, 'mean_down_s')
        sem_down  = _to_float(summary, 'sem_down_s')
        so_freq   = _to_float(summary, 'so_frequency')
        
        # 6. Add the cleaned text metrics to our list
        lines.append(f"  • Total States Detected : Up: {n_up} | Down: {n_down}")
        lines.append(f"  • Mean Up Duration      : {mean_up:.3f} seconds (± {sem_up:.3f} SEM)")
        lines.append(f"  • Mean Down Duration    : {mean_down:.3f} seconds (± {sem_down:.3f} SEM)")
        lines.append(f"  • Slow Oscillation Freq : {so_freq:.3f} Hz")
        
        # 7. Unpack and print your multi-unit firing rate intensities (mean logMUA)
        mean_up_fr   = _to_float(summary, 'mean_fr_up')
        mean_down_fr = _to_float(summary, 'mean_fr_down')
        
        lines.append(f"  • Mean Up Firing Rate   : {mean_up_fr:.3f} (logMUA strength)")
        lines.append(f"  • Mean Down Firing Rate : {mean_down_fr:.3f} (logMUA strength)")
            
    lines.append("\n" + "="*50)
    
    # 8. Join the lines together into a single block of string text
    full_output_text = "\n".join(lines)
    
    # 9. Write out to the console terminal screen
    print(full_output_text)
    

    processed_dir = os.path.dirname(cfg["paths"]["processed_dir"])
    fig_dir = os.path.join(processed_dir, "results", "figures")
    t_start = cfg['visualization']['t_start']
    t_end = cfg['visualization']['t_end']

    # 2. Build the name of the dynamic subfolder (e.g., "3280-3340")
    time_folder = f"{t_start}-{t_end}"
    fig_dir = os.path.join(fig_dir, time_folder)
    os.makedirs(fig_dir, exist_ok=True)

    metric_summary_path = os.path.join(fig_dir, f"Metric_summary_{time_folder}.txt")
    
    try:
        with open(metric_summary_path, "w", encoding="utf-8") as f:
            f.write(full_output_text)
        print(f"\n💾 [Saved Successfully]: Full report text written to '{os.path.abspath(metric_summary_path)}'\n")
    except Exception as e:
        print(f"\n⚠️ Warning: Could not write metrics to text file due to error: {e}\n")

from scipy import signal

def extract_lfp_from_raw(raw_signal, fs_raw, target_fs=1000.0, cutoff=300.0):
    """
    Extracts a clean LFP signal from raw high-frequency electrode data
    by low-pass filtering and downsampling.
    """
    shape = raw_signal.shape
    print(f"Extracting LFP from raw signal of shape: {shape}, fs_raw: {fs_raw}, target_fs: {target_fs}, cutoff: {cutoff}")
    # 1. Design a low-pass Butterworth filter
    # Nyquist frequency of the raw data
    nyquist = 0.5 * fs_raw
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    
    # 2. Filter the signal forward and backward (zero-phase shift)
    filtered_signal = signal.filtfilt(b, a, raw_signal)
    
    # 2. THE FIX: Rational resampling from 3125 Hz to 1000 Hz
    # We convert the ratio 1000 / 3125 to the integer fraction 8 / 25
    up = int(target_fs)     # 1000
    down = int(fs_raw)      # 3125
    
    # resample_poly changes the rate by up/down perfectly
    lfp_signal = signal.resample_poly(filtered_signal, up, down)
    
    shape_lfp = lfp_signal.shape
    print(f"Downsampled LFP signal shape: {shape_lfp}, new fs: {target_fs}")
    return lfp_signal

def remove_line_noise(lfp_signal, fs, noise_freq=50.0, bandwidth=2.0):
    """
    Applies a sharp digital notch filter to eliminate 50 Hz power line noise.
    """
    # Quality factor Q dictates how tight the notch is 
    # (High Q means it only kills 50 Hz and leaves 49 Hz and 51 Hz untouched)
    Q = noise_freq / bandwidth
    
    # Design the notch filter coefficients
    b, a = signal.iirnotch(noise_freq, Q, fs=fs)
    
    # Apply the filter forward and backward (zero phase shift)
    clean_lfp = signal.filtfilt(b, a, lfp_signal)
    
    return clean_lfp

def plot_lfp(lfp, fs, cfg):
    """
    Plots the raw LFP signal over time.
    """
    t = np.arange(len(lfp)) / fs
    plt.figure(figsize=(12, 4))
    # plt.plot(t, lfp, color="#566774", linewidth=0.5)
    plt.plot(t, lfp, color='#3681ba', linewidth=0.5)
    # plt.plot(t, lfp, color='#c13639', linewidth=0.5)
    plt.title('Raw LFP Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.xlim(t[0], t[-1])
    plt.grid(False)
    plt.tight_layout()

    # --- SAVE AS TRANSPARENT IMAGE ---
    processed_dir = os.path.dirname(cfg["paths"]["processed_dir"])
    fig_dir = os.path.join(processed_dir, "results", "figures")
    t_start = cfg['visualization']['t_start']
    t_end = cfg['visualization']['t_end']

    # 2. Build the name of the dynamic subfolder (e.g., "3280-3340")
    time_folder = f"{t_start}-{t_end}"
    fig_dir = os.path.join(fig_dir, time_folder)

    filename = f"LFP_{t_start}-{t_end}.png"
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



    plt.show()


if __name__ == "__main__":
    # Load configuration
    config_path = 'config4singleFile.yaml'
    cfg = load_config(config_path)

    # Load the .h5 file specified in the config
    H5_PATH = list_h5_file_single(cfg)
    if H5_PATH is None:
        print("No matching .h5 file found.")
    else:
        print('Selected file:', os.path.basename(H5_PATH))

    # Load data and metadata
    data, meta = load_h5_time_window(H5_PATH, cfg)
    fs  = meta['fs']
    lfp = select_channel_raw(data, cfg['preprocessing'])  # For now it is the LFP of channel 1.
    t   = np.arange(len(lfp)) / fs

    print(f'File     : {os.path.basename(H5_PATH)}')
    print(f'Duration : {meta["duration_seconds"]:.1f} s')
    print(f'Channels : {meta["n_channels"]}')
    print(f'Fs       : {fs} Hz')

    # Save the figure of raw LFP data
    fig, axes = plot_raw_lfp_time_window(lfp, fs, cfg)

    # features, t_ds = extract_features(lfp, meta['fs'], cfg['preprocessing'])
    # print(f'Feature matrix shape: {features.shape}')
    # print(f'Downsampled fs: {1/(t_ds[1]-t_ds[0]):.1f} Hz')

    # fig, axes = plot_features(features, t_ds, cfg)
    # plt.show()

    # 1. Run the loop collector over all recording tracks
    grouped_feats, states_data, metrics_data, t_ds = extract_features_all_channels(lfp, meta['fs'], cfg)
    
    # 2. Image 1: Plot all channels of Slow Oscillations (SO)
    fig_so = plot_multichannel_feature_group(grouped_feats["SO"], t_ds, "Slow Oscillations (SO)", cfg)
    
    # 3. Image 2: Plot all channels of Gamma Envelope
    fig_gamma = plot_multichannel_feature_group(grouped_feats["gamma"], t_ds, "Gamma Envelope", cfg)
    
    # 4. Image 3: Plot all channels of logMUA
    fig_mua = plot_multichannel_feature_group(grouped_feats["logMUA"], t_ds, "logMUA Data", cfg)

    # 5. Image 4: Plot all channels of PC1 and up-down states overlay
    fig, axes = plot_multichannel_states_overlay(states_data, t_ds, cfg)


    # 6. Call your brand new independent summary printer!
    print_channel_metrics_summary(metrics_data, states_data, cfg)
    
    # 7. Render everything to the desktop interface
    # plt.show()
    lfp_clean = extract_lfp_from_raw(lfp, fs_raw=meta['fs'], target_fs=1000.0)
    lfp_clean = remove_line_noise(lfp_clean, fs=1000.0)
    # plot_lfp(lfp_clean, fs=1000.0, cfg=cfg)
    freqs, times, Sxx_db = spectrogram(lfp_clean, fs=1000.0)
    plot_spectrogram_mua(grouped_feats["logMUA"], times, freqs, Sxx_db, t_ds, cfg)

