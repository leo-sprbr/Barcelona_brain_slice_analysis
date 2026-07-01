import sys, os

# from src.core.metrics import compute_firing_rates
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from src.config import load_config
from src.data_io import load_h5_time_window, list_single_h5_file
from src.preprocessing import select_channel_raw, _extract_raw_log_mua, extract_features_all_channels, highpass_filter
from src.visualization import plot_multichannel_states_overlay, plot_raw_lfp_time_window, plot_multichannel_feature_group, plot_pc1_distribution, plot_states_overlay, plot_spectrogram, plot_spectrogram_mua
from src.core.up_down import detect_states
from src.utils import spectrogram
from huimin_plot import print_channel_metrics_summary

from src.core.metrics import print_channel_metrics_summary

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
    lfp_signal = highpass_filter(lfp_signal, up, 0.1)
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
    H5_PATH = list_single_h5_file(cfg)
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
    
    # # 7. Render everything to the desktop interface
    # # plt.show()
    # lfp_clean = extract_lfp_from_raw(lfp, fs_raw=meta['fs'], target_fs=1000.0)
    # lfp_clean = remove_line_noise(lfp_clean, fs=1000.0)
    # # plot_lfp(lfp_clean, fs=1000.0, cfg=cfg)
    # freqs, times, Sxx_db = spectrogram(lfp_clean, fs=1000.0)
    # plot_spectrogram_mua(grouped_feats["logMUA"], times, freqs, Sxx_db, t_ds, cfg)

