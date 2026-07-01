import sys, os

from src.visualization import plot_multichannel_feature_group

# from src.core.metrics import compute_firing_rates
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from src.config import load_config
from src.data_io import load_h5_time_window, list_single_h5_file
# from src.preprocessing import select_channel_raw, _extract_raw_log_mua, extract_features_all_channels
from src.visualization import plot_multichannel_states_overlay, plot_multichannel_raw_lfp, plot_raw_lfp_time_window, plot_multichannel_feature_group, plot_pc1_distribution, plot_states_overlay, plot_spectrogram, plot_spectrogram_mua
# from src.core.up_down import detect_states
# from src.utils import spectrogram
from src.core.activity_analysis import Activity_analysis

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

    # h5_files   = list_h5_files(cfg['paths']['condition_dir'])
    # conditions = cfg['conditions']
    # experiments = {}   # {(slice_id, condition): Experiment}
    # summary_rows = []

    import os
    from src.data_io import get_slice_id
    from src.utils import get_electrode_display_order

    # Nest all figures under a subfolder named after the recording
    slice_name = get_slice_id(H5_PATH)   # e.g. "20260414_SLICE3_CONTROL_Nathalia"
    base_fig_dir = cfg["paths"]["figures_dir"]
    cfg["paths"]["figures_dir"] = os.path.join(base_fig_dir, slice_name)
    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    print(f"Figures will be saved to: {cfg['paths']['figures_dir']}")

    condition = 'baseline'
    exp = Activity_analysis(H5_PATH, condition=condition, cfg=cfg)
    exp.run()
    
    # --- Compute display order from actual number of channels ---
    n_channels = exp.lfp.shape[1] if exp.lfp.ndim > 1 else 1
    order = get_electrode_display_order(n_channels)
    print(f"Electrode display order ({n_channels} ch): {order}")

    # --- All plots use the same order ---
    fig_raw   = plot_multichannel_raw_lfp(exp.lfp, exp.fs, cfg, band=(0.1, 300), channel_order=order)
    fig_so    = plot_multichannel_feature_group(exp.so_matrix,    exp.t_ds, "Slow Oscillations (SO)", cfg, channel_order=order)
    fig_gamma = plot_multichannel_feature_group(exp.gamma_matrix, exp.t_ds, "Gamma Envelope",         cfg, channel_order=order)
    fig_mua   = plot_multichannel_feature_group(exp.mua_matrix,   exp.t_ds, "logMUA Data",            cfg, channel_order=order)
    fig_states, _ = plot_multichannel_states_overlay(exp.detected_states, exp.t_ds, cfg, channel_order=order)