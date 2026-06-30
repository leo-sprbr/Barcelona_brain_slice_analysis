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
from src.visualization import plot_multichannel_states_overlay, plot_raw_lfp_time_window, plot_multichannel_feature_group, plot_pc1_distribution, plot_states_overlay, plot_spectrogram, plot_spectrogram_mua
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


    condition = 'baseline'
    exp = Activity_analysis(H5_PATH, condition=condition, cfg=cfg)
    exp.run()

    # 2. Image 1: Plot all channels of Slow Oscillations (SO)
    fig_so = plot_multichannel_feature_group(exp.so_matrix, exp.t_ds, "Slow Oscillations (SO)", cfg)
    
    # 3. Image 2: Plot all channels of Gamma Envelope
    fig_gamma = plot_multichannel_feature_group(exp.gamma_matrix, exp.t_ds, "Gamma Envelope", cfg)
    
    # 4. Image 3: Plot all channels of logMUA
    fig_mua = plot_multichannel_feature_group(exp.mua_matrix, exp.t_ds, "logMUA Data", cfg)

    # 5. Image 4: Plot all channels of PC1 and up-down states overlay
    fig, axes = plot_multichannel_states_overlay(exp.detected_states, exp.t_ds, cfg)    