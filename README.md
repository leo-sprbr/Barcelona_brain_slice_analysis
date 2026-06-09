# Brain Slice Magnetoelectric Stimulation Analysis Pipeline

A modular Python pipeline for analysing LFP recordings from brain slice experiments,
focusing on Up/Down state dynamics under magnetoelectric (ME) stimulation.

---

## Scientific Background

This pipeline implements the analysis described in:
> Guillamon-Vivancos et al. (2022), *eLife* — https://pubmed.ncbi.nlm.nih.gov/33011419/

The core idea is that slow oscillations (SO, ~0.1–1 Hz) in cortical brain slices
alternate between two network states:
- **Up states**: periods of sustained, high-frequency network activity
- **Down states**: periods of near-silence

These states are detected by combining three LFP-derived signals into a single
1D score via PCA, then thresholding its bimodal distribution.

**This pipeline's focus:** How does magnetoelectric stimulation change Up/Down
state properties (duration, frequency, firing rate) relative to baseline, within
the same slice?

---

## Project Structure

```
my_brain_slice_analysis/
├── README.md               ← You are here
├── requirements.txt        ← Python dependencies (pip)
├── environment.yml         ← Conda environment (alternative)
├── config.yaml             ← ALL analysis parameters live here
│
├── data/
│   ├── raw/                ← .smr files go here (gitignored)
│   └── processed/          ← .h5 files go here (gitignored)
│
├── src/                    ← Core library (importable modules)
│   ├── __init__.py
│   ├── config.py           ← Loads and validates config.yaml
│   ├── data_io.py          ← Reading/writing .h5 files, metadata
│   ├── preprocessing.py    ← Signal filtering, feature extraction (SO, gamma, MUA)
│   ├── visualization.py    ← All plotting functions
│   ├── utils.py            ← Shared helpers (binning, stats, Gaussian smoothing)
│   │
│   └── core/               ← Analysis logic (the science)
│       ├── __init__.py
│       ├── experiment.py   ← Experiment class: loads one slice, runs all analyses
│       ├── up_down.py      ← Up/Down state detection via PCA thresholding
│       ├── firing_rates.py ← Firing rate extraction & transition alignment
│       └── metrics.py      ← Duration stats, SO frequency, delta metrics, stats tests
│
├── notebooks/              ← Jupyter notebooks for exploration & figures
│   ├── 01_exploratory.ipynb
│   ├── 02_up_down_analysis.ipynb
│   └── 03_figure_making.ipynb
│
├── scripts/                ← Runnable from the command line
│   ├── convert_smr_to_h5.py    ← Converts raw .smr → .h5
│   ├── run_full_analysis.py    ← Runs the full pipeline on one slice
│   └── batch_process_slices.py ← Loops over all slices in processed/
│
└── tests/                  ← Unit tests (pytest)
    ├── test_preprocessing.py
    ├── test_up_down.py
    └── test_metrics.py
```

---

## Installation

### Option A — pip (recommended for most users)

```bash
# Clone / download the project, then:
cd my_brain_slice_analysis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate brain_slice
```

---

## Quick Start

### 1. Convert raw data

Drop your `.smr` files into `data/raw/`, then run:

```bash
python scripts/convert_smr_to_h5.py
```

This reads paths from `config.yaml` and writes `.h5` files to `data/processed/`.

### 2. Run a single slice

```bash
python scripts/run_full_analysis.py --slice data/processed/my_slice.h5
```

### 3. Batch process all slices

```bash
python scripts/batch_process_slices.py
```

Results (CSVs + figures) are written to the path defined in `config.yaml` under
`paths.results_dir`.

---

## Configuration

**All tunable parameters live in `config.yaml`.**
You should never need to edit source files to change filter cutoffs, thresholds,
or file paths — just edit the YAML.

Key sections:
- `paths` — where to read/write data and results
- `preprocessing` — filter bands, window sizes
- `up_down` — PCA detection parameters and thresholding strategy
- `firing_rates` — smoothing kernel, alignment window
- `statistics` — alpha level, correction method

See `config.yaml` for full documentation of each parameter.

---

## How the Analysis Works (step by step)

```
Raw LFP (samples × channels)
        │
        ▼
[preprocessing.py]
  ├── SO signal      : bandpass 0.1–1.5 Hz, take slow deflection
  ├── Gamma envelope : bandpass 30–80 Hz, compute variance in sliding window
  └── logMUA         : bandpass 200–1500 Hz, power in 5ms windows, log-scaled
        │
        ▼ z-score each signal, stack into (samples × 3) matrix
        │
[up_down.py]
  └── PCA → PC1 (bimodal) → KDE threshold → Up/Down state epochs
        │
        ▼
[firing_rates.py]
  ├── Mean logMUA per Up/Down epoch
  └── Transition-aligned firing rate traces (Gaussian smoothed)
        │
        ▼
[metrics.py]
  ├── Up/Down durations, SO frequency
  ├── Outlier exclusion (95% / 99.9% CI per condition)
  ├── Delta duration  (stimulation − baseline)
  ├── Delta firing rate (relative change vs baseline)
  └── Statistical tests:
       ├── Mann-Whitney U (independent, e.g. baseline vs sham/control)
       └── Paired Wilcoxon  (within-slice: baseline vs stim / post-stim)
```

---

## Adapting the Pipeline

The pipeline is intentionally modular. Here are common customisation scenarios:

| Goal | What to change |
|------|---------------|
| Change filter bands | Edit `config.yaml` → `preprocessing.*_band` |
| Use a different channel | `config.yaml` → `preprocessing.channel` |
| Add a new metric | Add a function to `src/core/metrics.py`; call it from `experiment.py` |
| Add a new condition | Update the `conditions` list in `config.yaml`; `metrics.py` uses it dynamically |
| Change detection threshold | `config.yaml` → `up_down.threshold_method` (`"kde"` or `"manual"`) |
| Add a new plot | Add a function to `src/visualization.py`; call it from a notebook or script |
| Add a new statistical test | Add a function to `src/utils.py`; use it in `metrics.py` |

Each module has a **"Extension points"** comment block at the bottom listing
exactly what was designed to be swapped or extended.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dependencies

See `requirements.txt` for the full pinned list. Key packages:
- `neo` — reading Spike2 `.smr` files
- `h5py` — reading/writing `.h5` files
- `scipy` — filtering, PCA, statistical tests
- `scikit-learn` — PCA (sklearn wrapper)
- `numpy`, `pandas` — numerical computing and data frames
- `matplotlib`, `seaborn` — plotting
- `pyyaml` — configuration loading
- `tqdm` — progress bars

---

## Citation

If you use this pipeline, please cite the detection method:
> Tort-Colet et al. (2021). *Attractor competition enriches cortical dynamics during awakening from anesthesia.* Cell Systems. https://pubmed.ncbi.nlm.nih.gov/33011419/
