# Brain Slice Magnetoelectric Stimulation Analysis Pipeline


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
│   └── condition/          ← sliced trial conditions here (gitignored)
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
│   ├── 00_trial_definition.ipynb
│   ├── 01_exploratory.ipynb
│   ├── 02_up_down_analysis.ipynb
│   └── 03_figure_making.ipynb
│
├── scripts/                ← Runnable from the command line
│   ├── convert_smr_to_h5.py    ← Converts raw .smr → .h5
│   ├── run_full_analysis.py    ← Runs the full pipeline on one slice
│   └── batch_process_slices.py ← Loops over all slices in processed/

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
