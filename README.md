# PointNet++ for directionality reconstruction — Tutorial
This tutorial based on the server @10.102.33.217  （GPU04）
## Overview
The Jiangmen Underground Neutrino Observatory **(JUNO)** is a large liquid-scintillator detector designed to determine the neutrino mass ordering and precisely measure oscillation parameters.   
![](https://github.com/user-attachments/assets/e33b6881-fad5-4585-a326-d289810b430b)
Beyond reactor neutrinos, JUNO also observes atmospheric neutrinos whose charged-current interactions produce muons. This project focuses on reconstructing the muon direction from atmospheric neutrino interactions using a PointNet++ regression model on PMT-derived point-cloud features.
 
## Repository Structure
- `train.py` — training script (train/val split, saves best model + learning curve)
  - `models/pointnet_regression_ssg.py` — PointNet MSG-based regression model
  - `models/pointnet_regression_utils.py` — set abstraction layers and MSG ops
- `evaluation.py` — **test-set inference only**, saves predictions to `.npz`
- `plots.py` — reads saved predictions and generates plots
- `data_utils/`
  - `PMTLoader.py` — `CustomDataset`, and data stacking helpers
- `README.md` — brief usage notes

## Environment 

Dependencies:
- python=3.8
- numpy
- pytorch
- matplotlib
- tqdm
- scikit-learn
- pandas
- git

Setup :
```powershell
# Create and activate env
conda create -n pointnet2p python=3.8 -y
conda activate pointnet2p

# Install packages
conda install numpy scikit-learn matplotlib tqdm pandas git -y
conda install pytorch -c pytorch -y
```

## Data and Features
Implement or point the loader to your dataset through `data_utils/PMTLoader.py`.

Expected conventions:
- Inputs: stacked point features shaped `[N, P, C]`
  - N = samples, P = points per sample, C = input channels (e.g., coordinates + features)
- Targets: regression vectors shaped `[N, D]` (commonly 3D; labels are normalized to unit vectors in `train.py`)

Feature origin (PMT waveforms):
- All feature channels C are extracted from PMT charge/time waveforms measured or reconstructed from the detector.
- Typical examples:
  - fht: first-hit time of the PMT pulse
  - slope: rising-edge slope proxy
  - peak / peaktime: pulse peak amplitude and its time
  - timemax: time of maximum sample
  - nperatio5: charge ratio within a short window (e.g., 5 ns) to total charge
  - npe: number of photoelectrons (charge proxy)

Data sources (argument: --data_source):
- det — Features extracted from detector-level simulation. 
- elec — Features after electronics simulation.
- cnn — Features reconstructed by a CNN from waveforms. 
- rawnet — Features reconstructed by a RawNet model. 

## Model Overview
File: `models/pointnet_regression_ssg.py`
- Extracts hierarchical point features using:
  - `PointNetSetAbstractionMsg` (multi-scale grouping)
  - `PointNetSetAbstraction` (global stage)
- Aggregates multi-scale features and regresses to a continuous target.
- Configurable input channels (`in_channel`) to match your data (C above).

Core building blocks in `models/pointnet_regression_utils.py`:
- `PointNetSetAbstraction`
- `PointNetSetAbstractionMsg`

## Workflow (Train → Evaluate → Plot)

This repo now uses a clean **three-stage pipeline**:
1. **Training** uses **train** split and monitors **val** split (no test leakage).
2. **Evaluation** runs inference on the held-out **test** split only and saves raw predictions.
3. **Plots** reads evaluation outputs and generates figures.

### Output files in `--log_dir`
After training and evaluation, you should see:
- `best_pointnet_regression_model.pth` — best checkpoint (selected by lowest **val loss**)
- `learning_curve.png` — train/val loss vs epoch (log y-axis)
- `splits.npz` — fixed indices for `train_idx/val_idx/test_idx` (for reproducibility)
- `scalers.pkl` — feature normalization parameters fitted **on train only**
- `train_meta.json` — run configuration and split sizes
- `predictions_test.npz` — evaluation outputs (`y_pred`, `y_true`)
- (optional) plot images produced by `plots.py`

> Note: `splits.npz` and `scalers.pkl` are used to keep evaluation consistent with training (same split + same normalization).

---

## Training
Choose the one you prefer. Here use `nohup` as a demonstration.  
**(note that the --data_source parameter is required)**

```bash
nohup python train.py --data_source cnn --log_dir experiments/test > cnn_train.log 2>&1 &
```

Key arguments:
- `--gpu` GPU id (default: `2`)
- `--epoch` number of epochs
- `--batch_size` batch size
- `--test_size` fraction of the full dataset used for test (default `0.2`)
- `--val_size` fraction of the remaining trainval set used for validation (default `0.2`)
- `--seed` random seed for splitting

### Plots saved by `train.py`
- `learning_curve.png` — Train/Val loss vs epoch (log y-axis)

---

## Evaluation (Test Inference)
Run inference on the **test split** using the checkpoint saved by training.

### Recommended (no need to pass `--data_source`)
`evaluation.py` will automatically read `data_source` from `train_meta.json` under `--log_dir`, so you only need to provide the experiment directory:

```bash
nohup python evaluation.py --log_dir experiments/test > eval.log 2>&1 &
```

### Alternative (if `train_meta.json` is missing)
If `train_meta.json` does not exist in `--log_dir`, you must pass `--data_source` explicitly:

```bash
nohup python evaluation.py --data_source cnn --log_dir experiments/test > eval.log 2>&1 &
```

Outputs:
- `experiments/test/predictions_test.npz` containing:
  - `y_pred`: predicted direction vectors, shape `[N_test, 3]`
  - `y_true`: ground truth direction vectors, shape `[N_test, 3]`

Notes:
- Evaluation uses `splits.npz` to ensure the **same test split** as training.

---

## Plotting
Generate figures from `predictions_test.npz`:

```bash
python plots.py 
```

Outputs (saved under `--log_dir` passed to plots.py):
- `test_performance.png` — scatter of predicted vs true θ with y=x reference
- `error_distribution.png` — histogram of (pred − true) θ errors (deg)
- `angle_distribution.png` — opening angle α PDF (deg) + 68% quantile marker

---

## Reference Results
1 GeV muon direction reconstruction — α angle resolution (68th percentile).  
Different models vs. different data sources (best resolution per setting).  
Features used: ["fht", "slope", "peak", "timemax", "nperatio5", "npe"]

![1 GeV Muon α(°) 68th percentile — model/data comparison](https://github.com/user-attachments/assets/8f339837-5deb-46a4-af47-a96ad0c726ae)

Notes:
- α: opening angle between predicted and true directions (degrees).
- Use this plot to check if your run matches or improves the 68% quantile for your chosen data_source (det/elec/cnn/rawnet) and feature set.