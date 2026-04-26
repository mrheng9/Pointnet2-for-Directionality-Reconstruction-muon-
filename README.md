# PointNet++ for Particle Identification (PID 3-class) — Tutorial

## Overview
JUNO observes atmospheric neutrino interactions. This repo implements a **3-class PID** classifier:
- `nc`  → 0
- `Vmu` → 1
- `Ve`  → 2

Pipeline:
1. **Train** on train split and monitor val split (no test leakage)
2. **Evaluate** on the held-out test split (same split as training)
3. **Plot** confusion matrices + ROC/PR curves (from saved test predictions)

---

## Environment

Recommended (conda):
```bash
conda create -n pointnet2p python=3.8 -y
conda activate pointnet2p

conda install numpy scikit-learn matplotlib tqdm pandas -y
conda install pytorch -c pytorch -y
```

---

## Data Convention (PID)

Expected by `data_utils/PMTLoader.py`:
- `load_pid_dataset(pid_dataset_dir)` returns:
  - `points`: `float` array shaped `[N, P, 3]` with channel order **[nPE, slope, FHT]**
  - `labels`: `int` array shaped `[N]` with values in `{0,1,2}`

### Feature transform (`--feature_mode`)
Implemented in `train.py`:
- `normal`: `[nPE, FHT, slope]`
- `divide`: `[nPE, FHT, slope/(nPE+eps)]`
- `log`: `[nPE, FHT, log(slope+eps)]`
- `dlog`: `[log(nPE+eps), FHT, log(slope+eps)]`

### Coordinates option (`--coordinates`)
- If `--coordinates` is ON:
  - input becomes `[N, P, 6] = [xyz(3), features(3)]`
  - xyz is loaded from a **single template** and broadcast to all events:
    - default: `/disk_pool1/houyh/coords/norm_coords_single.npy` with shape `(17612, 3)`
    - with `--hamamatsu`: `/disk_pool1/houyh/coords/norm_coords_hama.npy` with shape `(4997, 3)`
- If `--coordinates` is OFF:
  - input is `[N, P, 3]` (features only), treated as pseudo-xyz by the network

### Hamamatsu-only mode (`--hamamatsu`)
If `--hamamatsu` is ON, the pipeline uses:
- **features** from: `pid_dataset_dir/pid_points_hama.npy` (shape `[N, 4997, 3]`)
- **coords** from: `/disk_pool1/houyh/coords/norm_coords_hama.npy` (shape `(4997, 3)`)

If `--hamamatsu` is OFF, it uses the default full-PMT dataset and coords template `(17612,3)`.

---

## Reproducible splitting + subsampling (important)

This repo supports downsampling the dataset using `--use_frac` (e.g. 0.3 = keep ~30% per class, stratified).
To keep **evaluation consistent** with training, the training split file stores:
- `train_idx`, `val_idx`, `test_idx`
- `subsample_idx` (the exact subsample mapping)

So evaluation will reuse the same subsample and split by loading `splits.npz`.

---

## Output files in `--log_dir`

After training + evaluation:
- `best_pointnet_regression_model.pth` — best checkpoint (lowest **val loss**)
- `learning_curve_loss.png` — train/val loss vs epoch (log y-axis)
- `learning_curve_acc.png` — train/val accuracy vs epoch
- `splits.npz` — split indices + `subsample_idx` (for reproducibility)
- `scalers.pkl` — feature normalization parameters fitted on **train only**
- `train_meta.json` — run configuration and split statistics (includes `hamamatsu/coordinates/feature_mode/use_frac/eps`)
- `predictions_test.npz` — evaluation outputs: `pred_class`, `true_class`, `prob`, `class_names`
- `metrics_test.json` — test accuracy + config snapshot
- `plots/` (or other plot dir) — confusion matrices, ROC/PR curves, Fig.9-style efficiency ROC curves

---

## Training

Minimal:
```bash
nohup python train.py --coordinates --hamamatsu > pid_train.log 2>&1 &
```

Example (Hamamatsu-only + xyz coords + dlog):
```bash
nohup python train.py --hamamatsu --coordinates --feature_mode dlog --use_frac 0.3 > pid_train_hama.log 2>&1 &
```

Key arguments:
- `--gpu` GPU id
- `--log_dir` experiment directory
- `--epoch` number of epochs
- `--batch_size` batch size
- `--use_frac` fraction of the full dataset to use (default `0.1`; set to `1.0` for full)
- `--test_size` test fraction (default `0.2`)
- `--val_size` val fraction from remaining trainval (default `0.2`)
- `--coordinates` include xyz coordinates (input C=6)
- `--feature_mode` `normal|divide|log|dlog`
- `--eps` numerical epsilon for divide/log
- `--hamamatsu` use Hamamatsu-only PMTs (P=4997)

---

## Evaluation (test inference)

Evaluation reads `train_meta.json` under `--log_dir` and overwrites runtime config for consistency:
- `pid_dataset_dir`
- `hamamatsu`
- `coordinates`
- `feature_mode`
- `use_frac`
- `eps`

It also loads `splits.npz` to ensure the **exact same test split** (and the same subsample).

Run (only keep basic flags):
```bash
nohup python evaluation.py --log_dir /path/to/experiment --gpu 0 --batch_size 64 > pid_eval.log 2>&1 &
```

Outputs:
- `predictions_test.npz`:
  - `pred_class` shape `[N_test]`
  - `true_class` shape `[N_test]`
  - `prob` shape `[N_test, 3]` (softmax probs)
  - `class_names`
- `metrics_test.json`

---

## Plotting

`plots.py` reads `predictions_test.npz` and generates:
- Confusion matrix (counts)
- Confusion matrix (efficiency normalization): **P(pred | true)** (row-normalized)
- OvR ROC curves + AUC for each class
- OvR PR curves + AP for each class
- Fig.9-style “efficiency ROC” curves:
  - y-axis: efficiency of target class
  - x-axis: efficiency of other two classes combined
  - AUC values displayed

Run (use defaults in plots.py), or specify explicitly:
```bash
python plots.py
```

---

## Notes / Troubleshooting

1. **If ROC/PR shows warnings about no positive/negative samples**
   - Your test set contains only one class. Check `evaluation.py` prints:
     `counts_test=[..., ..., ...]`
   - Ensure you are using the correct `--log_dir` with a valid `splits.npz` and `subsample_idx`.

2. **Coordinates file**
   - If `--coordinates` is ON, the coords template must exist and match your point count `P`.
   - Templates used:
     - `/disk_pool1/houyh/coords/norm_coords_single.npy` (P=17612)
     - `/disk_pool1/houyh/coords/norm_coords_hama.npy` (P=4997)

3. **Re-run requirement after changing split/subsample logic**
   - If you created `splits.npz` before adding `subsample_idx`, re-train once to regenerate it.