# PointNet++ for Directionality Reconstruction (PID 3-class) — Tutorial

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
  - xyz is loaded from: `COORDS_PATH = "/disk_pool1/houyh/coords/norm_coords"`
- If `--coordinates` is OFF:
  - input is `[N, P, 3]` (features only), treated as pseudo-xyz by the network

---

## Reproducible splitting + subsampling (important)

This repo supports downsampling the dataset using `--use_frac` (e.g. 0.3 = keep ~30% per class, stratified).
To keep **evaluation consistent** with training, the training split file now stores both:
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
- `train_meta.json` — run configuration and split statistics
- `predictions_test.npz` — evaluation outputs: `pred_class`, `true_class`, `prob`, `class_names`
- `metrics_test.json` — test accuracy + config snapshot
- `plots/` (or other plot dir) — confusion matrices, ROC/PR curves, Fig.9-style efficiency ROC curves

---

## Training

Minimal:
```bash
nohup python train.py --coordinates > pid.log 2>&1 & 
```

Common options:
```bash
# use 30% data (stratified), with xyz coords and dlog features
nohup python evaluation.py --coordinates --use_frac 0.3 > pid_eval.log 2>&1 & 
```

Key arguments:
- `--gpu` GPU id (default `2`)
- `--log_dir` experiment directory
- `--epoch` number of epochs
- `--batch_size` batch size
- `--use_frac` fraction of the full dataset to use (default `0.1`; set to `1.0` for full)
- `--test_size` test fraction (default `0.2`)
- `--val_size` val fraction from remaining trainval (default `0.2`)
- `--coordinates` include xyz coordinates (input C=6)
- `--feature_mode` `normal|divide|log|dlog`
- `--eps` numerical epsilon for divide/log

---

## Evaluation (test inference)

Evaluation reads `train_meta.json` under `--log_dir` and overwrites runtime args for consistency:
- `pid_dataset_dir`
- `coordinates`
- `feature_mode`
- `use_frac`
- `eps`

It also loads `splits.npz` to ensure the **exact same test split** (and the same subsample).

Run:
```bash
nohup python evaluation.py --coordinates > pid_eval.log 2>&1 & 
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
   - If `counts_test` is unbalanced or missing classes, ensure you are using the correct `--log_dir` with a valid `splits.npz` and `subsample_idx`.

2. **Coordinates file**
   - `COORDS_PATH` must exist and match your point count `P`.
   - You can inspect it with:
     ```bash
     python test.py
     ```

3. **Re-run requirement after changing split/subsample logic**
   - If you created `splits.npz` before adding `subsample_idx`, re-train once to regenerate it.

---