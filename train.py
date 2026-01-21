import os
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from data_utils.PMTLoader import CustomDataset, load_pid_dataset
from data_utils.PMTLoader import jitter_point_cloud, random_point_dropout
# from models.pointnet_regression import get_model,get_loss
from models.pointnet_regression_ssg import get_model,get_loss
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt
COORDS_SINGLE_PATH = "/disk_pool1/houyh/coords/norm_coords_single.npy"   # (17612,3)
COORDS_HAMA_PATH   = "/disk_pool1/houyh/coords/norm_coords_hama.npy"

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='3', help='specify gpu device')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=40, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay in training')
    
    parser.add_argument("--pid_dataset_dir",type=str,default="/disk_pool1/houyh/data/J23_J25_7_2/pid_dataset")
    parser.add_argument("--coordinates", action="store_true", help="use xyz coords as first 3 channels and use remaining as features")
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="normal",
        choices=["normal", "divide", "log", "dlog"],
        help="feature transform mode for PID dataset",
    )

    # NEW: whether to use Hamamatsu-only PMT points (N,4997,3)
    parser.add_argument("--hamamatsu",action="store_true",help="use Hamamatsu-only PMT points: pid_points_hama.npy")
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--use_frac",type=float,default=0.1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1e-6, help="numerical epsilon for divide/log")

    parser.add_argument('--log_dir', type=str, default='/home/houyh/Pointnet2-for-Directionality-Reconstruction-muon-/experiments/test_feat_normal', help='experiment root')
    return parser.parse_args()

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def fit_feature_scalers(points_train, feat_start=3):
    """
    Fit one StandardScaler per feature channel (over all events and PMTs).
    Returns list of (mean_, scale_) tuples to keep it pickleable and stable.
    """
    scalers = []
    for i in range(feat_start, points_train.shape[-1]):
        sc = StandardScaler()
        sc.fit(points_train[:, :, i])
        scalers.append((sc.mean_.copy(), sc.scale_.copy()))
    return scalers

def apply_feature_scalers(points, scalers, feat_start=3):
    """
    Apply saved (mean, scale) to points in-place.
    """
    for k, i in enumerate(range(feat_start, points.shape[-1])):
        mean, scale = scalers[k]
        # StandardScaler transform: (x - mean) / scale
        points[:, :, i] = (points[:, :, i] - mean) / (scale + 1e-12)
    return points

def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))

def _transform_pid_features(points: np.ndarray, mode: str, eps: float = 1e-6) -> np.ndarray:
    """
    points: [N,P,3] with assumed order: [nPE, slope, FHT]
    returns transformed points [N,P,3]
    Modes:
      normal: nPE, FHT, slope
      divide: nPE, FHT, slope/nPE
      log:    nPE, FHT, log(slope)
      dlog:   log(nPE), FHT, log(slope)
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points [N,P,3], got {points.shape}")

    nPE = points[:, :, 0].astype(np.float32, copy=False)
    slope = points[:, :, 1].astype(np.float32, copy=False)
    fht = points[:, :, 2].astype(np.float32, copy=False)

    if mode == "normal":
        out = np.stack([nPE, fht, slope], axis=-1)

    elif mode == "divide":
        out = np.stack([nPE, fht, slope / (nPE + eps)], axis=-1)

    elif mode == "log":
        # robust for slope<=0, avoids NaNs
        out = np.stack([nPE, fht, _signed_log1p(slope)], axis=-1)

    elif mode == "dlog":
        # keep nPE in log-space; clamp to avoid log of non-positive
        out = np.stack([np.log(np.maximum(nPE, eps)), fht, _signed_log1p(slope)], axis=-1)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    out = out.astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise ValueError(f"Non-finite values produced by feature_mode={mode}. Check raw points ranges.")
    return out

def _stratified_subsample(points: np.ndarray, labels: np.ndarray, frac: float, seed: int):
    if frac >= 1.0:
        keep_idx = np.arange(labels.shape[0])
        return points, labels, keep_idx
    if frac <= 0.0:
        raise ValueError(f"use_frac must be in (0,1], got {frac}")

    rng = np.random.RandomState(seed)
    keep_idx = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        k = max(1, int(round(idx.size * frac)))
        sel = rng.choice(idx, size=k, replace=False)
        keep_idx.append(sel)
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    return points[keep_idx], labels[keep_idx], keep_idx

def _load_coords_tiled(n: int, p: int, hamamatsu: bool = False) -> np.ndarray:
    """
    Load a single (P,3) coords template and broadcast to (N,P,3).
    - hamamatsu=False -> COORDS_SINGLE_PATH, expects (17612,3)
    - hamamatsu=True  -> COORDS_HAMA_PATH,   expects (4997,3)
    """
    coords_path = COORDS_HAMA_PATH if hamamatsu else COORDS_SINGLE_PATH
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Missing coords file: {coords_path}")

    coords = np.load(coords_path, mmap_mode="r")
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be [P,3], got {coords.shape} from {coords_path}")
    if coords.shape[0] != p:
        raise ValueError(f"P mismatch: coords P={coords.shape[0]} vs feats P={p} (file={coords_path})")

    tmpl = np.asarray(coords, dtype=np.float32)  # [P,3]
    return np.broadcast_to(tmpl[None, :, :], (n, p, 3)).copy()

def load_data_with_splits(args, save_splits_to=None, load_splits_from=None):
    """
    PID dataset version.
    If args.coordinates=True:
        input becomes [N,P,6] = [xyz(3), transformed_features(3)]
    else:
        input is [N,P,3] = transformed_features(3) (used as pseudo-xyz)
    """
    points, labels = load_pid_dataset(getattr(args, "pid_dataset_dir"))

    # optionally replace points with Hamamatsu-only points
    if bool(getattr(args, "hamamatsu", False)):
        hama_path = os.path.join(getattr(args, "pid_dataset_dir"), "pid_points_hama.npy")
        if not os.path.exists(hama_path):
            raise FileNotFoundError(f"--hamamatsu set but missing file: {hama_path}")
        points_hama = np.load(hama_path)
        if points_hama.ndim != 3 or points_hama.shape[-1] != 3:
            raise ValueError(f"pid_points_hama.npy must be [N,4997,3] (or [N,P,3]), got {points_hama.shape}")
        if points_hama.shape[0] != labels.shape[0]:
            raise ValueError(
                f"N mismatch: pid_points_hama N={points_hama.shape[0]} != labels N={labels.shape[0]}"
            )
        points = points_hama

    # --- reuse the exact same subsample if splits.npz provides it ---
    subsample_idx = None
    split_npz = None
    if load_splits_from and os.path.exists(load_splits_from):
        split_npz = np.load(load_splits_from)
        if "subsample_idx" in split_npz.files:
            subsample_idx = split_npz["subsample_idx"].astype(np.int64)
            points = points[subsample_idx]
            labels = labels[subsample_idx]
            
    if subsample_idx is None:
        points, labels, subsample_idx = _stratified_subsample(
            points,
            labels,
            frac=float(getattr(args, "use_frac", 1.0)),
            seed=int(getattr(args, "seed", 42)),
        )


    feats = _transform_pid_features(
        points,
        mode=getattr(args, "feature_mode", "normal"),
        eps=getattr(args, "eps", 1e-6),
    )  # [N,P,3]

    if bool(getattr(args, "coordinates", False)):
        coords = _load_coords_tiled(
            n=feats.shape[0],
            p=feats.shape[1],
            hamamatsu=bool(getattr(args, "hamamatsu", False)),
        )
        points = np.concatenate([coords, feats], axis=-1).astype(np.float32, copy=False)
        feat_start = 3
    else:
        points = feats.astype(np.float32, copy=False)
        feat_start = 0

    n = int(points.shape[0])

    # splits (use stratify so 3 classes stay balanced)
    if split_npz is not None:
        train_idx = split_npz["train_idx"]
        val_idx = split_npz["val_idx"]
        test_idx = split_npz["test_idx"]
    else:
        all_idx = np.arange(int(feats.shape[0]))
        trainval_idx, test_idx = train_test_split(
            all_idx,
            test_size=getattr(args, "test_size", 0.2),
            random_state=getattr(args, "seed", 42),
            shuffle=True,
            stratify=labels,
        )
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=getattr(args, "val_size", 0.2),
            random_state=getattr(args, "seed", 42),
            shuffle=True,
            stratify=labels[trainval_idx],
        )

        if save_splits_to:
            os.makedirs(os.path.dirname(save_splits_to), exist_ok=True)
            np.savez(
                save_splits_to,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                subsample_idx=subsample_idx,
            )

    points_train, labels_train = points[train_idx], labels[train_idx]
    points_val, labels_val = points[val_idx], labels[val_idx]
    points_test, labels_test = points[test_idx], labels[test_idx]

    # Standardize features only (if coordinates=True) else standardize all channels
    scalers = fit_feature_scalers(points_train, feat_start=feat_start)
    apply_feature_scalers(points_train, scalers, feat_start=feat_start)
    apply_feature_scalers(points_val, scalers, feat_start=feat_start)
    apply_feature_scalers(points_test, scalers, feat_start=feat_start)

    # print split sizes + counts at load time (train/eval both call this)
    counts_all = np.bincount(labels, minlength=3)
    counts_train = np.bincount(labels_train, minlength=3)
    counts_val = np.bincount(labels_val, minlength=3)
    counts_test = np.bincount(labels_test, minlength=3)

    print(
        f"[DATA] N_total_used={n} (P={points.shape[1]}, C={points.shape[2]}) "
        f"| counts_all={counts_all.tolist()} "
        f"| train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)} "
        f"| counts_train={counts_train.tolist()} counts_val={counts_val.tolist()} counts_test={counts_test.tolist()} "
        f"| use_frac={float(getattr(args,'use_frac',1.0))} "
        f"| coordinates={bool(getattr(args,'coordinates',False))} feature_mode={getattr(args,'feature_mode','normal')} "
        f"| hamamatsu={bool(getattr(args,'hamamatsu',False))}"
    )

    split_dict = {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "counts_train": counts_train.tolist(),
        "counts_val": counts_val.tolist(),
        "counts_test": counts_test.tolist(),
        "counts_all_used": counts_all.tolist(),
        "feature_mode": getattr(args, "feature_mode", "normal"),
        "coordinates": bool(getattr(args, "coordinates", False)),
        "feat_start": int(feat_start),
        "n_total_used": int(n),
        "hamamatsu": bool(getattr(args, "hamamatsu", False)),
    }
    return points_train, points_val, points_test, labels_train, labels_val, labels_test, scalers, split_dict

CLASS_NAMES = ["nc", "Vmu", "Ve"]

def draw_learning_curve(train_losses, val_losses, train_accs=None, val_accs=None, log_dir=None, epochs=None):
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.legend(fontsize=16)
    plt.title('Learning Curve (Loss)', fontsize=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.yscale('log')
    plt.savefig(os.path.join(log_dir, 'learning_curve_loss.png') if log_dir else 'learning_curve_loss.png', dpi=300)
    plt.close()

    if train_accs is not None and val_accs is not None:
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, epochs + 1), train_accs, label='Train Acc', linewidth=2)
        plt.plot(range(1, epochs + 1), val_accs, label='Val Acc', linewidth=2)
        plt.xlabel('Epoch', fontsize=25)
        plt.ylabel('Accuracy', fontsize=25)
        plt.legend(fontsize=16)
        plt.title('Learning Curve (Accuracy)', fontsize=30)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(log_dir, 'learning_curve_acc.png') if log_dir else 'learning_curve_acc.png', dpi=300)
        plt.close()


def train(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, target in tqdm(train_loader, desc='Training'):
        points = points.float().to(device)
        target = target.long().to(device)  # PID: class indices

        optimizer.zero_grad()
        logits, _ = model(points)
        loss = criterion(logits, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        pred_cls = torch.argmax(logits, dim=1)
        correct += (pred_cls == target).sum().item()
        total += target.numel()

    avg_loss = total_loss / max(1, len(train_loader))
    acc = correct / max(1, total)
    return avg_loss, acc

def evaluate(model, criterion, loader, device, desc="Eval"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for points, target in tqdm(loader, desc=desc):
            points = points.float().to(device)
            target = target.long().to(device)

            logits, _ = model(points)
            loss = criterion(logits, target)
            total_loss += loss.item()

            pred_cls = torch.argmax(logits, dim=1)
            correct += (pred_cls == target).sum().item()
            total += target.numel()

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc

def main(args):
    logger = setup_logging()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)

    splits_path = os.path.join(args.log_dir, "splits.npz") if args.log_dir else None
    scalers_path = os.path.join(args.log_dir, "scalers.pkl") if args.log_dir else None
    meta_path = os.path.join(args.log_dir, "train_meta.json") if args.log_dir else None

    points_train, points_val, points_test, labels_train, labels_val, labels_test, scalers, split_meta = load_data_with_splits(
        args,
        save_splits_to=splits_path,
        load_splits_from=None,
    )

    feat_dim = int(points_train.shape[-1] - 3) if bool(args.coordinates) else 0
    if scalers_path:
        with open(scalers_path, "wb") as f:
            pickle.dump({"feat_start": int(split_meta["feat_start"]), "scalers": scalers}, f)

    if meta_path:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task": "pid_3class",
                    "class_names": CLASS_NAMES,
                    "class_mapping": {"nc": 0, "Vmu": 1, "Ve": 2},

                    # dataset + preprocessing
                    "pid_dataset_dir": getattr(args, "pid_dataset_dir", None),
                    "hamamatsu": bool(getattr(args, "hamamatsu", False)),
                    "coordinates": bool(getattr(args, "coordinates", False)),
                    "feature_mode": getattr(args, "feature_mode", "normal"),
                    "eps": float(getattr(args, "eps", 1e-6)),
                    "input_channels": int(points_train.shape[-1]),   # C
                    "num_points": int(points_train.shape[1]),        # P

                    # splits + counts
                    "split_meta": split_meta,
                    "use_frac": float(getattr(args, "use_frac", 1.0)),

                    # training hyperparams
                    "batch_size": int(args.batch_size),
                    "epoch": int(args.epoch),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "seed": int(getattr(args, "seed", 42)),

                    # model architecture info
                    "model": {
                        "name": "pointnet2_ssg_pid",
                        "num_class": 3,
                        "coordinates": bool(getattr(args, "coordinates", False)),
                        "feat_dim": int(feat_dim),
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


    model = get_model(3, coordinates=bool(args.coordinates), feat_dim=int(feat_dim)).to(device).float()

    criterion = get_loss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    train_loader = DataLoader(CustomDataset(points_train, labels_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CustomDataset(points_val, labels_val), batch_size=args.batch_size, shuffle=False)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(args.epoch):
        train_loss, train_acc = train(model, criterion, optimizer, train_loader, device)
        val_loss, val_acc = evaluate(model, criterion, val_loader, device, desc="Validation")

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        logger.info(
            f'Epoch [{epoch+1}/{args.epoch}] '
            f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}'
        )

    if args.log_dir is not None and best_model_state is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        torch.save(best_model_state, os.path.join(args.log_dir, 'best_pointnet_regression_model.pth'))

    draw_learning_curve(train_losses, val_losses, train_accs=train_accs, val_accs=val_accs, log_dir=args.log_dir, epochs=args.epoch)

if __name__ == '__main__':
    args = parse_args()
    main(args)