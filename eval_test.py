import os
import json
import argparse
import pickle
from typing import Tuple, Dict 
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils.PMTLoader import CustomDataset, load_eval_source_points, get_stacked_data
from models.pointnet_regression_ssg import get_model

# NEW: same coords paths as train.py
COORDS_SINGLE_PATH = "/disk_pool1/houyh/coords/norm_coords_single.npy"   # (17612,3)
COORDS_HAMA_PATH   = "/disk_pool1/houyh/coords/norm_coords_hama.npy"     # (4997,3)
COORDS_NNVT_PATH   = "/disk_pool1/houyh/coords/norm_coords_nnvt.npy"     # (12615,3) will be auto-generated if missing
PMT_TYPE_CSV_PATH = "/disk_pool1/houyh/data/PMTType_CD_LPMT.csv"

CLASS_NAMES = ["nc", "Vmu", "Ve"]
VMU_LABEL = 1


def parse_args():
    p = argparse.ArgumentParser("eval_test")
    p.add_argument("--use_cpu", action="store_true", default=False)
    p.add_argument("--gpu", type=str, default="3")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--log_dir",
        type=str,
        default="/disk_pool/houyh/Pointnet2-for-Directionality-Reconstruction-muon-/experiments/test_1.0_all",
        help="experiment dir produced by train.py (must contain train_meta.json & best_pointnet_regression_model.pth)",
    )
    p.add_argument(
        "--source",
        type=str,
        default="mc_muon_new",
        choices=["chimney", "FC", "data_muon", "mc_muon", "data_muon_new", "mc_muon_new"],
        help="which external dataset to run inference on",
    )
    return p.parse_args()


def _apply_meta_defaults(args):
    meta_path = os.path.join(args.log_dir, "train_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing train_meta.json at: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # nnvt is optional for backward compatibility
    required = ["coordinates", "feature_mode", "eps", "hamamatsu"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(f"train_meta.json missing keys: {missing}. Please re-train to regenerate meta.")

    args.coordinates = bool(meta["coordinates"])
    args.feature_mode = str(meta["feature_mode"])
    args.eps = float(meta["eps"])
    args.hamamatsu = bool(meta["hamamatsu"])
    args.nnvt = bool(meta.get("nnvt", False))  # NEW: from meta only, no new CLI hyperparam
    args.num_points = int(meta.get("num_points", 17612))
    return meta


def _load_pmt_indices_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (hamamatsu_idx, non_hamamatsu_idx) sorted.
    non_hamamatsu_idx includes anything not labeled 'Hamamatsu' in the CSV
    (e.g., NNVT, HighQENNVT, etc.) which matches your request.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    hama_idx = []
    non_idx = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header_seen = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            if not header_seen and parts[0].lower() == "index" and parts[1].lower() == "type":
                header_seen = True
                continue
            if parts[0].isdigit():
                i = int(parts[0])
                t = parts[1]
                if t == "Hamamatsu":
                    hama_idx.append(i)
                else:
                    non_idx.append(i)

    if not hama_idx:
        raise RuntimeError(f"No Hamamatsu indices found in: {csv_path}")
    if not non_idx:
        raise RuntimeError(f"No non-Hamamatsu indices found in: {csv_path}")

    return np.array(sorted(hama_idx), dtype=np.int64), np.array(sorted(non_idx), dtype=np.int64)


def _select_pmt_subset(points: np.ndarray, hamamatsu: bool, nnvt: bool, csv_path: str = PMT_TYPE_CSV_PATH) -> Tuple[np.ndarray, Dict]:
    """
    Apply PMT selection based on two flags:
      - hamamatsu=True -> keep Hamamatsu PMTs only
      - nnvt=True      -> keep non-Hamamatsu PMTs only
      - both False     -> keep all PMTs
    Disallow both True.
    """
    if points.ndim != 3:
        raise ValueError(f"points must be [N,P,C], got {points.shape}")

    if hamamatsu and nnvt:
        raise ValueError("Invalid config: both hamamatsu=True and nnvt=True. Choose only one.")

    n, p, c = points.shape
    if not (hamamatsu or nnvt):
        info = {"filter": "all", "csv": csv_path, "P_in": int(p), "P_out": int(p)}
        return points.astype(np.float32, copy=False), info

    hama_idx, non_idx = _load_pmt_indices_from_csv(csv_path)
    if hama_idx.max() >= p or non_idx.max() >= p:
        raise ValueError(
            f"PMT index out of range for current points: P={p}, "
            f"hama_max={int(hama_idx.max())}, non_hama_max={int(non_idx.max())}"
        )

    if hamamatsu:
        kept = points[:, hama_idx, :].astype(np.float32, copy=False)
        info = {"filter": "hamamatsu", "csv": csv_path, "P_in": int(p), "P_out": int(kept.shape[1])}
        return kept, info

    kept = points[:, non_idx, :].astype(np.float32, copy=False)
    info = {"filter": "non_hamamatsu", "csv": csv_path, "P_in": int(p), "P_out": int(kept.shape[1])}
    return kept, info


def _apply_saved_scalers(points: np.ndarray, scalers_path: str) -> np.ndarray:
    """
    Apply training scalers.pkl to eval points (in-place).
    scalers.pkl structure saved by train.py: {"feat_start": int, "scalers": [(mean, scale), ...]}
    """
    if not os.path.exists(scalers_path):
        raise FileNotFoundError(f"Missing scalers.pkl at: {scalers_path}")

    with open(scalers_path, "rb") as f:
        obj = pickle.load(f)

    feat_start = int(obj["feat_start"])
    scalers = obj["scalers"]
    n_scalers = len(scalers)

    if points.ndim != 3:
        raise ValueError(f"points must be [N,P,C], got {points.shape}")
    if feat_start < 0 or feat_start >= points.shape[-1]:
        raise ValueError(f"Invalid feat_start={feat_start} for points with C={points.shape[-1]}")

    n_feat = int(points.shape[-1] - feat_start)
    if n_feat != n_scalers:
        raise ValueError(
            f"Scaler/feature mismatch: points has {n_feat} feature channels (C={points.shape[-1]}, feat_start={feat_start}) "
            f"but scalers.pkl has {n_scalers}. "
            f"This means your eval input channels are not the same as training."
        )

    for k in range(n_scalers):
        i = feat_start + k
        mean, scale = scalers[k]
        points[:, :, i] = (points[:, :, i] - mean) / (scale + 1e-12)

    return points

def _load_points_by_source(source: str, hamamatsu: bool) -> Tuple[np.ndarray, Dict]:
    """
    Returns (points, meta) for the requested source.

    - chimney / FC / data_muon_new: use PMTLoader.load_eval_source_points (3 channels: [nPE,FHT,slope])
    - data_muon / mc_muon / mc_muon_new: stack multiple features via get_stacked_data
    """
    key = str(source).strip().lower()

    # NEW: data_muon_new follows chimney/FC format
    if key in ("chimney", "fc", "data_muon_new"):
        return load_eval_source_points(key)

    # NEW: mc_muon_new follows data_muon/mc_muon format
    if key in ("data_muon", "mc_muon", "mc_muon_new"):
        if key == "data_muon":
            folder = "/disk_pool1/wangjb/data_muon"
        elif key == "mc_muon":
            folder = "/disk_pool1/wangjb/mc_muon"
        else:
            folder = "/disk_pool1/houyh/data/test/mc_muon_k/pmt_npy"

        feature_list = ["npe", "fht", "slope4"]

        # IMPORTANT: do NOT force p_hint=4997 here.
        # Load full PMT first (likely 17612), then main() will filter to 4997 if hamamatsu=True.
        all_features = [get_stacked_data(folder, f) for f in feature_list]
        x_all = np.stack(all_features, axis=-1).astype(np.float32, copy=False)

        meta = {
            "source": key,
            "dir": folder,
            "feature_list": feature_list,
            "N": int(x_all.shape[0]),
            "P": int(x_all.shape[1]),
            "C": int(x_all.shape[2]),
        }
        return x_all, meta

    raise ValueError(f"Unknown --source={source}")


def _load_hamamatsu_indices(csv_path: str) -> np.ndarray:
    """
    Copied from filter.py logic (no new hyperparams):
    Return sorted indices where type == 'Hamamatsu'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    idxs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header_seen = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("//"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            if not header_seen and parts[0].lower() == "index" and parts[1].lower() == "type":
                header_seen = True
                continue

            if parts[0].isdigit():
                i = int(parts[0])
                t = parts[1]
                if t == "Hamamatsu":
                    idxs.append(i)

    if not idxs:
        raise RuntimeError(f"No Hamamatsu indices found in: {csv_path}")

    return np.array(sorted(idxs), dtype=np.int64)


def _filter_points_by_pmt_type(points: np.ndarray, hamamatsu: bool, csv_path: str = PMT_TYPE_CSV_PATH) -> Tuple[np.ndarray, Dict]:
    """
    Apply filtering on PMT dimension P based on PMTType_CD_LPMT.csv.

    - hamamatsu=True  -> keep only type == 'Hamamatsu'
    - hamamatsu=False -> keep NNVT PMTs: type in {'NNVT', 'HighQENNVT'}
    """
    if points.ndim != 3:
        raise ValueError(f"points must be [N,P,C], got {points.shape}")

    n, p, c = points.shape
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    hama_idx = []
    nnvt_idx = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header_seen = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            if not header_seen and parts[0].lower() == "index" and parts[1].lower() == "type":
                header_seen = True
                continue
            if parts[0].isdigit():
                i = int(parts[0])
                t = parts[1]
                if t == "Hamamatsu":
                    hama_idx.append(i)
                elif t in ("NNVT", "HighQENNVT"):
                    nnvt_idx.append(i)

    if not hama_idx:
        raise RuntimeError(f"No Hamamatsu indices found in: {csv_path}")
    if not nnvt_idx:
        raise RuntimeError(f"No NNVT/HighQENNVT indices found in: {csv_path}")

    hama_idx = np.array(sorted(hama_idx), dtype=np.int64)
    nnvt_idx = np.array(sorted(nnvt_idx), dtype=np.int64)

    if hama_idx.max() >= p or nnvt_idx.max() >= p:
        raise ValueError(f"Index out of range: P={p}, hama_max={int(hama_idx.max())}, nnvt_max={int(nnvt_idx.max())}")

    if hamamatsu:
        kept = points[:, hama_idx, :].astype(np.float32, copy=False)
        info = {"filter": "hamamatsu", "csv": csv_path, "P_in": int(p), "P_out": int(kept.shape[1])}
        return kept, info

    kept = points[:, nnvt_idx, :].astype(np.float32, copy=False)
    info = {"filter": "nnvt", "csv": csv_path, "P_in": int(p), "P_out": int(kept.shape[1])}
    return kept, info

@torch.no_grad()
def predict(model, loader, device, desc="Predicting"):
    model.eval()
    pred_class_all, prob_all, logits_all = [], [], []
    for points, _ in tqdm(loader, desc=desc, total=len(loader), leave=True):
        points = points.float().to(device)
        logits, _ = model(points)              # logits: [B,3]
        prob = torch.softmax(logits, dim=1)    # prob:   [B,3]
        pred_class = torch.argmax(logits, dim=1)

        logits_all.append(logits.detach().cpu().numpy())
        pred_class_all.append(pred_class.detach().cpu().numpy())
        prob_all.append(prob.detach().cpu().numpy())

    return (
        np.concatenate(pred_class_all, axis=0).astype(np.int64),
        np.concatenate(prob_all, axis=0).astype(np.float32),
        np.concatenate(logits_all, axis=0).astype(np.float32),
    )


@torch.no_grad()
def predict_logits(model, loader, device, desc="Predicting"):
    model.eval()
    logits_all = []
    for points, _ in tqdm(loader, desc=desc, total=len(loader), leave=True):
        points = points.float().to(device)
        logits, _ = model(points)  # logits: [B,3]
        logits_all.append(logits.detach().cpu().numpy())
    return np.concatenate(logits_all, axis=0).astype(np.float32)  # [N,3]


def _setup_plot_style():
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['figure.dpi'] = 500
    return colors


def _plot_scores_template(prob: np.ndarray, out_dir: str, tag: str = ""):
    """
    Plot 3-class softmax scores following the provided template style.
    prob: [N,3] with class order = CLASS_NAMES = ["nc","Vmu","Ve"].
    Saves:
      - {tag}_mulikescore.png
      - {tag}_elikescore.png
      - {tag}_NClikescore.png
    """
    if prob.ndim != 2 or prob.shape[1] != 3:
        raise ValueError(f"prob must be [N,3], got {prob.shape}")

    os.makedirs(out_dir, exist_ok=True)
    colors = _setup_plot_style()

    # we only have one distribution here (current source), so we mimic "MC vs Data"
    # by drawing the same data twice? No: follow template structure but use single curve.
    # Use colors[0] and label as the source tag.
    label0 = tag if tag else "score"

    # mu-like (Vmu) -> index 1
    plt.figure(figsize=(10, 6))
    plt.hist(
        prob[:, 1], range=[0, 1], bins=50,
        color=colors[0], histtype='step', linewidth=2,
        label=label0, density=True
    )
    plt.xlabel(r"$\mu$-like score")
    plt.ylabel("P.D.F")
    plt.title(r"Hamm-PMT Only, $\mu$-like")
    plt.legend(loc='upper left', frameon=False, fontsize=20)
    plt.savefig(os.path.join(out_dir, f"{tag}_mulikescore.png" if tag else "mulikescore.png"))
    plt.close()

    # # e-like (Ve) -> index 2
    # plt.figure(figsize=(10, 6))
    # plt.hist(
    #     prob[:, 2], range=[0, 1], bins=50,
    #     color=colors[0], histtype='step', linewidth=2,
    #     label=label0, density=True
    # )
    # plt.xlabel(r"e-like score")
    # plt.ylabel("P.D.F")
    # plt.title(r"Hamm-PMT Only, e-like")
    # plt.legend(loc='upper right', frameon=False, fontsize=20)
    # plt.savefig(os.path.join(out_dir, f"{tag}_elikescore.png" if tag else "elikescore.png"))
    # plt.close()

    # # NC-like (nc) -> index 0
    # plt.figure(figsize=(10, 6))
    # plt.hist(
    #     prob[:, 0], range=[0, 1], bins=50,
    #     color=colors[0], histtype='step', linewidth=2,
    #     label=label0, density=True
    # )
    # plt.xlabel(r"NC-like score")
    # plt.ylabel("P.D.F")
    # plt.title(r"Hamm-PMT Only, NC-like")
    # plt.legend(loc='upper right', frameon=False, fontsize=20)
    # plt.savefig(os.path.join(out_dir, f"{tag}_NClikescore.png" if tag else "NClikescore.png"))
    # plt.close()

def _load_coords_tiled(n: int, p: int, hamamatsu: bool = False, nnvt: bool = False) -> np.ndarray:
    """
    Load (P,3) coords template then broadcast to (N,P,3).
    - hamamatsu=True -> COORDS_HAMA_PATH
    - nnvt=True      -> COORDS_NNVT_PATH (must exist)
    - both False     -> COORDS_SINGLE_PATH
    """
    if hamamatsu and nnvt:
        raise ValueError("Invalid config: both hamamatsu=True and nnvt=True for coords.")

    if hamamatsu:
        coords_path = COORDS_HAMA_PATH
    elif nnvt:
        coords_path = COORDS_NNVT_PATH
    else:
        coords_path = COORDS_SINGLE_PATH

    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Missing coords file: {coords_path}")

    coords = np.load(coords_path, mmap_mode="r")
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be [P,3], got {coords.shape} from {coords_path}")
    if coords.shape[0] != p:
        raise ValueError(f"P mismatch: coords P={coords.shape[0]} vs feats P={p} (file={coords_path})")

    tmpl = np.asarray(coords, dtype=np.float32)  # [P,3]
    return np.broadcast_to(tmpl[None, :, :], (n, p, 3)).copy()


def _maybe_concat_coords(points: np.ndarray, coordinates: bool, hamamatsu: bool, nnvt: bool) -> Tuple[np.ndarray, int]:
    """
    If coordinates=True, prepend xyz to features.
    Returns (new_points, feat_start) where feat_start=3 if coords used else 0.
    """
    if not coordinates:
        return points.astype(np.float32, copy=False), 0

    if points.ndim != 3:
        raise ValueError(f"points must be [N,P,C], got {points.shape}")

    n, p, _ = points.shape
    coords = _load_coords_tiled(n=n, p=p, hamamatsu=hamamatsu, nnvt=nnvt)  # [N,P,3]
    out = np.concatenate([coords, points], axis=-1).astype(np.float32, copy=False)  # [N,P,3+C]
    return out, 3

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    model_path = os.path.join(args.log_dir, "best_pointnet_regression_model.pth")
    scalers_path = os.path.join(args.log_dir, "scalers.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    meta = _apply_meta_defaults(args)

    # load features by --source (external data)
    points, src_meta = _load_points_by_source(args.source, hamamatsu=bool(args.hamamatsu))
    src_meta = dict(src_meta)

    # Select PMT subset based on meta flags (hamamatsu/nnvt)
    points, filt_meta = _select_pmt_subset(
        points,
        hamamatsu=bool(getattr(args, "hamamatsu", False)),
        nnvt=bool(getattr(args, "nnvt", False)),
        csv_path=PMT_TYPE_CSV_PATH,
    )
    src_meta["pmt_filter"] = filt_meta

    # Sanity check: match training num_points if present
    expected_p = int(getattr(args, "num_points", points.shape[1]))
    if expected_p != int(points.shape[1]):
        raise ValueError(
            f"num_points mismatch vs training meta: expected {expected_p}, got {int(points.shape[1])}. "
            f"Check hamamatsu/nnvt flags and CSV indexing."
        )

    # always follow training meta
    use_coords = bool(args.coordinates)

    points, _feat_start = _maybe_concat_coords(
        points,
        coordinates=use_coords,
        hamamatsu=bool(getattr(args, "hamamatsu", False)),
        nnvt=bool(getattr(args, "nnvt", False)),
    )

    _apply_saved_scalers(points, scalers_path)

    labels = np.full((points.shape[0],), VMU_LABEL, dtype=np.int64)
    loader = DataLoader(CustomDataset(points, labels), batch_size=args.batch_size, shuffle=False)

    if meta and "model" in meta and "feat_dim" in meta["model"]:
        feat_dim = int(meta["model"]["feat_dim"])
    else:
        feat_dim = int(points.shape[-1] - 3) if use_coords else 0

    model = get_model(3, coordinates=use_coords, feat_dim=feat_dim).to(device).float()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    pred_cls, prob, logits = predict(model, loader, device, desc=f"Predicting ({args.source})")
    # logits = predict_logits(model, loader, device, desc=f"Predicting logits ({args.source})")

    # NEW: print score range for debugging/interpretation
    mu_score = prob[:, VMU_LABEL]
    q = np.quantile(mu_score, [0.0, 0.01, 0.5, 0.99, 1.0])
    print(
        "[EVAL] mu-like score stats: "
        f"min={q[0]:.6g}, p01={q[1]:.6g}, median={q[2]:.6g}, p99={q[3]:.6g}, max={q[4]:.6g}"
    )
    print(f"[EVAL] mu-like == 1.0 count: {int(np.sum(mu_score == 1.0))} / {int(mu_score.size)}")


    out_dir = os.path.join(args.log_dir, "test")
    os.makedirs(out_dir, exist_ok=True)

    out_npz = os.path.join(out_dir, f"pred_{str(args.source).lower()}.npz")
    np.savez(
        out_npz,
        logits=logits,
        prob=prob,
        mu_like_score=prob[:, VMU_LABEL],
        pred_class=pred_cls,
        class_names=np.array(CLASS_NAMES),
        source=np.array([str(args.source)]),
        source_meta=json.dumps(src_meta, ensure_ascii=False),
        points_shape=np.array(points.shape, dtype=np.int64),
    )

    _plot_scores_template(prob=prob, out_dir=out_dir, tag=str(args.source).lower())

    run_json = os.path.join(out_dir, f"run_{str(args.source).lower()}.json")
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": args.source,
                "log_dir": args.log_dir,
                "out_dir": out_dir,
                "gpu": args.gpu,
                "batch_size": args.batch_size,
                "coordinates": bool(args.coordinates),
                "feature_mode": str(args.feature_mode),
                "eps": float(args.eps),
                "hamamatsu": bool(getattr(args, "hamamatsu", False)),
                "nnvt": bool(getattr(args, "nnvt", False)),
                "num_points": int(getattr(args, "num_points", points.shape[1])),
                "points_shape": list(points.shape),
                "src_meta": src_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[EVAL] source={args.source} points={points.shape} (saved under {out_dir})")
    # print(f"Saved pred : {out_npz}")
    # print(f"Saved plots: {os.path.join(out_dir, f'{str(args.source).lower()}_mulikescore.png')}")
    # print(f"Saved run  : {run_json}")

if __name__ == "__main__":
    main(parse_args())