import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.PMTLoader import CustomDataset
from models.pointnet_regression_ssg import get_model

from train import load_data_with_splits

def parse_args():
    p = argparse.ArgumentParser("evaluation")
    p.add_argument('--use_cpu', action='store_true', default=False)
    p.add_argument('--gpu', type=str, default='3')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument(
        '--log_dir',
        type=str,
        default='/home/houyh/Pointnet2-for-Directionality-Reconstruction-muon-/experiments/test_divide',
        help='experiment dir produced by train.py (must contain train_meta.json & splits.npz)',
    )
    return p.parse_args()

def _apply_meta_defaults(args):
    meta_path = os.path.join(args.log_dir, "train_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing train_meta.json at: {meta_path}. "
            f"Please evaluate with a valid --log_dir produced by train.py."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # required keys for reproducible eval
    required = ["pid_dataset_dir", "coordinates", "feature_mode", "use_frac", "eps", "hamamatsu"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(f"train_meta.json missing keys: {missing}. Please re-train to regenerate meta.")

    # inject into args (so load_data_with_splits sees them)
    args.pid_dataset_dir = meta["pid_dataset_dir"]
    args.coordinates = bool(meta["coordinates"])
    args.feature_mode = str(meta["feature_mode"])
    args.use_frac = float(meta["use_frac"])
    args.eps = float(meta["eps"])
    args.hamamatsu = bool(meta["hamamatsu"])

    return meta

@torch.no_grad()
def predict(model, loader, device, desc="Evaluating"):
    model.eval()
    pred_class_all, true_class_all, prob_all = [], [], []

    for points, target in tqdm(loader, desc=desc, total=len(loader), leave=True):
        points = points.float().to(device)
        target = target.long().to(device)

        logits, _ = model(points)
        prob = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1)

        pred_class_all.append(pred_class.detach().cpu().numpy())
        true_class_all.append(target.detach().cpu().numpy())
        prob_all.append(prob.detach().cpu().numpy())

    return (
        np.concatenate(pred_class_all, axis=0),
        np.concatenate(true_class_all, axis=0),
        np.concatenate(prob_all, axis=0),
    )

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    model_path = os.path.join(args.log_dir, "best_pointnet_regression_model.pth")
    splits_path = os.path.join(args.log_dir, "splits.npz")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(splits_path):
        raise FileNotFoundError(splits_path)

    meta = _apply_meta_defaults(args)

    class_names = ["nc", "Vmu", "Ve"]
    if meta and "class_names" in meta:
        class_names = meta["class_names"]

    # load test split exactly as training (splits.npz)
    _, _, points_test, _, _, labels_test, _, split_meta = load_data_with_splits(
        args,
        save_splits_to=None,
        load_splits_from=splits_path,
    )

    print(
        f"[EVAL DATA] N_test={int(points_test.shape[0])} (P={int(points_test.shape[1])}, C={int(points_test.shape[2])}) "
        f"| counts_test={np.bincount(labels_test, minlength=3).tolist()} "
        f"| hamamatsu={bool(getattr(args,'hamamatsu',False))} coordinates={bool(getattr(args,'coordinates',False))} "
        f"| feature_mode={getattr(args,'feature_mode','?')}"
    )

    test_loader = DataLoader(
        CustomDataset(points_test, labels_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if meta and "model" in meta and "feat_dim" in meta["model"]:
        feat_dim = int(meta["model"]["feat_dim"])
    else:
        feat_dim = int(points_test.shape[-1] - 3) if bool(args.coordinates) else 0

    model = get_model(3, coordinates=bool(args.coordinates), feat_dim=feat_dim).to(device).float()

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    pred_class, true_class, prob = predict(model, test_loader, device, desc="Predicting (test)")

    out_path = os.path.join(args.log_dir, "predictions_test.npz")
    np.savez(
        out_path,
        pred_class=pred_class.astype(np.int64),
        true_class=true_class.astype(np.int64),
        prob=prob.astype(np.float32),
        class_names=np.array(class_names),
    )

    acc = float((pred_class == true_class).mean())
    metrics_path = os.path.join(args.log_dir, "metrics_test.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "num_test": int(true_class.shape[0]),
                "class_names": class_names,
                "split_meta": split_meta,
                "hamamatsu": bool(args.hamamatsu),
                "coordinates": bool(args.coordinates),
                "feature_mode": args.feature_mode,
                "eps": float(args.eps),
                "pid_dataset_dir": args.pid_dataset_dir,
                "use_frac": float(args.use_frac),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved: {out_path}")
    print(f"Saved: {metrics_path}  accuracy={acc:.6f}")

if __name__ == "__main__":
    main(parse_args())