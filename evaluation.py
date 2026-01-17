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
    p.add_argument('--gpu', type=str, default='2')
    p.add_argument('--batch_size', type=int, default=32)

    # PID dataset args (defaults; may be overwritten by train_meta.json)
    p.add_argument("--pid_dataset_dir", type=str, default="/disk_pool1/houyh/data/J23_J25_7_2/pid_dataset")
    p.add_argument("--coordinates", action="store_true", help="must match training")
    p.add_argument("--feature_mode", type=str, default="normal", choices=["normal", "divide", "log", "dlog"])
    p.add_argument("--use_frac", type=float, default=1.0) 
    p.add_argument("--eps", type=float, default=1e-6)

    p.add_argument('--log_dir', type=str, default='/home/houyh/Pointnet2-for-Directionality-Reconstruction-muon-/experiments/test1', help='experiment dir produced by train.py')
    return p.parse_args()

def _apply_meta_defaults(args):
    meta_path = os.path.join(args.log_dir, "train_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # override runtime args with meta to guarantee consistency
    if "pid_dataset_dir" in meta and meta["pid_dataset_dir"]:
        args.pid_dataset_dir = meta["pid_dataset_dir"]
    if "coordinates" in meta:
        args.coordinates = bool(meta["coordinates"])
    if "feature_mode" in meta:
        args.feature_mode = meta["feature_mode"]
    if "use_frac" in meta:
        args.use_frac = float(meta["use_frac"])
    if "eps" in meta:
        args.eps = float(meta["eps"])

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
    meta_path = os.path.join(args.log_dir, "train_meta.json")

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

    # extra explicit print (so you see it at eval start)
    print(
        f"[EVAL DATA] N_test={int(points_test.shape[0])} (P={int(points_test.shape[1])}, C={int(points_test.shape[2])}) "
        f"| counts_test={np.bincount(labels_test, minlength=3).tolist()}"
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
                "coordinates": bool(args.coordinates),
                "feature_mode": args.feature_mode,
                "eps": float(args.eps),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved: {out_path}")
    print(f"Saved: {metrics_path}  accuracy={acc:.6f}")

if __name__ == "__main__":
    main(parse_args())