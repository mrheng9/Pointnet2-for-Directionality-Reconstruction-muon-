import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.PMTLoader import CustomDataset
from models.pointnet_regression_ssg import get_model, get_loss

# Import the same loader/utilities from train.py
from train import load_data_with_splits

def parse_args():
    p = argparse.ArgumentParser("evaluation")
    p.add_argument('--use_cpu', action='store_true', default=False)
    p.add_argument('--gpu', type=str, default='1')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--data_source', type=str, default=None, help='det|elec|cnn|rawnet (default: read from train_meta.json)')
    p.add_argument('--log_dir', type=str, default='/home/houyh/experiments/test', help='experiment dir produced by train.py')
    return p.parse_args()

@torch.no_grad()
def predict(model, loader, device, desc="Evaluating"):
    model.eval()
    preds, targets = [], []
    for points, target in tqdm(loader, desc=desc, total=len(loader), leave=True):
        points = points.float().to(device)
        target = target.float().to(device)
        pred, _ = model(points)
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)

def _resolve_data_source(args):
    if args.data_source is not None:
        return args.data_source
    meta_path = os.path.join(args.log_dir, "train_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        ds = meta.get("data_source", None)
        if ds is None:
            raise ValueError(f"`data_source` not found in {meta_path}. Please pass --data_source explicitly.")
        args.data_source = ds
        return ds
    raise ValueError("Missing --data_source and train_meta.json not found. Please pass --data_source explicitly.")

def main(args):
    _resolve_data_source(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    model_path = os.path.join(args.log_dir, "best_pointnet_regression_model.pth")
    splits_path = os.path.join(args.log_dir, "splits.npz")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(splits_path):
        raise FileNotFoundError(splits_path)

    points_train, points_val, points_test, labels_train, labels_val, labels_test, scalers, split_meta = load_data_with_splits(
        args,
        save_splits_to=None,
        load_splits_from=splits_path,
    )

    test_loader = DataLoader(
        CustomDataset(points_test, labels_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = get_model(3, normal_channel=True).to(device).float()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    preds, targets = predict(model, test_loader, device, desc="Predicting (test)")

    out_path = os.path.join(args.log_dir, "predictions_test.npz")
    np.savez(out_path, y_pred=preds.astype(np.float32), y_true=targets.astype(np.float32))
    print(f"Saved: {out_path}  pred shape={preds.shape}, true shape={targets.shape}")
if __name__ == "__main__":
    main(parse_args())