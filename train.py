import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1,2,3'
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from data_utils.PMTLoader import PMTDataLoader,CustomDataset,get_stacked_datanorm,get_stacked_datawei,get_stacked_datachy,get_stacked_dataweiCNN
from data_utils.PMTLoader import jitter_point_cloud, random_point_dropout
# from models.pointnet_regression import get_model,get_loss
from models.pointnet_regression_ssg import get_model,get_loss
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='2', help='specify gpu device')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=1, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay in training')
    parser.add_argument('--data_source',type=str,default='cnn',help='choose from: det|elec|cnn|rawnet')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)

    parser.add_argument('--log_dir', type=str, default='/home/houyh/Pointnet2-for-Directionality-Reconstruction-muon-/experiments/test', help='experiment root')
    return parser.parse_args()

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def augment_point_cloud(batch_data, jitter=True, dropout=True):
    if isinstance(batch_data, torch.Tensor):
        is_tensor = True
        device = batch_data.device
        batch_data = batch_data.cpu().numpy()
    else:
        is_tensor = False
        
    coords = batch_data[:, :, 0:3].copy()
    features = batch_data[:, :, 3:].copy() if batch_data.shape[2] >= 3 else None
    
    # 1. simulate jitter (only coordinates are affected)
    if jitter:
        coords = jitter_point_cloud(coords, sigma=0.005, clip=0.02)

    if features is not None:
        augmented_data = np.concatenate([coords, features], axis=2)
    else:
        augmented_data = coords
        
    # 2. simulate dropout (only coordinates are affected)
    if dropout:
        augmented_data = random_point_dropout(augmented_data, max_dropout_ratio=0.3)
    
    if is_tensor:
        augmented_data = torch.from_numpy(augmented_data).to(device)
        
    return augmented_data

def split_indices(n, test_size=0.2, val_size=0.2, seed=42):
    """
    val_size: fraction of (n - n_test), i.e. applied after removing test split.
    """
    all_idx = np.arange(n)
    trainval_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_size, random_state=seed, shuffle=True)
    return train_idx, val_idx, test_idx

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

def load_data_with_splits(args, save_splits_to=None, load_splits_from=None):
    """
    Returns:
      points_train, points_val, points_test, labels_train, labels_val, labels_test, scalers, split_dict
    """
    coord_all = np.load('/disk_pool1/houyh/coords/norm_coords')
    folder_path_y = '/disk_pool1/houyh/data/y'
    folder_path1 = '/disk_pool1/chenzhx/rebuilt_data/rawnet/pmt_together4'
    folder_path2 = '/disk_pool1/houyh/data/det_pmt'
    folder_path3 = '/disk_pool1/houyh/data/reco_pmt'
    folder_path4 = '/disk_pool1/houyh/data/elec_pmt'

    src = args.data_source.lower()
    all_features = []

    if src == 'det':
        feature_list = ["pmt_fht", "pmt_slope","pmt_nperatio5","pmt_peaktime","pmt_timemax","pmt_npe"]
        all_features = [get_stacked_datawei(folder_path2, f) for f in feature_list]
        x_all = np.stack(all_features, axis=-1)

    elif src == 'elec':
        feature_list = ["fht","slope","peak","timemax","nperatio5", "npe"]
        all_features = [get_stacked_dataweiCNN(folder_path4, f) for f in feature_list]
        x_all = np.stack(all_features, axis=-1)
        coord_all = coord_all[:9850]

    elif src == 'cnn':
        feature_list = ["fht","slope","peak","timemax","nperatio5"]
        all_features = [get_stacked_dataweiCNN(folder_path3, f) for f in feature_list]
        all_features.append(get_stacked_dataweiCNN(folder_path4, "npe"))
        x_all = np.stack(all_features, axis=-1)
        coord_all = coord_all[:9850]

    elif src == 'rawnet':
        x_all = get_stacked_datanorm(folder_path1)

    else:
        raise ValueError(f'Unknown data_source: {args.data_source}')

    # optional ratio feature correction (keep your intent, but fix condition)
    if src in ('det', 'elec', 'cnn', 'rawnet') and x_all.shape[-1] >= 2:
        epsilon = 1e-8
        x_all[:, :, 1] = x_all[:, :, 1] / (x_all[:, :, -1] + epsilon)

    x_all = np.concatenate([coord_all, x_all], axis=-1)

    y_all = get_stacked_datachy(folder_path_y, "y")
    if src in ('cnn', 'elec'):
        y_all = y_all[:9850]

    coordx_in = np.sin(y_all[:, 0]) * np.cos(y_all[:, 1])
    coordy_in = np.sin(y_all[:, 0]) * np.sin(y_all[:, 1])
    coordz_in = np.cos(y_all[:, 0])
    labels = np.stack((coordx_in, coordy_in, coordz_in), axis=-1).astype(np.float32)
    points = x_all.astype(np.float32)

    n = len(points)

    # splits
    if load_splits_from and os.path.exists(load_splits_from):
        split_npz = np.load(load_splits_from)
        train_idx = split_npz["train_idx"]
        val_idx = split_npz["val_idx"]
        test_idx = split_npz["test_idx"]
    else:
        train_idx, val_idx, test_idx = split_indices(
            n,
            test_size=getattr(args, "test_size", 0.2),
            val_size=getattr(args, "val_size", 0.2),
            seed=getattr(args, "seed", 42),
        )
        if save_splits_to:
            os.makedirs(os.path.dirname(save_splits_to), exist_ok=True)
            np.savez(save_splits_to, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    points_train, labels_train = points[train_idx], labels[train_idx]
    points_val, labels_val = points[val_idx], labels[val_idx]
    points_test, labels_test = points[test_idx], labels[test_idx]

    # fit scalers on train only, apply to val/test
    feat_start = 3
    scalers = fit_feature_scalers(points_train, feat_start=feat_start)
    apply_feature_scalers(points_train, scalers, feat_start=feat_start)
    apply_feature_scalers(points_val, scalers, feat_start=feat_start)
    apply_feature_scalers(points_test, scalers, feat_start=feat_start)

    # ensure unit vectors (optional safety)
    eps = 1e-12
    labels_train = labels_train / (np.linalg.norm(labels_train, axis=1, keepdims=True) + eps)
    labels_val = labels_val / (np.linalg.norm(labels_val, axis=1, keepdims=True) + eps)
    labels_test = labels_test / (np.linalg.norm(labels_test, axis=1, keepdims=True) + eps)

    split_dict = {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
    }
    return points_train, points_val, points_test, labels_train, labels_val, labels_test, scalers, split_dict


def draw_learning_curve(train_losses, test_losses):
    plt.figure(figsize=(12, 8))  
    plt.plot(range(1, args.epoch + 1), train_losses, label='Train Loss', linewidth=2)
    plt.plot(range(1, args.epoch + 1), test_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.legend(fontsize=16)
    plt.title('Learning Curve', fontsize=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.yscale('log')
    plt.savefig(os.path.join(args.log_dir, 'learning_curve.png') if args.log_dir else 'learning_curve.png', dpi=300)
    plt.close()



def train(model, criterion, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for points, target in tqdm(train_loader,desc='Training'):
        points = points.float().to(device)
        target = target.float().to(device)

        # data augmentation
        #points = augment_point_cloud(points, jitter=True, dropout=True)

        optimizer.zero_grad()
        pred, _ = model(points)

        loss = criterion(pred, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, criterion, loader, device, desc="Eval"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for points, target in tqdm(loader, desc=desc):
            points = points.float().to(device)
            target = target.float().to(device)
            pred, _ = model(points)
            loss = criterion(pred, target)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))

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

    if scalers_path:
        with open(scalers_path, "wb") as f:
            pickle.dump({"feat_start": 3, "scalers": scalers}, f)

    if meta_path:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_source": args.data_source,
                    "split_meta": split_meta,
                    "batch_size": args.batch_size,
                    "epoch": args.epoch,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "seed": getattr(args, "seed", 42),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    model = get_model(3, normal_channel=True).to(device).float()
    criterion = get_loss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)

    train_loader = DataLoader(CustomDataset(points_train, labels_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CustomDataset(points_val, labels_val), batch_size=args.batch_size, shuffle=False)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(args.epoch):
        train_loss = train(model, criterion, optimizer, train_loader, device)
        val_loss = evaluate(model, criterion, val_loader, device, desc="Validation")

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        logger.info(f'Epoch [{epoch+1}/{args.epoch}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    if args.log_dir is not None and best_model_state is not None:
        torch.save(best_model_state, os.path.join(args.log_dir, 'best_pointnet_regression_model.pth'))

    # learning curve now is train vs val (not test)
    draw_learning_curve(train_losses, val_losses)

if __name__ == '__main__':
    args = parse_args()
    main(args)