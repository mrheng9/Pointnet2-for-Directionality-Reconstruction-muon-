import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

###############################################################
#1. Direction Reconstruction Performance Plots
###############################################################

def parse_args():
    p = argparse.ArgumentParser("plots")
    p.add_argument("--log_dir", type=str, default='/home/houyh/experiments/test/plots')
    p.add_argument("--pred_file", type=str, default="/home/houyh/experiments/test/predictions_test.npz")
    return p.parse_args()

def draw_performance(x, y, out_path):
    plt.clf()
    plt.grid()
    plt.scatter(x, y, label="predictions", color='red', s=10)
    plt.plot([np.amin(x), np.amax(x)], [np.amin(x), np.amax(x)], 'k-', alpha=0.75, zorder=0, label="y=x", linewidth=2)
    plt.legend()
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Test Performance")
    plt.savefig(out_path, dpi=300)
    plt.close()

def draw_error_distribution(true_vals, pred_vals, out_path):
    plt.figure(figsize=(12, 8))
    differences = (pred_vals - true_vals) * 180 / np.pi
    plt.hist(differences, bins=50, alpha=0.7, density=True, label='Prediction Errors')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    plt.axvline(x=mean_diff, color='g', linestyle='--', label='Mean Error')
    plt.legend(loc="upper right")
    plt.xlabel('Prediction Error (deg)')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=300)
    plt.close()

def draw_angle_distribution(true_vecs, pred_vecs, out_path):
    cos_angles = np.sum(true_vecs * pred_vecs, axis=1) / (
        np.linalg.norm(true_vecs, axis=1) * np.linalg.norm(pred_vecs, axis=1)
    )
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles_deg = np.rad2deg(np.arccos(cos_angles))

    (mu, sigma) = norm.fit(angles_deg)
    sorted_angles = np.sort(angles_deg)
    size = len(sorted_angles)
    q68 = sorted_angles[int(0.68 * size) - 1] if size > 0 else float('nan')

    plt.figure()
    plt.hist(angles_deg, bins=100, range=(0, 18), color='green', density=True)
    plt.axvline(x=q68, color='black', linewidth=2, linestyle='--', label=f'68% quantile: {q68:.2f}')
    plt.xlim(0, 18)
    plt.legend(frameon=False)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel('Opening Angle α (deg)')
    plt.ylabel('P.D.F')
    plt.title(f'Opening Angle P.D.F (mu={mu:.2f}, sigma={sigma:.2f})')
    plt.savefig(out_path, dpi=500)
    plt.close()

def main(args):
    pred_path = os.path.join(args.pred_file)
    data = np.load(pred_path)
    y_pred = data["y_pred"]
    y_true = data["y_true"]

    # theta performance (same as your train.py)
    theta_pred = np.arctan(np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2) / y_pred[:, 2])
    theta_true = np.arctan(np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2) / y_true[:, 2])
    theta_pred[theta_pred < 0] += np.pi
    theta_true[theta_true < 0] += np.pi
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
#1. Direction Reconstruction Performance Plots
    draw_performance(theta_true, theta_pred, os.path.join(args.log_dir, "test_performance.png"))
    draw_error_distribution(theta_true, theta_pred, os.path.join(args.log_dir, "error_distribution.png"))
    draw_angle_distribution(y_true, y_pred, os.path.join(args.log_dir, "angle_distribution.png"))

    print("Plots saved to:", args.log_dir)

if __name__ == "__main__":
    main(parse_args())