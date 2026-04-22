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
    p.add_argument("--log_dir", type=str, default='experiments_vertex/test/plots')
    p.add_argument("--pred_file", type=str, default="experiments_vertex/test/predictions_test.npz")
    p.add_argument(
        "--task",
        type=str,
        choices=["direction", "vertex"],
        default="direction",
        help="Which plots to generate.",
    )
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

###############################################################
#2. Vertex Reconstruction Performance Plots
###############################################################

def draw_vertex_distance_q68(true_xyz, pred_xyz, out_path, bins=80):
    """
    Vertex reconstruction metric:
    per-sample Euclidean distance d = ||pred - true||,
    plot its distribution and mark 68% quantile.
    """
    true_xyz = np.asarray(true_xyz)
    pred_xyz = np.asarray(pred_xyz)
    if true_xyz.ndim != 2 or pred_xyz.ndim != 2 or true_xyz.shape[1] != 3 or pred_xyz.shape[1] != 3:
        raise ValueError(f"Expected true/pred shape (N,3). Got {true_xyz.shape} and {pred_xyz.shape}")

    d = np.linalg.norm(pred_xyz - true_xyz, axis=1)
    d = d[np.isfinite(d)]

    if d.size == 0:
        raise ValueError("No finite distances to plot.")

    q68 = np.quantile(d, 0.68)
    median = np.quantile(d, 0.50)
    mean = float(np.mean(d))

    plt.figure(figsize=(10, 7))
    plt.hist(d, bins=bins, density=True, alpha=0.75, color="steelblue", label="|pred-true| distances")
    plt.axvline(q68, color="black", linewidth=2, linestyle="--", label=f"68% quantile: {q68:.4g}")
    plt.axvline(median, color="green", linewidth=2, linestyle="--", label=f"median: {median:.4g}")
    plt.axvline(mean, color="red", linewidth=2, linestyle="--", label=f"mean: {mean:.4g}")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.xlabel("Euclidean distance ||pred - true||")
    plt.ylabel("P.D.F")
    plt.title("Vertex Reconstruction Error Distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def main(args):
    data = np.load(args.pred_file)
    y_pred = np.asarray(data["y_pred"])
    y_true = np.asarray(data["y_true"])

    os.makedirs(args.log_dir, exist_ok=True)

    if args.task == "direction":
        # theta performance (more stable with arctan2 than arctan(x/z))
        theta_pred = np.arctan2(np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2), y_pred[:, 2])
        theta_true = np.arctan2(np.sqrt(y_true[:, 0]**2 + y_true[:, 1]**2), y_true[:, 2])
        theta_pred[theta_pred < 0] += np.pi
        theta_true[theta_true < 0] += np.pi

        draw_performance(theta_true, theta_pred, os.path.join(args.log_dir, "test_performance.png"))
        draw_error_distribution(theta_true, theta_pred, os.path.join(args.log_dir, "error_distribution.png"))
        draw_angle_distribution(y_true, y_pred, os.path.join(args.log_dir, "angle_distribution.png"))

    elif args.task == "vertex":
        draw_vertex_distance_q68(
            y_true,
            y_pred,
            os.path.join(args.log_dir, "vertex_distance_q68.png"),
        )

    else:
        raise ValueError(f"Unknown task: {args.task}")
    print(f"Plots saved to: {args.log_dir}")

if __name__ == "__main__":
    main(parse_args())