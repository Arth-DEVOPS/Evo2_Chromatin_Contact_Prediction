import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

def analyze_performance(
    pred_path="evo2/contact_map/HFF/pred.npy",
    target_path="evo2/contact_map/HFF/target.npy",
    output_dir="evo2/contact_map/HFF/analysis_improved_head"
):
    """
    Analyze Evo2 Contact Map predictions by computing Pearson correlation coefficients (PCC)
    and generating representative visualizations.

    Args:
        pred_path (str): Path to predicted contact maps (.npy file)
        target_path (str): Path to ground truth contact maps (.npy file)
        output_dir (str): Directory to save analysis results
    """

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load predictions and targets
    if not os.path.exists(pred_path) or not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Could not find pred.npy or target.npy at {os.path.dirname(pred_path)}. "
            "Make sure evaluate_contact_map.py ran successfully."
        )

    preds = np.load(pred_path)
    targets = np.load(target_path)

    # Ensure shape [N, 40, 40]
    if preds.ndim == 2:
        preds = preds.reshape(len(preds), 40, 40)
        targets = targets.reshape(len(targets), 40, 40)

    n_samples = preds.shape[0]
    print(f"Loaded {n_samples} samples from {os.path.dirname(pred_path)}")

    # Compute PCC for each sample
    pcc_scores = []
    for i in range(n_samples):
        pred_flat = preds[i].flatten()
        targ_flat = targets[i].flatten()
        if np.std(pred_flat) == 0 or np.std(targ_flat) == 0:
            pcc = 0.0
        else:
            pcc, _ = pearsonr(pred_flat, targ_flat)
        pcc_scores.append(pcc)

    pcc_scores = np.array(pcc_scores)
    avg_pcc = np.mean(pcc_scores)
    std_pcc = np.std(pcc_scores)

    print(f"\nAverage PCC across all samples: {avg_pcc:.4f} ± {std_pcc:.4f}")

    # Find representative example near mean PCC
    idx = np.argmin(np.abs(pcc_scores - avg_pcc))
    example_pred = preds[idx]
    example_true = targets[idx]
    example_pcc = pcc_scores[idx]

    print(f"Representative example index: {idx} (PCC = {example_pcc:.4f})")

    # Save numerical results
    metrics_path = os.path.join(output_dir, "performance_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Average PCC: {avg_pcc:.4f}\n")
        f.write(f"Std PCC: {std_pcc:.4f}\n")
        f.write(f"Representative Example Index: {idx}\n")
        f.write(f"Representative PCC: {example_pcc:.4f}\n")
    print(f"Saved metrics to {metrics_path}")

    # Visualization
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(example_true, cmap="viridis", origin="lower")
    plt.title(f"Ground Truth\n(Index {idx})")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.imshow(example_pred, cmap="viridis", origin="lower")
    plt.title(f"Prediction\nPCC = {example_pcc:.4f}")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.suptitle(f"Evo2 Contact Map Prediction Performance\nAverage PCC = {avg_pcc:.4f}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = os.path.join(output_dir, "representative_contact_map.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Saved representative visualization to: {fig_path}")

    return avg_pcc, std_pcc, pcc_scores

if __name__ == "__main__":
    analyze_performance()
