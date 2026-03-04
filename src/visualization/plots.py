from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    """
    Save a confusion matrix heatmap-like image without seaborn.
    cm should be shape (2,2).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Normal", "Anomaly"])
    plt.yticks([0, 1], ["Normal", "Anomaly"])

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, pr_auc: float, save_path: Path):
    """
    Save Precision-Recall curve.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve (PR-AUC={pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_probability_distribution(y_prob: np.ndarray, y_true: np.ndarray, save_path: Path, bins: int = 50):
    """
    Plot predicted anomaly probability distribution for normal vs anomaly classes.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)

    plt.figure()
    plt.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, density=True, label="Normal")
    plt.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, density=True, label="Anomaly")
    plt.title("Predicted Anomaly Probability Distribution (Test)")
    plt.xlabel("P(Anomaly)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()