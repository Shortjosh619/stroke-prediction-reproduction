from model_training import get_models, build_pipeline, load_data, DATA_PATH
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import confusion_matrix

RESULTS_PATH = Path(r"C:\stroke-prediction-reproduction\results\threshold_analysis")

def log_reg_pipeline(data_path):

    X_train, X_test, y_train, y_test = load_data(data_path)

    pipe = build_pipeline(get_models()["LogReg"])
    pipe.fit(X_train, y_train)
    y_preds = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]

    return y_test, y_proba


def metrics_per_threshold(y_test, y_proba, thresholds):
    results = []

    for pt in thresholds: #pt for threshold probability
        y_pred = (y_proba >= pt).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": pt,
            "recall": recall,
            "specificity": specificity,
            "precision": precision,
            "f1": f1
        })

    return pd.DataFrame(results)



def plot_threshold_analysis(metrics_df, y_test, y_proba, optimal_threshold, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Recall and specificity vs threshold
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics_df["threshold"], metrics_df["recall"], label="Recall (Sensitivity)")
    ax.plot(metrics_df["threshold"], metrics_df["specificity"], label="Specificity")
    ax.axvline(optimal_threshold, linestyle="--", linewidth=1, label=f"Optimal threshold = {optimal_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Recall vs Specificity across Thresholds (Logistic Regression)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "recall_specificity_vs_threshold.png", dpi=300)
    plt.close(fig)

    # Plot 2: Precision-Recall curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics_df["recall"], metrics_df["precision"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Logistic Regression)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "precision_recall_curve.png", dpi=300)
    plt.close(fig)

    # Plot 3: Confusion matrix at optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_optimal, ax=ax)
    ax.set_title(f"Confusion Matrix at Threshold = {optimal_threshold:.2f}")
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix_optimal_threshold.png", dpi=300)
    plt.close(fig)

    print(f"Plots saved to {save_dir.resolve()}")


def main():
    print("Fitting Logistic Regression pipeline...")
    y_test, y_proba = log_reg_pipeline(DATA_PATH)

    thresholds = np.arange(0.05, 0.95, 0.01)

    print("Computing metrics across thresholds...")
    metrics_df = metrics_per_threshold(y_test, y_proba, thresholds)

    # Select optimal threshold: lowest threshold where recall >= 0.90
    high_recall = metrics_df[metrics_df["recall"] >= 0.90]
    if not high_recall.empty:
        optimal_threshold = float(high_recall["threshold"].iloc[0])
    else:
        print("Warning: recall >= 0.90 not achievable. Using max recall threshold.")
        optimal_threshold = float(metrics_df.loc[metrics_df["recall"].idxmax(), "threshold"])

    print(f"\nOptimal threshold (recall >= 0.90): {optimal_threshold:.2f}")
    optimal_row = metrics_df[metrics_df["threshold"].round(2) == round(optimal_threshold, 2)]
    print(optimal_row.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_threshold_analysis(metrics_df, y_test, y_proba, optimal_threshold, RESULTS_PATH)

    metrics_df.to_csv(RESULTS_PATH / "threshold_metrics.csv", index=False)
    print(f"\nCSV saved to {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()