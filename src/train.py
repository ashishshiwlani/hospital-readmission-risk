"""
Training script for hospital readmission risk predictor.

End-to-end workflow:
  1. Generate or load synthetic MIMIC-III dataset
  2. Extract tabular and NLP features
  3. Train/validation/test split (stratified for class balance)
  4. Fit readmission risk model
  5. Evaluate metrics (AUC, precision, recall, F1, Brier score)
  6. Save model and generate evaluation plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

from src.synthetic_data import generate_dataset
from src.feature_engineering import extract_tabular_features
from src.model import ReadmissionRiskModel


def evaluate_model(
    model: ReadmissionRiskModel,
    X_tab_test: np.ndarray,
    X_text_test: list,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate trained model on test set.

    Computes AUC-ROC, precision, recall, F1, Brier score (calibration),
    and other clinically relevant metrics.

    Args:
        model: Trained ReadmissionRiskModel instance.
        X_tab_test: Test tabular features.
        X_text_test: Test discharge notes.
        y_test: True labels for test set.

    Returns:
        Dictionary of metrics:
            - auc_roc: Area under ROC curve
            - precision: True positives / (TP + FP)
            - recall: True positives / (TP + FN)
            - f1: Harmonic mean of precision and recall
            - brier_score: Mean squared error of probability estimates
            - specificity: True negatives / (TN + FP)
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X_tab_test, X_text_test)
    y_pred = model.predict(X_tab_test, X_text_test, threshold=0.5)

    # Compute metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_pred_proba[:, 1])

    # Specificity: true negatives / (TN + FP)
    # Measures how well model identifies non-readmission cases
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,
        "specificity": specificity,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    output_path: str,
) -> None:
    """
    Plot and save ROC curve.

    The ROC curve visualizes the trade-off between true positive rate (recall)
    and false positive rate (1-specificity) across all probability thresholds.
    AUC summarizes this into a single score (0.5 = random, 1.0 = perfect).

    Args:
        fpr: False positive rates array.
        tpr: True positive rates array.
        auc: Area under the ROC curve.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hospital Readmission Risk")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"ROC curve saved to {output_path}")
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    n_bins: int = 10,
) -> float:
    """
    Plot and save a reliability diagram (calibration curve) with ECE annotation.

    A well-calibrated model has fraction_of_positives ≈ mean_predicted_value at
    every probability level.  The diagonal dashed line represents perfect calibration.

    Expected Calibration Error (ECE) is the weighted mean of |predicted – actual|
    across bins.  Lower is better; ECE < 0.05 is generally considered well-calibrated.

    Why this matters clinically:
      - If a model says "60% readmission risk" it should be right ~60% of the time.
      - Poor calibration means the raw probability output cannot be trusted directly.
      - Can be corrected post-hoc with Platt scaling or isotonic regression.

    Args:
        y_true:      Ground-truth binary labels.
        y_prob:      Predicted probabilities for the positive class.
        output_path: Path to save the PNG.
        n_bins:      Number of calibration bins (default: 10).

    Returns:
        ECE (float) — lower is better.
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error: weighted average of |gap| per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        bin_counts[i] = mask.sum()

    # Align bin_counts length with calibration curve output (sklearn may drop empty bins)
    n_points = len(mean_predicted)
    if n_points < n_bins:
        # Recompute counts for the bins that had data
        active_counts = np.array([
            ((y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])).sum()
            for i in range(n_bins)
            if ((y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])).sum() > 0
        ], dtype=float)
        if len(active_counts) == n_points:
            bin_counts = active_counts

    weights = bin_counts[:n_points] / max(bin_counts[:n_points].sum(), 1)
    ece = float(np.sum(weights * np.abs(fraction_of_positives - mean_predicted)))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        mean_predicted, fraction_of_positives,
        "s-", color="#3498db", linewidth=2.5, markersize=7,
        label=f"Readmission model (ECE = {ece:.3f})",
    )
    ax.plot(
        [0, 1], [0, 1],
        "k--", linewidth=1.5, alpha=0.6,
        label="Perfect calibration",
    )

    # Shade calibration gap
    ax.fill_between(
        mean_predicted,
        mean_predicted,          # perfect diagonal
        fraction_of_positives,   # model curve
        alpha=0.15, color="#3498db",
        label="Calibration gap",
    )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Observed Frequency)", fontsize=12)
    ax.set_title("Reliability Diagram — 30-Day Readmission Model", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.10])

    # ECE annotation box
    ax.text(
        0.98, 0.05,
        f"ECE = {ece:.3f}\n({'Well calibrated ✓' if ece < 0.05 else 'Needs calibration'})",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf4ff", edgecolor="#3498db", alpha=0.9),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Calibration curve saved to {output_path}  (ECE = {ece:.3f})")
    return ece


def plot_feature_importance(
    importances: Dict[int, float],
    n_features: int,
    output_path: str,
) -> None:
    """
    Plot and save top-20 feature importance bar chart.

    Feature importance (gain) represents the total decrease in loss (gini impurity
    for classification) when splitting on that feature across all trees.

    Args:
        importances: Dictionary mapping feature index → importance score.
        n_features: Total number of features (for context).
        output_path: Path to save the plot.
    """
    # Sort by importance and take top 20
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_k = min(20, len(sorted_features))
    top_features = sorted_features[:top_k]

    feature_indices = [f[0] for f in top_features]
    feature_scores = [f[1] for f in top_features]
    feature_labels = [f"Feature {i}" for i in feature_indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_labels)), feature_scores, color="steelblue")
    plt.yticks(range(len(feature_labels)), feature_labels)
    plt.xlabel("XGBoost Feature Importance (Gain)")
    plt.title(f"Top {top_k} Features for Readmission Prediction")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Feature importance plot saved to {output_path}")
    plt.close()


def main():
    """
    Main training pipeline.

    1. Parse command-line arguments
    2. Generate or load synthetic dataset
    3. Extract features (tabular + NLP)
    4. Perform stratified train/val/test split
    5. Train model
    6. Evaluate and save results
    """
    parser = argparse.ArgumentParser(
        description="Train hospital readmission risk predictor"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory for synthetic data (default: ./data)",
    )
    parser.add_argument(
        "--n_patients",
        type=int,
        default=5000,
        help="Number of synthetic patients to generate (default: 5000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save model and plots (default: ./models)",
    )
    parser.add_argument(
        "--use_bert",
        action="store_true",
        help="Use Bio_ClinicalBERT embeddings (slower but higher quality)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    # Check if data exists; otherwise generate
    data_path = os.path.join(args.data_dir, "synthetic_patients.csv")
    if os.path.exists(data_path):
        print(f"Loading existing dataset from {data_path}...")
        df = pd.read_csv(data_path)
        # Reconstruct list columns from CSV (pandas limitation)
        df["diagnosis_codes"] = df["diagnosis_codes"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
        df["lab_values"] = df["lab_values"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )
    else:
        print(f"Generating {args.n_patients} synthetic patients...")
        df = generate_dataset(n_patients=args.n_patients, random_seed=args.random_seed)
        # Save for future reference (note: diagnosis_codes and lab_values are serialized)
        df.to_csv(data_path, index=False)
        print(f"Dataset saved to {data_path}")

    print(f"Dataset shape: {df.shape}")
    print(f"Readmission rate: {df['readmitted_30d'].mean():.1%}")

    # Extract features
    print("\nExtracting tabular features...")
    X_tabular, feature_names = extract_tabular_features(df)
    X_text = df["discharge_note"].tolist()
    y = df["readmitted_30d"].values

    print(f"Tabular features shape: {X_tabular.shape}")
    print(f"Text features count: {len(X_text)}")

    # Stratified train/test split (preserves class distribution)
    print("\nPerforming stratified train/test split...")
    X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = (
        train_test_split(
            X_tabular,
            X_text,
            y,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=y,  # Maintain class balance in splits
        )
    )

    print(f"Train set size: {len(y_train)} (readmission rate: {y_train.mean():.1%})")
    print(f"Test set size: {len(y_test)} (readmission rate: {y_test.mean():.1%})")

    # Train model
    print("\nTraining readmission risk model...")
    model = ReadmissionRiskModel(use_bert=args.use_bert)
    model.fit(X_tab_train, X_text_train, y_train)

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, X_tab_test, X_text_test, y_test)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"AUC-ROC:        {metrics['auc_roc']:.3f}")
    print(f"Precision:      {metrics['precision']:.3f}")
    print(f"Recall:         {metrics['recall']:.3f}")
    print(f"Specificity:    {metrics['specificity']:.3f}")
    print(f"F1 Score:       {metrics['f1']:.3f}")
    print(f"Brier Score:    {metrics['brier_score']:.3f}")
    print("=" * 60)

    # Save model
    model_path = os.path.join(args.output_dir, "model.pkl")
    model.save(model_path)

    # Generate ROC curve
    y_pred_proba = model.predict_proba(X_tab_test, X_text_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plot_roc_curve(fpr, tpr, metrics["auc_roc"], roc_path)

    # Generate feature importance plot
    importances = model.get_feature_importance()
    importance_path = os.path.join(args.output_dir, "feature_importance.png")
    plot_feature_importance(importances, len(feature_names), importance_path)

    # Generate calibration curve + ECE
    calib_path = os.path.join(args.output_dir, "calibration_curve.png")
    ece = plot_calibration_curve(y_test, y_pred_proba[:, 1], calib_path)
    metrics["ece"] = ece

    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    print(f"\nTraining complete! Model saved to {model_path}")


if __name__ == "__main__":
    main()
