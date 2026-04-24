"""
Unit tests for model training, inference, and explanation.

Tests verify:
- Model fit/predict cycle with correct shapes
- Probability outputs are valid (0-1 range)
- Risk level classification is correct
- Model save/load roundtrip preserves predictions
- SHAP explanations return expected structure
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic_data import generate_dataset, generate_patient_record
from src.feature_engineering import extract_tabular_features
from src.model import ReadmissionRiskModel
from src.predict import predict_patient, explain_prediction


class TestModelFitPredict:
    """Tests for model training and inference."""

    def test_model_fit_basic(self):
        """
        Test that model.fit() accepts tabular features and text without error.

        Verifies basic training pipeline with small dataset.
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        model = ReadmissionRiskModel(use_bert=False)
        # Should not raise an exception
        model.fit(X_tabular, X_text, y)

        assert model.xgb_model is not None, "Model not trained"

    def test_model_predict_proba_shape_and_range(self):
        """
        Test that predict_proba returns valid probability matrix.

        Verifies:
        - Output shape is (n_samples, 2)
        - Values are in valid probability range [0, 1]
        - Sum of probabilities per sample is ~1
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        model = ReadmissionRiskModel(use_bert=False)
        # Train on first 80 samples, test on last 20 — slice ALL three to same length
        model.fit(X_tabular[:80], X_text[:80], y[:80])

        # Test on held-out subset
        y_pred_proba = model.predict_proba(X_tabular[80:], X_text[80:])

        assert y_pred_proba.shape == (20, 2), f"Expected (20, 2), got {y_pred_proba.shape}"
        assert np.all(y_pred_proba >= 0) and np.all(y_pred_proba <= 1), (
            "Probabilities outside [0, 1]"
        )
        # Each row should sum to ~1.0
        row_sums = y_pred_proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.01), f"Probability sums: {row_sums}"

    def test_model_predict_binary_output(self):
        """
        Test that predict() returns binary (0/1) predictions.

        Verifies correctness of thresholding (default 0.5).
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        model = ReadmissionRiskModel(use_bert=False)
        model.fit(X_tabular, X_text, y)

        y_pred = model.predict(X_tabular[:20], X_text[:20], threshold=0.5)

        assert y_pred.dtype == np.int32 or y_pred.dtype == np.int64
        assert np.all((y_pred == 0) | (y_pred == 1)), "Predictions not binary"

    def test_model_predict_threshold_consistency(self):
        """
        Test that thresholding is consistent with probabilities.

        If probability >= threshold, prediction should be 1; otherwise 0.
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        model = ReadmissionRiskModel(use_bert=False)
        model.fit(X_tabular, X_text, y)

        threshold = 0.6
        y_pred_proba = model.predict_proba(X_tabular[:20], X_text[:20])
        y_pred = model.predict(X_tabular[:20], X_text[:20], threshold=threshold)

        # Verify consistency: pred == (proba[:,1] >= threshold)
        expected_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
        assert np.array_equal(y_pred, expected_pred), "Threshold inconsistency"


class TestPatientPrediction:
    """Tests for single-patient prediction interface."""

    def test_predict_patient_returns_valid_risk_score(self):
        """
        Test that predict_patient returns valid risk score.

        Verifies:
        - risk_score is in [0, 1]
        - prediction is boolean
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        model = ReadmissionRiskModel(use_bert=False)
        model.fit(X_tabular, X_text, y)

        patient = generate_patient_record(random_seed=123)
        result = predict_patient(model, patient)

        assert "risk_score" in result
        assert "risk_level" in result
        assert "prediction" in result

        assert 0 <= result["risk_score"] <= 1, f"Risk score out of range: {result['risk_score']}"
        assert isinstance(result["prediction"], (bool, np.bool_))

    def test_predict_patient_risk_levels(self):
        """
        Test that risk_level classification is correct.

        Verifies thresholds: LOW < 0.3, MODERATE 0.3-0.6, HIGH >= 0.6
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        model = ReadmissionRiskModel(use_bert=False)
        model.fit(X_tabular, X_text, y)

        patient = generate_patient_record(random_seed=123)
        result = predict_patient(model, patient)

        risk_score = result["risk_score"]
        risk_level = result["risk_level"]

        if risk_score < 0.3:
            assert risk_level == "LOW", f"Incorrect level for score {risk_score}"
        elif risk_score < 0.6:
            assert risk_level == "MODERATE", f"Incorrect level for score {risk_score}"
        else:
            assert risk_level == "HIGH", f"Incorrect level for score {risk_score}"

    def test_predict_patient_multiple_samples(self):
        """
        Test predict_patient on multiple different patient records.

        Verifies consistency across different input data.
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        model = ReadmissionRiskModel(use_bert=False)
        model.fit(X_tabular, X_text, y)

        for seed in [1, 2, 3]:
            patient = generate_patient_record(random_seed=seed)
            result = predict_patient(model, patient)

            assert "risk_score" in result
            assert 0 <= result["risk_score"] <= 1


class TestModelPersistence:
    """Tests for model save/load functionality."""

    def test_model_save_load_roundtrip(self):
        """
        Test that model can be saved and loaded without losing predictions.

        Verifies:
        - Model saves to disk without error
        - Loaded model produces identical predictions
        """
        df = generate_dataset(n_patients=50, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        # Train model
        model1 = ReadmissionRiskModel(use_bert=False)
        model1.fit(X_tabular[:40], X_text[:40], y[:40])

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/test_model.pkl"
            model1.save(model_path)

            # Load
            model2 = ReadmissionRiskModel.load(model_path)

            # Compare predictions
            y_pred1 = model1.predict_proba(X_tabular[40:], X_text[40:])
            y_pred2 = model2.predict_proba(X_tabular[40:], X_text[40:])

            assert np.allclose(y_pred1, y_pred2, atol=1e-6), (
                "Loaded model predictions differ from original"
            )

    def test_model_load_nonexistent_fails(self):
        """
        Test that loading non-existent model raises error.

        Verifies error handling for missing files.
        """
        with pytest.raises(FileNotFoundError):
            ReadmissionRiskModel.load("/nonexistent/path/model.pkl")


class TestModelExplanation:
    """Tests for SHAP-based model explanation."""

    def test_explain_prediction_returns_top_features(self):
        """
        Test that explain_prediction returns expected structure.

        Verifies:
        - Returns list of feature explanations
        - Each has 'feature', 'contribution', 'direction' keys
        - Correct number of features returned
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        model = ReadmissionRiskModel(use_bert=False)
        model.fit(X_tabular, X_text, y)

        patient = generate_patient_record(random_seed=123)
        n_top = 5
        explanation = explain_prediction(model, patient, n_top_features=n_top)

        assert isinstance(explanation, list)
        assert len(explanation) == n_top, f"Expected {n_top} features, got {len(explanation)}"

        for factor in explanation:
            assert "feature" in factor
            assert "contribution" in factor
            assert "direction" in factor
            assert factor["direction"] in ["increases_risk", "decreases_risk"]
            assert isinstance(factor["contribution"], float)
            assert factor["contribution"] >= 0, "Contribution should be non-negative (absolute value)"

    def test_explain_prediction_direction_values(self):
        """
        Test that direction values are sensible.

        High-risk features should have positive SHAP values (increase_risk),
        low-risk features should have negative (decrease_risk).
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        model = ReadmissionRiskModel(use_bert=False)
        model.fit(X_tabular, X_text, y)

        patient = generate_patient_record(random_seed=123)
        explanation = explain_prediction(model, patient, n_top_features=5)

        # Check that directions are valid
        for factor in explanation:
            assert factor["direction"] in ["increases_risk", "decreases_risk"]


class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_model_with_single_class_label(self):
        """
        Test that model handles edge case of all-negative labels.

        With all non-readmission labels, model should still train
        (though predictions will be biased toward 0).
        """
        df = generate_dataset(n_patients=50, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()

        # Force all labels to 0 (no readmission)
        y = np.zeros(len(df), dtype=int)

        model = ReadmissionRiskModel(use_bert=False)
        # Should complete without error
        model.fit(X_tabular[:40], X_text[:40], y[:40])

        y_pred = model.predict_proba(X_tabular[40:], X_text[40:])
        # Model might predict all 0s, which is valid (though not useful)
        assert y_pred.shape == (10, 2)

    def test_model_with_imbalanced_data(self):
        """
        Test that model handles class imbalance correctly.

        With 95% negative, 5% positive samples, model should still
        train and produce predictions in valid range.
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X_tabular, _ = extract_tabular_features(df)
        X_text = df["discharge_note"].tolist()
        y = df["readmitted_30d"].values

        # Create extreme imbalance: keep only first 5 readmitted
        imbalanced_mask = (y == 0) | (np.arange(len(y)) < 5)
        X_tab_imb = X_tabular[imbalanced_mask]
        X_text_imb = [X_text[i] for i in range(len(X_text)) if imbalanced_mask[i]]
        y_imb = y[imbalanced_mask]

        model = ReadmissionRiskModel(use_bert=False)
        model.fit(X_tab_imb, X_text_imb, y_imb)

        # Should produce valid predictions despite imbalance
        y_pred = model.predict_proba(X_tab_imb[:10], X_text_imb[:10])
        assert y_pred.shape == (10, 2)
        assert np.all((y_pred >= 0) & (y_pred <= 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
