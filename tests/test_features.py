"""
Unit tests for feature engineering module.

Tests verify:
- Synthetic data generation produces valid datasets
- Readmission label correlates with risk factors
- Feature extraction handles edge cases (missing values)
- TF-IDF keyword extraction works on clinical text
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic_data import generate_dataset, generate_patient_record
from src.feature_engineering import extract_tabular_features
from src.nlp_features import extract_keyword_features


class TestSyntheticDataGeneration:
    """Tests for synthetic MIMIC-III data generator."""

    def test_generate_dataset_shape(self):
        """
        Test that generate_dataset produces correct DataFrame shape.

        Verifies:
        - n_patients rows are generated
        - All expected columns are present
        - No duplicate patient IDs
        """
        n_patients = 100
        df = generate_dataset(n_patients=n_patients, random_seed=42)

        assert df.shape[0] == n_patients, f"Expected {n_patients} rows, got {df.shape[0]}"

        expected_cols = {
            "patient_id",
            "age",
            "gender",
            "admission_type",
            "insurance",
            "diagnosis_codes",
            "num_procedures",
            "num_medications",
            "lab_values",
            "prior_admissions_12mo",
            "length_of_stay_days",
            "discharge_note",
            "readmitted_30d",
        }
        actual_cols = set(df.columns)
        assert expected_cols == actual_cols, f"Column mismatch: {expected_cols ^ actual_cols}"

        # Check for duplicates
        assert df["patient_id"].nunique() == n_patients, "Duplicate patient IDs found"

    def test_readmitted_label_correlated_with_age(self):
        """
        Test that readmission probability increases with age.

        Validates that the synthetic label is NOT random but correlated
        with risk factors. Elderly patients (>75) should have higher
        readmission rate than young patients (<40).
        """
        n_patients = 1000
        df = generate_dataset(n_patients=n_patients, random_seed=42)

        elderly = df[df["age"] > 75]
        young = df[df["age"] < 40]

        elderly_readmit_rate = elderly["readmitted_30d"].mean()
        young_readmit_rate = young["readmitted_30d"].mean()

        # Elderly should have higher readmission rate
        assert (
            elderly_readmit_rate > young_readmit_rate
        ), f"Age correlation broken: elderly {elderly_readmit_rate:.2%} "
        f"vs young {young_readmit_rate:.2%}"

    def test_readmitted_label_correlated_with_prior_admissions(self):
        """
        Test that readmission rate increases with prior admissions.

        Patients with multiple prior admissions should have higher
        readmission risk (strong predictor in healthcare literature).
        """
        n_patients = 1000
        df = generate_dataset(n_patients=n_patients, random_seed=42)

        no_prior = df[df["prior_admissions_12mo"] == 0]
        many_prior = df[df["prior_admissions_12mo"] >= 3]

        no_prior_readmit = no_prior["readmitted_30d"].mean()
        many_prior_readmit = many_prior["readmitted_30d"].mean()

        assert (
            many_prior_readmit > no_prior_readmit
        ), f"Prior admissions correlation broken: "
        f"many_prior {many_prior_readmit:.2%} vs no_prior {no_prior_readmit:.2%}"


class TestFeatureEngineering:
    """Tests for tabular feature extraction."""

    def test_extract_tabular_features_no_nans(self):
        """
        Test that extract_tabular_features handles missing values.

        Verifies that output feature matrix contains no NaN values,
        even if input data had missing lab values.
        """
        df = generate_dataset(n_patients=50, random_seed=42)
        X, feature_names = extract_tabular_features(df)

        assert X.shape[0] == 50, f"Expected 50 samples, got {X.shape[0]}"
        assert not np.any(np.isnan(X)), "NaN values found in feature matrix"
        assert len(feature_names) == X.shape[1], "Feature name count mismatch"

    def test_extract_features_consistent_output_shape(self):
        """
        Test that feature extraction produces consistent output shapes.

        Multiple calls with different random seeds should produce
        the same feature matrix dimensions.
        """
        X1, names1 = extract_tabular_features(generate_dataset(n_patients=100, random_seed=1))
        X2, names2 = extract_tabular_features(generate_dataset(n_patients=100, random_seed=2))

        assert X1.shape[1] == X2.shape[1], "Feature dimension mismatch"
        assert names1 == names2, "Feature names don't match"

    def test_features_are_numeric(self):
        """
        Test that all extracted features are numeric (float/int).

        Verifies that no categorical or object types leak into the
        feature matrix (should be preprocessed).
        """
        df = generate_dataset(n_patients=50, random_seed=42)
        X, _ = extract_tabular_features(df)

        assert X.dtype in [np.float32, np.float64], f"Unexpected dtype: {X.dtype}"

    def test_normalized_features_have_reasonable_range(self):
        """
        Test that normalized features are in reasonable range.

        After standardization (StandardScaler), features should have
        mean ~0 and std ~1 (most values within [-3, 3]).
        """
        df = generate_dataset(n_patients=100, random_seed=42)
        X, _ = extract_tabular_features(df)

        # Most values should be within 3 standard deviations
        assert np.max(np.abs(X)) < 10, f"Features outside reasonable range: {np.max(np.abs(X))}"


class TestNLPFeatures:
    """Tests for clinical NLP feature extraction."""

    def test_tfidf_keyword_features_shape(self):
        """
        Test TF-IDF keyword feature extraction shape and non-sparsity.

        Verifies that output is dense (not sparse) and matches
        expected dimensions (n_notes, vocab_size).
        """
        notes = [
            "Patient with pneumonia and respiratory distress. "
            "Readmit concerns noted.",
            "Routine follow-up. Stable. No concerns.",
            "Unstable patient. Monitor closely for infection.",
        ]

        X = extract_keyword_features(notes)

        assert X.shape[0] == len(notes), f"Expected {len(notes)} rows, got {X.shape[0]}"
        assert X.shape[1] > 0, "No features extracted"
        assert X.dtype in [np.float32, np.float64], f"Unexpected dtype: {X.dtype}"
        # TF-IDF should have non-zero values for relevant terms
        assert np.any(X > 0), "All TF-IDF values are zero"

    def test_tfidf_consistent_vocab_size(self):
        """
        Test that TF-IDF produces consistent vocabulary size.

        Multiple calls should use the same vocabulary size,
        enabling model retraining consistency.
        """
        notes1 = ["Patient with pneumonia", "Follow-up needed"]
        notes2 = ["Unstable condition", "Readmit risk high"]

        X1 = extract_keyword_features(notes1)
        X2 = extract_keyword_features(notes2)

        # Should have same feature count (fixed vocabulary)
        assert X1.shape[1] == X2.shape[1], "Vocabulary size mismatch"

    def test_keyword_features_detect_clinical_terms(self):
        """
        Test that keyword extractor detects expected clinical terms.

        Notes containing high-signal terms like 'readmit', 'unstable',
        'poorly controlled' should produce non-zero features.
        """
        note_with_signal = (
            "Patient at high risk for readmission. Unstable vital signs. "
            "Poorly controlled diabetes."
        )
        note_without_signal = "Patient stable. Routine discharge. No issues."

        X_with = extract_keyword_features([note_with_signal])
        X_without = extract_keyword_features([note_without_signal])

        # Note with clinical terms should have higher total TF-IDF
        assert np.sum(X_with) > np.sum(X_without), (
            "High-signal note should have higher TF-IDF sum"
        )


class TestIntegration:
    """Integration tests combining data generation, features, and predictions."""

    def test_full_feature_pipeline(self):
        """
        Test end-to-end feature extraction pipeline.

        Generates dataset, extracts features, verifies compatibility
        with ML models (correct shapes, no NaNs, numeric types).
        """
        df = generate_dataset(n_patients=50, random_seed=42)
        X_tabular, feature_names = extract_tabular_features(df)
        X_text = extract_keyword_features(df["discharge_note"].tolist())

        # Verify shapes
        assert X_tabular.shape[0] == 50
        assert X_text.shape[0] == 50

        # Verify no missing values
        assert not np.any(np.isnan(X_tabular))
        assert not np.any(np.isnan(X_text))

        # Verify numeric types
        assert X_tabular.dtype in [np.float32, np.float64]
        assert X_text.dtype in [np.float32, np.float64]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
