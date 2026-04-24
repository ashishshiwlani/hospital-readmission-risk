"""
Readmission risk model: XGBoost late-fusion classifier combining tabular and NLP features.

Late fusion architecture:
  Tabular features (30-50 dims) + NLP embeddings (768 dims from BERT or 20 dims from TF-IDF)
  → concatenate → XGBoost classifier

Why late fusion?
  - Allows separate optimization of tabular and NLP feature extractors
  - XGBoost naturally handles high-dimensional embeddings
  - Interpretable feature importance across both modalities
"""

import pickle
from typing import Tuple
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)

from src.feature_engineering import extract_tabular_features
from src.nlp_features import extract_keyword_features


class ReadmissionRiskModel:
    """
    Readmission risk predictor combining tabular and NLP features.

    Architecture:
      1. Fit sklearn feature engineering pipeline on tabular data
      2. (Optional) Initialize and fit Bio_ClinicalBERT embedder on clinical notes
      3. Concatenate [tabular_features || text_embeddings]
      4. Train XGBoost classifier on fused feature matrix

    Attributes:
        use_bert (bool): Whether to use BERT embeddings (True) or TF-IDF (False)
        bert_embedder: Initialized ClinicalNoteEmbedder (if use_bert=True)
        xgb_model: Trained XGBClassifier
        feature_names (List[str]): Names of tabular features for SHAP
        n_tabular_features (int): Dimension of tabular feature space
    """

    def __init__(
        self,
        use_bert: bool = False,
        n_tabular_features: int | None = None,
    ):
        """
        Initialize the readmission risk model.

        Args:
            use_bert: If True, use Bio_ClinicalBERT embeddings. Otherwise, use TF-IDF.
            n_tabular_features: Expected dimension of tabular features (optional,
                for validation during inference).
        """
        self.use_bert = use_bert
        self.n_tabular_features = n_tabular_features
        self.bert_embedder = None
        self.xgb_model = None
        self.feature_names = None

        # If BERT is requested, initialize the embedder
        # (actual model loading is deferred to fit() to allow CPU training without GPU)
        if self.use_bert:
            try:
                from nlp_features import ClinicalNoteEmbedder

                self.bert_embedder = None  # Lazy initialization in fit()
            except ImportError:
                raise ImportError(
                    "Bio_ClinicalBERT requires torch and transformers. "
                    "Install with: pip install torch transformers"
                )

    def fit(
        self,
        X_tabular: np.ndarray,
        X_text: list[str],
        y: np.ndarray,
    ) -> None:
        """
        Train the fusion model on tabular features and clinical notes.

        Steps:
          1. Ensure tabular features are extracted (or already provided as array)
          2. Extract text embeddings (BERT or TF-IDF)
          3. Concatenate features
          4. Train XGBoost with class weights for imbalance handling

        Args:
            X_tabular: Tabular feature matrix (n_samples, n_tabular_features).
                Can also be a DataFrame (will be converted to array).
            X_text: List of discharge note strings (n_samples,).
            y: Binary target labels (n_samples,) indicating readmission (0/1).
        """
        # Convert DataFrame to array if needed
        if hasattr(X_tabular, "values"):
            X_tabular = X_tabular.values

        self.n_tabular_features = X_tabular.shape[1]

        print(f"Extracting {len(X_text)} clinical note embeddings...")

        # Extract text embeddings based on use_bert flag
        if self.use_bert:
            from nlp_features import ClinicalNoteEmbedder

            if self.bert_embedder is None:
                # Lazy initialization on first fit
                self.bert_embedder = ClinicalNoteEmbedder()
            X_text_features = self.bert_embedder.embed_notes(X_text)
            print(f"BERT embeddings shape: {X_text_features.shape}")
        else:
            # Use lightweight TF-IDF features
            X_text_features = extract_keyword_features(X_text)
            print(f"TF-IDF features shape: {X_text_features.shape}")

        # Concatenate tabular and text features for late fusion
        X_fused = np.concatenate([X_tabular, X_text_features], axis=1)
        print(f"Fused feature matrix shape: {X_fused.shape}")

        # Train XGBoost classifier with class weight balancing
        # scale_pos_weight addresses class imbalance (~18% readmission rate)
        # By upweighting minority class, we improve recall without explicit threshold tuning
        n_readmitted = y.sum()
        n_non_readmitted = len(y) - n_readmitted
        # Guard against zero-division when all labels are the same class (edge case in tests)
        scale_pos_weight = (n_non_readmitted / n_readmitted) if n_readmitted > 0 else 1.0

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        print(
            f"Training XGBoost with scale_pos_weight={scale_pos_weight:.2f} "
            f"(positive class upweighting for imbalance)"
        )
        self.xgb_model.fit(X_fused, y)
        print("Model training complete")

    def predict_proba(
        self,
        X_tabular: np.ndarray,
        X_text: list[str],
    ) -> np.ndarray:
        """
        Predict readmission probabilities for new patients.

        Args:
            X_tabular: Tabular feature matrix (n_samples, n_tabular_features).
            X_text: List of discharge note strings.

        Returns:
            np.ndarray of shape (n_samples, 2) with columns [P(no readmit), P(readmit)].
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Convert DataFrame to array if needed
        if hasattr(X_tabular, "values"):
            X_tabular = X_tabular.values

        # Extract text features with same embedder as training
        if self.use_bert:
            X_text_features = self.bert_embedder.embed_notes(X_text)
        else:
            X_text_features = extract_keyword_features(X_text)

        # Late fusion: concatenate features
        X_fused = np.concatenate([X_tabular, X_text_features], axis=1)

        return self.xgb_model.predict_proba(X_fused)

    def predict(
        self,
        X_tabular: np.ndarray,
        X_text: list[str],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict binary readmission labels for new patients.

        Args:
            X_tabular: Tabular feature matrix.
            X_text: List of discharge notes.
            threshold: Probability threshold for positive class (default 0.5).
                Adjust higher to improve precision, lower to improve recall.

        Returns:
            np.ndarray of shape (n_samples,) with binary predictions (0/1).
        """
        proba = self.predict_proba(X_tabular, X_text)
        return (proba[:, 1] >= threshold).astype(int)

    def get_feature_importance(self) -> dict:
        """
        Get feature importance scores from the trained XGBoost model.

        Returns:
            Dictionary mapping feature index → importance score.
            Higher scores indicate greater influence on predictions.
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        importances = self.xgb_model.feature_importances_
        return {i: float(importances[i]) for i in range(len(importances))}

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Serializes the XGBoost model and BERT embedder (if used).

        Args:
            path: File path to save to (typically .pkl extension).
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        model_state = {
            "xgb_model": self.xgb_model,
            "bert_embedder": self.bert_embedder,
            "use_bert": self.use_bert,
            "n_tabular_features": self.n_tabular_features,
        }

        with open(path, "wb") as f:
            pickle.dump(model_state, f)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ReadmissionRiskModel":
        """
        Load a trained model from disk.

        Args:
            path: File path to load from.

        Returns:
            ReadmissionRiskModel instance with loaded weights.
        """
        with open(path, "rb") as f:
            model_state = pickle.load(f)

        model = cls(
            use_bert=model_state["use_bert"],
            n_tabular_features=model_state["n_tabular_features"],
        )
        model.xgb_model = model_state["xgb_model"]
        model.bert_embedder = model_state["bert_embedder"]

        print(f"Model loaded from {path}")
        return model


def build_xgboost_classifier() -> xgb.XGBClassifier:
    """
    Build a configured XGBoost classifier for readmission prediction.

    Hyperparameter choices explained:

    Why XGBoost over Random Forest?
      - XGBoost: Gradient boosting iteratively improves on weak learners
      - Faster training, better handling of mixed feature scales
      - Better feature importance (gain-based vs gini/entropy in RF)
      - SHAP values more reliable with boosted models

    Hyperparameter tuning:
      - max_depth=6: Balance bias/variance; medical data is complex but not huge
      - n_estimators=300: More trees = lower bias, but with early stopping
      - learning_rate=0.05: Conservative; lower rates are more stable
      - subsample=0.8: Stochastic gradient boosting prevents overfitting
      - colsample_bytree=0.8: Feature subsampling improves generalization

    Returns:
        Configured XGBClassifier ready for training.
    """
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=5.0,  # Will be recalculated in fit() based on class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )


if __name__ == "__main__":
    # Quick test: fit and predict on synthetic data
    from synthetic_data import generate_dataset

    print("Generating synthetic dataset...")
    df = generate_dataset(n_patients=500, random_seed=42)

    print("Extracting features...")
    X_tabular, _ = extract_tabular_features(df)
    X_text = df["discharge_note"].tolist()
    y = df["readmitted_30d"].values

    # Train on 80% of data
    from sklearn.model_selection import train_test_split

    X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = (
        train_test_split(X_tabular, X_text, y, test_size=0.2, random_state=42)
    )

    print("Training model...")
    model = ReadmissionRiskModel(use_bert=False)  # TF-IDF for speed
    model.fit(X_tab_train, X_text_train, y_train)

    print("Evaluating on test set...")
    y_pred_proba = model.predict_proba(X_tab_test, X_text_test)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"Test AUC-ROC: {auc:.3f}")

    print("\nModel saved/loaded test...")
    model.save("/tmp/test_model.pkl")
    loaded_model = ReadmissionRiskModel.load("/tmp/test_model.pkl")
    y_pred_proba_loaded = loaded_model.predict_proba(X_tab_test, X_text_test)
    assert np.allclose(y_pred_proba, y_pred_proba_loaded)
    print("Save/load test passed")
