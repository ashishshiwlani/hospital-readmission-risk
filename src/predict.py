"""
Inference module for hospital readmission risk prediction.

Functions for making predictions on new patients and explaining model decisions
via SHAP (SHapley Additive exPlanations) values.
"""

from typing import Dict, List
import numpy as np
import shap

from src.model import ReadmissionRiskModel
from src.feature_engineering import extract_tabular_features


def predict_patient(
    model: ReadmissionRiskModel,
    patient_dict: Dict[str, any],
) -> Dict[str, any]:
    """
    Predict readmission risk for a single patient.

    Takes a dictionary matching the schema from synthetic_data.generate_patient_record()
    and returns a risk score and category (LOW/MODERATE/HIGH).

    Risk thresholds:
      - LOW:      risk_score < 0.3 (low intervention priority)
      - MODERATE: 0.3 <= risk_score < 0.6 (monitor and prepare discharge planning)
      - HIGH:     risk_score >= 0.6 (intensive intervention recommended)

    Args:
        model: Trained ReadmissionRiskModel instance.
        patient_dict: Dictionary with keys:
            - age, gender, admission_type, insurance
            - diagnosis_codes (list), num_procedures, num_medications
            - length_of_stay_days, prior_admissions_12mo
            - lab_values (dict), discharge_note (str)

    Returns:
        Dictionary with:
            - risk_score (float): Probability of readmission [0, 1]
            - risk_level (str): 'LOW', 'MODERATE', or 'HIGH'
            - prediction (bool): True if risk_score >= 0.5, False otherwise
    """
    # Convert single patient dict to DataFrame format expected by feature extraction
    import pandas as pd

    df_single = pd.DataFrame([patient_dict])

    # Extract tabular features
    X_tabular, _ = extract_tabular_features(df_single)

    # Extract clinical note
    X_text = [patient_dict["discharge_note"]]

    # Get probability prediction
    y_pred_proba = model.predict_proba(X_tabular, X_text)
    risk_score = float(y_pred_proba[0, 1])  # Probability of readmission

    # Classify risk level
    if risk_score < 0.3:
        risk_level = "LOW"
    elif risk_score < 0.6:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "prediction": bool(risk_score >= 0.5),
    }


def explain_prediction(
    model: ReadmissionRiskModel,
    patient_dict: Dict[str, any],
    background_data: np.ndarray | None = None,
    n_top_features: int = 5,
) -> List[Dict[str, any]]:
    """
    Explain model prediction using SHAP TreeExplainer.

    SHAP values attribute the model's prediction to each feature by computing
    the marginal contribution of each feature to pushing the prediction away
    from the baseline (average prediction on background data).

    For readmission risk, SHAP allows us to answer:
      "Which clinical features drove the model to predict high/low readmission risk?"

    Args:
        model: Trained ReadmissionRiskModel instance.
        patient_dict: Patient record to explain.
        background_data: Background dataset for SHAP (shape: n_bg, n_features).
            If None, uses a small synthetic background. For production, use
            a representative sample of training data.
        n_top_features: Number of top contributing features to return.

    Returns:
        List of dicts, one per top feature, with keys:
            - feature: Feature name or index
            - contribution: SHAP value (magnitude of feature's impact)
            - direction: 'increases_risk' if positive, 'decreases_risk' if negative
    """
    import pandas as pd

    # Convert patient to feature matrix
    df_single = pd.DataFrame([patient_dict])
    X_tabular, feature_names = extract_tabular_features(df_single)
    X_text = [patient_dict["discharge_note"]]

    # Extract text features (same as training)
    if model.use_bert:
        X_text_features = model.bert_embedder.embed_notes(X_text)
    else:
        from src.nlp_features import extract_keyword_features  # src-prefixed for correct path resolution

        X_text_features = extract_keyword_features(X_text)

    # Late fusion: concatenate tabular + text
    X_fused = np.concatenate([X_tabular, X_text_features], axis=1)

    # Generate background data if not provided
    if background_data is None:
        # Create a small synthetic background by averaging training statistics
        # In production, use a representative sample of actual training data
        print("Creating synthetic background data for SHAP...")
        background_data = np.random.normal(
            loc=0, scale=1, size=(100, X_fused.shape[1])
        )

    # Use XGBoost's *native* SHAP computation (pred_contribs=True).
    # WHY: The shap library's TreeExplainer has a serialisation incompatibility
    # with XGBoost >= 2.0 (base_score format changed), and shap.Explainer needs
    # an explicit masker for XGBClassifier.  XGBoost's own Tree SHAP
    # implementation (Lundberg et al., 2018) is the same algorithm and has no
    # such version mismatch because it lives inside XGBoost itself.
    import xgboost as xgb_lib

    dmat = xgb_lib.DMatrix(X_fused)
    # pred_contribs=True returns shape (n_samples, n_features + 1)
    # the last column is the bias term (expected model output), which we drop
    raw_contribs = model.xgb_model.get_booster().predict(dmat, pred_contribs=True)
    shap_values = raw_contribs[0, :-1]   # first (only) sample, drop bias column

    # Get absolute SHAP values and sort
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-n_top_features:][::-1]

    # Build explanation list
    explanation = []
    for idx in top_indices:
        if idx < len(feature_names):
            feature_name = feature_names[idx]
        else:
            # Text embedding features (BERT or TF-IDF)
            text_idx = idx - len(feature_names)
            feature_name = f"NLP_embedding_{text_idx}"

        contribution = float(abs_shap[idx])
        direction = "increases_risk" if shap_values[idx] > 0 else "decreases_risk"

        explanation.append(
            {
                "feature": feature_name,
                "contribution": contribution,
                "direction": direction,
            }
        )

    return explanation


if __name__ == "__main__":
    # Quick test: train a small model and explain a prediction
    from synthetic_data import generate_patient_record, generate_dataset
    from sklearn.model_selection import train_test_split

    print("Generating synthetic dataset for demo...")
    df = generate_dataset(n_patients=200, random_seed=42)

    X_tabular, _ = extract_tabular_features(df)
    X_text = df["discharge_note"].tolist()
    y = df["readmitted_30d"].values

    X_tab_train, X_tab_test, X_text_train, X_text_test, y_train, y_test = (
        train_test_split(X_tabular, X_text, y, test_size=0.2, random_state=42)
    )

    print("Training model...")
    model = ReadmissionRiskModel(use_bert=False)
    model.fit(X_tab_train, X_text_train, y_train)

    # Make a prediction on a new patient
    print("\nGenerating test patient...")
    test_patient = generate_patient_record(random_seed=123)

    print("\nPredicting readmission risk...")
    prediction = predict_patient(model, test_patient)
    print(f"Risk Score: {prediction['risk_score']:.2%}")
    print(f"Risk Level: {prediction['risk_level']}")
    print(f"Prediction: {prediction['prediction']}")

    print("\nExplaining prediction with SHAP...")
    explanation = explain_prediction(model, test_patient, n_top_features=5)
    print("Top contributing factors:")
    for i, factor in enumerate(explanation, 1):
        print(
            f"  {i}. {factor['feature']}: "
            f"{factor['contribution']:.4f} ({factor['direction']})"
        )
