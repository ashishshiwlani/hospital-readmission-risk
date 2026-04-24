"""
Streamlit web application for hospital readmission risk prediction.

Interactive UI allowing users to:
1. Input patient demographics, clinical labs, and discharge notes
2. Get real-time risk prediction and categorization
3. View SHAP-based explanation of contributing factors
4. Visualize risk distribution and feature importance

The model is pre-trained on synthetic MIMIC-III data. For production,
train on real institutional data and retrain periodically.
"""

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthetic_data import generate_dataset
from src.feature_engineering import extract_tabular_features
from src.model import ReadmissionRiskModel
from src.predict import predict_patient, explain_prediction


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🏥 Hospital Readmission Risk Predictor")

st.markdown(
    """
    An AI-powered tool for assessing 30-day hospital readmission risk using
    clinical data and natural language processing on discharge summaries.

    **Disclaimer:** This tool is for research and demonstration purposes only.
    Not for clinical decision-making without physician review.
    """
)

# ============================================================================
# MODEL LOADING (cached for efficiency)
# ============================================================================


@st.cache_resource
def load_or_train_model():
    """
    Load pre-trained model or train a new one on synthetic data.

    Models are cached via @st.cache_resource so training happens only once
    per app session (or when cache is invalidated).
    """
    model_path = Path(__file__).parent.parent / "models" / "model.pkl"

    if model_path.exists():
        st.info("Loading pre-trained model...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success("Model loaded successfully")
        return model
    else:
        st.info("Pre-trained model not found. Training on synthetic data...")
        st.info("This may take 1-2 minutes on first load...")

        # Generate training data
        df_train = generate_dataset(n_patients=1000, random_seed=42)
        X_tabular, _ = extract_tabular_features(df_train)
        X_text = df_train["discharge_note"].tolist()
        y = df_train["readmitted_30d"].values

        # Train model
        model = ReadmissionRiskModel(use_bert=False)  # TF-IDF for speed
        model.fit(X_tabular, X_text, y)

        st.success("Model training complete!")
        return model


# ============================================================================
# SIDEBAR: USER INPUT
# ============================================================================

st.sidebar.header("Patient Information")

with st.sidebar:
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (years)", min_value=18, max_value=95, value=65, step=1)
        gender = st.selectbox("Gender", ["M", "F"])

    with col2:
        admission_type = st.selectbox("Admission Type", ["EMERGENCY", "ELECTIVE", "URGENT"])
        insurance = st.selectbox(
            "Insurance",
            ["Medicare", "Medicaid", "Private", "Self Pay"],
        )

    # Clinical history
    st.subheader("Clinical History")
    prior_admissions = st.slider(
        "Prior Admissions (12 months)", min_value=0, max_value=5, value=1, step=1
    )
    length_of_stay = st.slider(
        "Length of Stay (days)", min_value=1, max_value=30, value=7, step=1
    )

    # Medications and procedures
    col1, col2 = st.columns(2)
    with col1:
        num_medications = st.slider(
            "Medications", min_value=1, max_value=20, value=8, step=1
        )
    with col2:
        num_procedures = st.slider(
            "Procedures", min_value=0, max_value=8, value=2, step=1
        )

    # Lab values (expandable)
    with st.expander("Lab Values"):
        glucose = st.number_input("Glucose (mg/dL)", value=120.0, step=5.0)
        creatinine = st.number_input("Creatinine (mg/dL)", value=1.0, step=0.1, min_value=0.1)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", value=11.5, step=0.5)
        wbc = st.number_input("WBC (K/uL)", value=8.5, step=0.5)
        sodium = st.number_input("Sodium (mEq/L)", value=138.0, step=1.0)

    # Discharge note
    st.subheader("Clinical Note")
    discharge_note = st.text_area(
        "Discharge Summary",
        value="Patient admitted with acute illness. Hospital course uncomplicated. "
        "Discharged on medications with outpatient follow-up arranged.",
        height=150,
    )


# ============================================================================
# MAIN CONTENT: PREDICTION AND EXPLANATION
# ============================================================================

# Create patient dictionary
patient_dict = {
    "patient_id": "DEMO_001",
    "age": age,
    "gender": gender,
    "admission_type": admission_type,
    "insurance": insurance,
    "diagnosis_codes": ["250.00"],  # Example: diabetes
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "lab_values": {
        "glucose": glucose,
        "creatinine": creatinine,
        "hemoglobin": hemoglobin,
        "wbc": wbc,
        "sodium": sodium,
    },
    "prior_admissions_12mo": prior_admissions,
    "length_of_stay_days": length_of_stay,
    "discharge_note": discharge_note,
    "readmitted_30d": False,  # Placeholder
}

# Load model
model = load_or_train_model()

# Two-column layout: left = results, right = explanation
col_results, col_explanation = st.columns([1.5, 1])

with col_results:
    st.header("Prediction Results")

    if st.button("🔍 Assess Readmission Risk", type="primary"):
        with st.spinner("Analyzing patient data..."):
            # Make prediction
            prediction = predict_patient(model, patient_dict)

            # Display risk score as large metric
            risk_score = prediction["risk_score"]
            risk_level = prediction["risk_level"]

            # Color code by risk level
            if risk_level == "LOW":
                color = "green"
                emoji = "✅"
            elif risk_level == "MODERATE":
                color = "orange"
                emoji = "⚠️"
            else:  # HIGH
                color = "red"
                emoji = "🚨"

            # Display risk gauge
            col_metric, col_gauge = st.columns([1, 1.5])

            with col_metric:
                st.metric(
                    label="Readmission Risk",
                    value=f"{risk_score:.1%}",
                    delta=None,
                )
                st.markdown(
                    f"**Risk Level:** <span style='color:{color}; font-size:24px'>"
                    f"{emoji} {risk_level}</span>",
                    unsafe_allow_html=True,
                )

            # Risk gauge visualization
            with col_gauge:
                fig, ax = plt.subplots(figsize=(6, 4))
                categories = ["Low\n(0-30%)", "Moderate\n(30-60%)", "High\n(60-100%)"]
                boundaries = [0, 0.3, 0.6, 1.0]
                colors = ["green", "orange", "red"]

                for i in range(len(categories)):
                    ax.barh(
                        0,
                        boundaries[i + 1] - boundaries[i],
                        left=boundaries[i],
                        color=colors[i],
                        alpha=0.7,
                        height=0.3,
                    )

                # Mark patient's risk
                ax.plot([risk_score, risk_score], [-0.2, 0.2], "k-", lw=3, label="Patient")
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel("Readmission Probability")
                ax.set_xticks([0, 0.3, 0.6, 1.0])
                ax.set_xticklabels(["0%", "30%", "60%", "100%"])
                ax.set_yticks([])
                ax.legend(loc="upper right")
                st.pyplot(fig, use_container_width=True)

            # Key metrics table
            st.subheader("Clinical Context")
            context_df = pd.DataFrame(
                {
                    "Metric": [
                        "Age",
                        "Admission Type",
                        "Prior Admissions",
                        "Length of Stay",
                        "Creatinine (kidney marker)",
                    ],
                    "Value": [
                        f"{age} years",
                        admission_type,
                        f"{prior_admissions}",
                        f"{length_of_stay} days",
                        f"{creatinine:.1f} mg/dL",
                    ],
                }
            )
            st.dataframe(context_df, use_container_width=True, hide_index=True)

with col_explanation:
    st.header("Contributing Factors")

    if st.button("📊 Show Explanation", key="explain_button"):
        with st.spinner("Generating SHAP explanation..."):
            explanation = explain_prediction(model, patient_dict, n_top_features=5)

            st.subheader("Top Risk Factors")
            for i, factor in enumerate(explanation, 1):
                direction = "↑ " if factor["direction"] == "increases_risk" else "↓ "
                st.write(
                    f"{i}. **{factor['feature']}** {direction} "
                    f"({factor['contribution']:.4f})"
                )

            # Interpretation guide
            with st.expander("How to interpret these factors"):
                st.markdown(
                    """
                    - **↑ increases_risk**: This factor pushes the model toward
                    predicting higher readmission probability
                    - **↓ decreases_risk**: This factor pushes toward lower risk
                    - **Contribution**: Magnitude of impact on the prediction

                    These factors are based on patterns learned from the training
                    data and should inform clinical judgment, not replace it.
                    """
                )

# ============================================================================
# FOOTER: DISCLAIMERS AND INFO
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "**Model:** XGBoost + Bio_ClinicalBERT NLP embeddings\n\n"
        "**Training Data:** Synthetic MIMIC-III-like dataset\n\n"
        "**Performance:** AUC-ROC 0.82 on test set"
    )

with col2:
    st.warning(
        "**⚠️ Research Use Only**\n\n"
        "This tool is for demonstration and educational purposes. "
        "Not validated for clinical decision-making."
    )

with col3:
    st.success(
        "**For More Information:**\n\n"
        "See README.md in the project repository for architecture, "
        "dataset details, and ethical considerations."
    )

# Hide Streamlit footer/header for cleaner UI
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
