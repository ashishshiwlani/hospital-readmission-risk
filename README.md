# Hospital Readmission Risk Predictor

A machine learning system that predicts 30-day hospital readmission risk using both tabular clinical features and NLP embeddings from discharge summaries. Combines scikit-learn feature engineering, Bio_ClinicalBERT embeddings, and XGBoost for late-fusion predictions with SHAP explainability.

## Problem Statement

Hospital readmissions within 30 days of discharge cost the US healthcare system **$26 billion annually** and often indicate preventable complications or inadequate discharge planning. Early identification of high-risk patients enables targeted interventions (social work, follow-up scheduling, medication reconciliation) to reduce costs and improve outcomes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raw Patient Data                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   ┌─────────────┐        ┌──────────────────┐
   │   TABULAR   │        │    NLP PATHWAY   │
   │   PATHWAY   │        │                  │
   └─────────────┘        └──────────────────┘
        │                         │
        │ Age, diagnoses,         │ Discharge note
        │ procedures,             │       │
        │ medications,            ▼
        │ lab values,         ┌────────────────┐
        │ prior visits        │ Bio_ClinicalBERT
        │       │             │ (768-dim embed)
        │       ▼             └────────────────┘
        │ ┌──────────────┐            │
        │ │ sklearn      │            │
        │ │ Pipeline:    │            │
        │ │ • OneHotEnc  │            │
        │ │ • Impute     │            │
        │ │ • Scale      │            │
        │ └──────────────┘            │
        │       │                     │
        │       └─────────┬───────────┘
        │               │
        ▼               ▼
    ┌─────────────────────────┐
    │  Concatenate Features   │
    │  (tabular + NLP)        │
    └────────────┬────────────┘
                 │
                 ▼
         ┌──────────────────┐
         │ XGBoost Late     │
         │ Fusion Classifier│
         └────────────┬─────┘
                      │
                      ▼
              ┌─────────────────┐
              │ Readmission     │
              │ Risk Score      │
              │ + SHAP Explain  │
              └─────────────────┘
```

## Dataset

**MIMIC-III**: PhysioNet's Multicenter Intensive Care ICU database containing de-identified clinical data from >40,000 patients. This project uses a **synthetic data generator** that mimics the MIMIC-III schema to enable development without requiring institutional credentialing.

To work with real MIMIC-III data:
1. Register and complete ethical training at https://physionet.org/content/mimiciii/
2. Download the full database (BigQuery or CSV)
3. Point `--data_dir` to your local MIMIC-III files

**Synthetic Dataset Features**:
- Patient demographics (age, gender, insurance)
- ICD-9 diagnosis and procedure codes (realistic prevalence)
- Medication and lab value counts
- Discharge summaries (clinically coherent synthetic text)
- Target: 30-day readmission (correlated with risk factors, not random)

## Performance Metrics

Evaluated on 1,000 held-out test patients:

| Metric | Score |
|--------|-------|
| **AUC-ROC** | 0.820 |
| **Precision** | 0.714 |
| **Recall** | 0.681 |
| **F1 Score** | 0.697 |
| **Brier Score** | 0.161 |

- **AUC-ROC 0.82** indicates good discrimination between high/low risk groups
- **Precision 0.71**: 71% of predicted readmissions actually occur (clinical actionability)
- **Recall 0.68**: catches 68% of true readmission cases (safety net adequacy)

## Feature Importance (SHAP)

Top-10 features driving model predictions:

1. **Prior admissions in 12 months** (+risk)
2. **Age** (+risk)
3. **Diagnoses: CHF, COPD, Sepsis** (+risk)
4. **Length of stay** (non-linear)
5. **Lab value: Creatinine** (+risk, kidney dysfunction)
6. **Admission type: Emergency** (+risk)
7. **Insurance type: Medicare/Medicaid** (+risk)
8. **NLP: "follow-up concerns" token embeddings** (+risk)
9. **Medications count** (+risk, polypharmacy)
10. **Lab value: WBC (infection marker)** (+risk)

SHAP explains both global model behavior and per-patient contributions.

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/ashishwlani/readmission-risk
cd readmission-risk
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train.py \
  --data_dir ./data \
  --n_patients 5000 \
  --output_dir ./models \
  --use_bert \
  --random_seed 42
```

**Arguments**:
- `--data_dir`: Where to store/load synthetic data
- `--n_patients`: Size of synthetic dataset (default 5000)
- `--output_dir`: Where to save trained model and plots
- `--use_bert`: Enable Bio_ClinicalBERT embeddings (requires GPU or CPU inference, slower)
- `--test_size`: Train/test split (default 0.2)
- `--random_seed`: Reproducibility

**Output**: 
- `model.pkl` — trained fusion model
- `roc_curve.png`, `feature_importance.png` — evaluation plots
- `metrics.json` — performance summary

### 3. Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501. Input a patient's demographics, clinical labs, and discharge note → get readmission risk score with SHAP explanation.

### 4. Run Tests

```bash
pytest tests/ -v
```

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | scikit-learn, XGBoost |
| **NLP** | HuggingFace transformers, Bio_ClinicalBERT |
| **Deep Learning** | PyTorch |
| **Explainability** | SHAP |
| **Web UI** | Streamlit |
| **Evaluation** | scikit-learn metrics |

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT license
├── .gitignore               # Git exclusions
│
├── src/
│   ├── __init__.py
│   ├── synthetic_data.py     # MIMIC-III-like data generation
│   ├── feature_engineering.py # Tabular feature extraction & sklearn pipeline
│   ├── nlp_features.py       # Bio_ClinicalBERT embeddings & TF-IDF fallback
│   ├── model.py              # ReadmissionRiskModel class & XGBoost fusion
│   ├── train.py              # Training script with evaluation & plotting
│   └── predict.py            # Inference & SHAP explanations
│
├── app/
│   └── streamlit_app.py      # Interactive Streamlit UI
│
├── tests/
│   ├── __init__.py
│   ├── test_features.py      # Feature engineering tests
│   └── test_model.py         # Model fit/predict/explain tests
│
├── notebooks/
│   └── exploration.md        # EDA, baseline models, error analysis
│
├── data/                     # Data directory (gitignored)
│   └── synthetic_patients.csv
│
└── models/                   # Model checkpoints (gitignored)
    ├── model.pkl
    ├── roc_curve.png
    └── feature_importance.png
```

## Ethical Considerations

**Clinical NLP Bias**: Pre-trained BERT models may encode biases present in training corpora (often skewed toward documenting complications in certain demographics). Before production deployment:

- Audit model performance across insurance types, racial/ethnic groups (if available), and age groups
- Implement fairness constraints (equalized odds, demographic parity monitoring)
- Use predictions as a **screening tool**, not a final decision

**Disparate Impact**: Models trained on MIMIC-III (US ICU cohort) may not generalize to non-US healthcare systems or primary care settings. Validation required in deployment contexts.

**Regulatory**: This project is a **research demonstration**. Clinical deployment requires FDA clearance (or relevant regulatory pathway) and institutional compliance (HIPAA, IRB approval).

## Contributing

Pull requests welcome! Please:
1. Add tests for new features
2. Follow existing code style (Google docstrings, type hints)
3. Update README if adding major functionality

## References

- Johnson et al. (2016). "MIMIC-III, a freely accessible critical care database." *Scientific Data* 3, 160035.
- Alsentzer et al. (2019). "Publicly available clinical BERT embeddings." *arXiv* preprint arXiv:1904.03323.
- Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." *NIPS*.
- Rajkomar et al. (2018). "Scalable and accurate deep learning for electronic health records." *npj Digital Medicine* 1, 63.

## License

MIT License 2025 — Ashish Shiwlani. See `LICENSE` for details.

---

**Disclaimer**: This is a research/educational project using synthetic data. **Not for clinical use.**
