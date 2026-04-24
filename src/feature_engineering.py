"""
Tabular feature engineering for hospital readmission prediction.

Extracts and transforms demographic, clinical, and lab features from raw
patient records into a feature matrix suitable for scikit-learn models.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Fixed categorical value lists — used to build one-hot columns consistently.
# WHY: pd.get_dummies only creates columns for values actually present in the
# input DataFrame. On a single-patient DataFrame the model might see gender='Male'
# only, producing 0 columns with drop_first=True, vs. 6 columns at training time.
# By hardcoding the complete set of known values we get the same feature schema
# for training data AND single-patient inference.
KNOWN_CATEGORICAL_VALUES: dict = {
    "gender":         ["Female", "Male"],
    "admission_type": ["ELECTIVE", "EMERGENCY", "URGENT"],
    "insurance":      ["Medicaid", "Medicare", "Private", "Self Pay"],
}

# Diagnoses of clinical interest for readmission risk
# Maps readable names to ICD-9 code prefixes for easy flagging
DIAGNOSES_OF_INTEREST = {
    "Diabetes": "250",
    "Congestive Heart Failure": "428",
    "COPD": "496",
    "Pneumonia": "486",
    "Sepsis": "038",
    "Acute Kidney Injury": "584",
    "Myocardial Infarction": "410",
    "Atrial Fibrillation": "427",
}


def extract_tabular_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Extract and engineer tabular features from raw patient dataframe.

    Performs the following transformations:
    1. One-hot encodes categorical variables (gender, admission_type, insurance)
    2. Normalizes numerical variables (age, procedures, medications, LOS, prior admits)
    3. Extracts binary diagnosis flags for diagnoses of interest
    4. Extracts lab values (glucose, creatinine, hemoglobin, WBC, sodium)
    5. Handles missing values with median imputation

    Args:
        df: Raw patient DataFrame with columns: age, gender, admission_type,
            insurance, diagnosis_codes (list), num_procedures, num_medications,
            length_of_stay_days, prior_admissions_12mo, lab_values (dict).

    Returns:
        Tuple of:
            - feature_matrix (np.ndarray): (n_patients, n_features)
            - feature_names (List[str]): column names for interpretability
    """
    # Create a working copy to avoid modifying original
    working_df = df.copy()

    # === CATEGORICAL FEATURES ===
    # Build one-hot columns using the fixed KNOWN_CATEGORICAL_VALUES dict.
    # We skip the first value in each list (same effect as drop_first=True)
    # to avoid multicollinearity, but we ALWAYS produce columns for the
    # remaining values — even when only one value appears in the input DataFrame.
    # This guarantees the same feature schema for training (n=thousands) and
    # single-patient inference (n=1), preventing XGBoost's feature-count mismatch.
    categorical_features = pd.DataFrame(index=working_df.index)
    for col, known_values in KNOWN_CATEGORICAL_VALUES.items():
        for val in known_values[1:]:          # skip index 0 → drop_first equivalent
            col_name = f"{col}_{val}"
            categorical_features[col_name] = (working_df[col] == val).astype(int)
    categorical_names = list(categorical_features.columns)

    # === NUMERICAL FEATURES ===
    # Normalize: age, num_procedures, num_medications, length_of_stay_days, prior_admissions_12mo
    # These are naturally numerical and require scaling for algorithms like XGBoost
    numerical_cols = [
        "age",
        "num_procedures",
        "num_medications",
        "length_of_stay_days",
        "prior_admissions_12mo",
    ]
    numerical_features = working_df[numerical_cols].copy()
    # Handle any missing values (rare in synthetic data, but good practice)
    numerical_features = numerical_features.fillna(numerical_features.median())
    # Standardize to zero mean, unit variance for better ML performance
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)
    numerical_features = pd.DataFrame(numerical_features, columns=numerical_cols)
    numerical_names = list(numerical_features.columns)

    # === DIAGNOSIS FLAGS ===
    # For each diagnosis of interest, create a binary flag
    # This captures high-risk diagnoses without requiring a sparse ICD-9 matrix
    diagnosis_features = {}
    for diag_name, icd9_prefix in DIAGNOSES_OF_INTEREST.items():
        # Check if any diagnosis code in the patient's list starts with this prefix
        diagnosis_features[f"has_{diag_name.replace(' ', '_')}"] = [
            any(
                str(code).startswith(icd9_prefix)
                for code in (codes if isinstance(codes, list) else [])
            )
            for codes in working_df["diagnosis_codes"]
        ]
    diagnosis_features_df = pd.DataFrame(diagnosis_features).astype(int)
    diagnosis_names = list(diagnosis_features_df.columns)

    # === LAB VALUES ===
    # Extract structured lab values with imputation for missing values
    # Lab abnormalities are strong predictors of clinical severity
    lab_features = {}
    lab_keys = ["glucose", "creatinine", "hemoglobin", "wbc", "sodium"]

    for lab_key in lab_keys:
        lab_values = []
        for lab_dict in working_df["lab_values"]:
            if isinstance(lab_dict, dict) and lab_key in lab_dict:
                lab_values.append(lab_dict[lab_key])
            else:
                lab_values.append(np.nan)
        lab_features[lab_key] = lab_values

    lab_features_df = pd.DataFrame(lab_features)
    # Impute missing lab values with column median (e.g., normal range)
    lab_features_df = lab_features_df.fillna(lab_features_df.median())
    # Normalize lab values (they're on different scales)
    lab_scaler = StandardScaler()
    lab_features_df = pd.DataFrame(
        lab_scaler.fit_transform(lab_features_df),
        columns=lab_keys,
    )
    lab_names = list(lab_features_df.columns)

    # === CONCATENATE ALL FEATURES ===
    feature_matrix = pd.concat(
        [
            categorical_features.reset_index(drop=True),
            numerical_features.reset_index(drop=True),
            diagnosis_features_df.reset_index(drop=True),
            lab_features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    feature_names = categorical_names + numerical_names + diagnosis_names + lab_names

    # Cast to float64: pd.get_dummies returns bool columns in pandas >= 1.5,
    # which creates object dtype when concatenated with float columns.
    # Explicit cast ensures downstream numpy operations (np.isnan, sklearn) work correctly.
    return feature_matrix.values.astype(np.float64), feature_names


def build_sklearn_pipeline() -> Pipeline:
    """
    Build a scikit-learn preprocessing pipeline for mixed feature types.

    Note: This is a general-purpose pipeline structure. In practice, since
    extract_tabular_features already handles all preprocessing, this serves
    as a template for custom transformations or additional scaling.

    For the main training loop, we use extract_tabular_features directly
    to ensure full control over feature engineering logic.

    Returns:
        sklearn.pipeline.Pipeline object (used as reference; see extract_tabular_features
        for the actual preprocessing applied).
    """
    # Define column groups for different feature types
    numerical_cols = [
        "age",
        "num_procedures",
        "num_medications",
        "length_of_stay_days",
        "prior_admissions_12mo",
    ]
    categorical_cols = ["gender", "admission_type", "insurance"]

    # Numerical: impute missing values → scale to zero mean, unit variance
    # StandardScaler is preferred over MinMaxScaler because:
    # - It's invariant to outliers (via z-score) vs being sensitive to min/max
    # - Tree-based models (XGBoost) are scale-invariant, but good for SHAP interpretability
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical: impute mode → one-hot encode
    # OneHotEncoder avoids ordinal assumptions (e.g., insurance type has no order)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
        ]
    )

    # Combine transformers via ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    # Quick test: generate synthetic data and extract features
    from synthetic_data import generate_dataset

    df = generate_dataset(n_patients=100, random_seed=42)
    X, feature_names = extract_tabular_features(df)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"\nFirst 20 feature names:")
    for i, name in enumerate(feature_names[:20]):
        print(f"  {i}: {name}")
    print(f"\nSample feature row (patient 0):")
    print(X[0, :10])
