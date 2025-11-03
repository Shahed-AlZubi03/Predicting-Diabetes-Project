"""
Professional training script for the Diabetes Prediction model, based on
the provided Jupyter Notebook.

This script performs:
1.  Argument parsing for inputs.
2.  Modular data preprocessing and feature engineering.
3.  Correct train/test splitting and scaling to prevent data leakage.
4.  Hyperparameter tuning using GridSearchCV with recall as the key metric.
5.  Training of a final Random Forest model.
6.  Optimal threshold finding to meet a minimum recall requirement.
7.  Comprehensive logging of parameters, metrics, and artifacts using MLflow.

To run:
mlflow run .
or
python train.py --data_path C:\\path\\to\\diabetes.csv --experiment_name "Diabetes Prediction"
"""

import argparse
import logging
import warnings
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.exceptions import UndefinedMetricWarning

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Helper Functions ---

def load_data(data_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    LOGGER.info(f"Loading data from {data_path}")
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        LOGGER.error(f"Error: Data file not found at {data_path}")
        sys.exit(1)

def _bmi_category(bmi: float) -> str:
    """Helper to categorize BMI."""
    if bmi < 18.5: return 'Underweight'
    if bmi < 25: return 'Normal'
    if bmi < 30: return 'Overweight'
    return 'Obese'

def preprocess_data(df: pd.DataFrame, is_train: bool, imputer: SimpleImputer = None, trained_columns: list = None) -> (pd.DataFrame, SimpleImputer, list):
    """
    Applies the full preprocessing and feature engineering pipeline.
    - Fits Imputer on training data.
    - Aligns columns for test data.
    """
    LOGGER.info("Starting preprocessing and feature engineering...")
    
    # 1. Handle Zeros (convert to NaN for imputation)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

    # 2. Median Imputation
    if is_train:
        imputer = SimpleImputer(strategy='median')
        df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])
    else:
        if imputer is None:
            raise ValueError("Imputer must be provided for test data.")
        df[cols_with_zeros] = imputer.transform(df[cols_with_zeros])
        
    # 3. Log Transformations
    for col in ['Insulin', 'Glucose', 'BMI']:
        df[col + '_log'] = np.log1p(df[col])
    
    # 4. Feature Engineering
    df['Age_group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 100], labels=['20-30', '30-40', '40-50', '50+'], right=False)
    # Use np.expm1 on the log-transformed BMI to get original value for categorization
    df['BMI_cat'] = df['BMI_log'].apply(lambda x: _bmi_category(np.expm1(x)))
    
    # 5. Interaction Terms
    df['Glucose_log*Age'] = df['Glucose_log'] * df['Age']
    df['BMI_log*Age'] = df['BMI_log'] * df['Age']

    # 6. Dummification
    df = pd.get_dummies(df, columns=['Age_group', 'BMI_cat'], drop_first=True)
    
    # 7. Drop original columns that were transformed
    df = df.drop(columns=['Insulin', 'Glucose', 'BMI'], errors='ignore')

    # 8. Align columns (for test set)
    if is_train:
        trained_columns = df.columns.tolist()
        if 'Outcome' in trained_columns:
            trained_columns.remove('Outcome')
    else:
        if trained_columns is None:
            raise ValueError("Trained columns must be provided for test data.")
        # Add missing dummy columns (if any)
        for col in trained_columns:
            if col not in df.columns:
                df[col] = 0
        # Reorder to match training
        df = df[trained_columns]

    LOGGER.info("Preprocessing complete.")
    return df, imputer, trained_columns

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: list) -> (pd.DataFrame, pd.DataFrame, StandardScaler, list):
    """Fits scaler on X_train and transforms both X_train and X_test."""
    LOGGER.info("Scaling numeric features...")
    
    numeric_cols = [
        'Pregnancies', 'BloodPressure', 'SkinThickness', 'DiabetesPedigreeFunction', 'Age',
        'Insulin_log', 'Glucose_log', 'BMI_log', 'Glucose_log*Age', 'BMI_log*Age'
    ]
    # Ensure only columns present in the dataframe are scaled
    cols_to_scale = [col for col in numeric_cols if col in feature_names]
    
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    LOGGER.info("Scaling complete.")
    return X_train_scaled, X_test_scaled, scaler

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calculates classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }
    return metrics

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, recall_floor: float) -> (float, dict, np.ndarray):
    """Finds the best threshold that meets recall_floor and maximizes F1."""
    LOGGER.info(f"Finding optimal threshold for recall >= {recall_floor}...")
    thresholds = np.arange(0.1, 0.91, 0.01)
    rows = []
    
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        rows.append({
            'threshold': t,
            'recall': recall_score(y_true, y_pred_t),
            'precision': precision_score(y_true, y_pred_t),
            'f1': f1_score(y_true, y_pred_t),
            'accuracy': accuracy_score(y_true, y_pred_t)
        })
    
    df_thresh = pd.DataFrame(rows)
    candidates = df_thresh[df_thresh['recall'] >= recall_floor]
    
    if candidates.empty:
        LOGGER.warning(f"No threshold achieves recall >= {recall_floor}. Using default 0.5.")
        best_t = 0.5
        best_row = df_thresh[df_thresh['threshold'] == 0.5].iloc[0].to_dict()
    else:
        best_row = candidates.sort_values(['f1', 'precision', 'accuracy'], ascending=False).iloc[0]
        best_t = best_row['threshold']
        best_row = best_row.to_dict()
        LOGGER.info(f"Optimal threshold found: {best_t:.2f}")

    y_final_pred = (y_proba >= best_t).astype(int)
    return best_t, best_row, y_final_pred


# --- Main Training Function ---

def main(args):
    """Main training and logging pipeline."""
    
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        LOGGER.info(f"Starting MLflow Run: {run_id}")
        
        # --- 1. Log Parameters ---
        mlflow.log_params({
            "data_path": args.data_path,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "recall_floor": args.recall_floor,
            "cv_folds": args.cv_folds
        })

        # --- 2. Load and Preprocess ---
        df = load_data(args.data_path)
        X_raw = df.drop("Outcome", axis=1, errors='ignore')
        y = df['Outcome']
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=args.test_size, stratify=y, random_state=args.random_state
        )
        
        # Preprocessing (Fit imputer on train, transform both)
        X_train, imputer, trained_cols = preprocess_data(X_train_raw, is_train=True)
        X_test, _, _ = preprocess_data(X_test_raw, is_train=False, imputer=imputer, trained_columns=trained_cols)
        
        # Scaling (Fit scaler on train, transform both)
        X_train, X_test, scaler = scale_data(X_train, X_test, feature_names=trained_cols)

        LOGGER.info(f"Training data shape: {X_train.shape}")
        LOGGER.info(f"Test data shape: {X_test.shape}")

        # --- 3. Hyperparameter Tuning ---
        LOGGER.info("Starting GridSearchCV for RandomForest...")
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [4, 6, 8, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']
        }
        
        rf = RandomForestClassifier(random_state=args.random_state)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=args.cv_folds,
            scoring='recall',
            n_jobs=-1,
            verbose=0  # Set to 1 or 2 for more verbosity
        )
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        LOGGER.info(f"GridSearch complete. Best params: {best_params}")
        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})

        # --- 4. Train Final Model ---
        LOGGER.info("Training final model with best parameters...")
        final_model = RandomForestClassifier(**best_params, random_state=args.random_state)
        final_model.fit(X_train, y_train)

        # --- 5. Evaluation (Default Threshold 0.5) ---
        LOGGER.info("Evaluating model with default 0.5 threshold...")
        y_pred_default = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]
        
        default_metrics = eval_metrics(y_test, y_pred_default, y_proba)
        mlflow.log_metrics({f"default_{k}": v for k, v in default_metrics.items()})
        LOGGER.info(f"Default Metrics: {default_metrics}")

        # --- 6. Optimal Threshold Tuning ---
        best_thresh, opt_metrics, y_pred_optimized = find_optimal_threshold(y_test, y_proba, args.recall_floor)
        
        mlflow.log_metric("optimal_threshold", best_thresh)
        mlflow.log_metrics({f"optimized_{k}": v for k, v in opt_metrics.items() if k != 'threshold'})
        LOGGER.info(f"Optimized Metrics: {opt_metrics}")

        # --- 8. Log Artifacts (Model & Preprocessors) ---
        LOGGER.info("Logging model and preprocessing artifacts...")
        
        # Log the model
        mlflow.sklearn.log_model(final_model, "model")
        
        # Save and log other artifacts
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        scaler_path = artifacts_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path))
        
        imputer_path = artifacts_dir / "imputer.pkl"
        joblib.dump(imputer, imputer_path)
        mlflow.log_artifact(str(imputer_path))
        
        columns_path = artifacts_dir / "trained_features.json"
        joblib.dump(trained_cols, columns_path)
        mlflow.log_artifact(str(columns_path))
        
        LOGGER.info(f"Training complete. Run ID: {run_id}")
        print(f"\n--- MLflow Run Finished ---")
        print(f"Run ID: {run_id}")
        print(f"Experiment: {args.experiment_name}")
        print(f"To view, run: mlflow ui")
        print(f"---------------------------\n")

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diabetes Prediction Training Script")
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"C:\Users\Shahe\OneDrive\Desktop\ml_project\diabetes.csv",
        help="Path to the input diabetes.csv file."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Diabetes-Prediction",
        help="Name of the MLflow experiment."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility."
    )
    parser.add_argument(
        "--recall_floor",
        type=float,
        default=0.75,
        help="Minimum recall required when optimizing threshold."
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=15,
        help="Number of cross-validation folds for GridSearchCV."
    )
    
    args = parser.parse_args()
    main(args)
