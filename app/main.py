from typing import Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pydantic import BaseModel, Field
api = FastAPI()

MODEL_PATH = "models/rf_diabetes_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/trained_features.pkl"
THRESHOLD = 0.55

app = FastAPI(title="Diabetes Prediction API") #, version="1.0")

class PatientData(BaseModel):
    Pregnancies: int = Field(..., description="Number of pregnancies")
    Glucose: int = Field(..., description="Glucose level")
    BloodPressure: int = Field(..., description="Blood pressure level")
    SkinThickness: int = Field(..., description="Skin thickness")
    Insulin: int = Field(..., description="Insulin level")
    BMI: float = Field(..., description="Body Mass Index")
    DiabetesPedigreeFunction: float = Field(..., description="Diabetes pedigree function")
    Age: int = Field(..., description="Age of the patient")


# 1) Feature Engineering Functions

def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 0 with NaN in selected columns (same as training pipeline).
    """
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)
    return df

def median_impute(df):
    imputer = SimpleImputer(strategy='median')
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = imputer.fit_transform(df[cols])
    return df


def log_transform_features(df):
    for col in ['Insulin', 'Glucose', 'BMI']:
        df[col + '_log'] = np.log1p(df[col])
    return df

def create_age_group(df):
    df['Age_group'] = pd.cut(df['Age'], bins=[20,30,40,50,100],
                             labels=['20-30','30-40','40-50','50+'])
    return df

def create_bmi_category(df):
    def bmi_category(bmi):
        if bmi < 18.5: return 'Underweight'
        elif bmi < 25: return 'Normal'
        elif bmi < 30: return 'Overweight'
        else: return 'Obese'
    df['BMI_cat'] = df['BMI_log'].apply(lambda x: bmi_category(np.expm1(x)))
    return df

def create_interactions(df):
    df['Glucose_log*Age'] = df['Glucose_log'] * df['Age']
    df['BMI_log*Age'] = df['BMI_log'] * df['Age']
    return df

# 2) Preprocessing function (training & new data)
def preprocess(df, scaler=None, fit_scaler=False):
    df = replace_zeros_with_nan(df)
    df = median_impute(df)
    df = log_transform_features(df)
    df = create_age_group(df)
    df = create_bmi_category(df)
    df = create_interactions(df)
    
    # One-hot encoding for categorical features
    df = pd.get_dummies(df, columns=['Age_group', 'BMI_cat'], drop_first=False)
    
    # List of numeric columns to scale
    numeric_cols = ['Pregnancies','BloodPressure','SkinThickness','DiabetesPedigreeFunction',
                    'Age','Insulin_log','Glucose_log','BMI_log','Glucose_log*Age','BMI_log*Age']
    
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True
    
    if fit_scaler:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

def align_features(df, trained_columns):
    for col in trained_columns:
        if col not in df.columns:
            df[col] = 0
    return df[trained_columns]


# Load model, scaler, and trained features
try: 
    best_rf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)   
    trained_features = joblib.load(FEATURES_PATH)
except Exception as e:
    print(f"Error loading model or scaler: {e}")


# API endpoint for predicting diabetes
@app.get("/")
def read_root():
    return{"status":"ok", "message":"Diabetes Prediction API is running"}

@app.post("/predict")
def predict_diabetes( patient: PatientData ):
    patient_df = pd.DataFrame([ patient.dict()])

    try:
        # Preprocess (use saved scaler)
        df_proc = preprocess(patient_df.copy(), scaler=scaler, fit_scaler=False)

        # Align columns to trained features
        X = align_features(df_proc, trained_features)

        # Predict probability and apply threshold
        proba = best_rf.predict_proba(X)[:,1][0]
        pred = int(proba >= THRESHOLD)

        return {
            "predicted_outcome": pred,
            "probability": float(proba),
            "threshold_used": THRESHOLD
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
