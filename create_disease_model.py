#!/usr/bin/env python3
"""
Create a machine learning model for disease outbreak risk prediction
using the disease_outbreak_dataset_1500.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import os

# Paths
DATA_PATH = "app/artifacts/disease_outbreak_dataset_1500.csv"
MODEL_PATH = "app/artifacts/inference_pipeline.joblib"
COLS_PATH = "app/artifacts/expected_columns.json"
REF_PATH = "app/artifacts/reference_sample.csv"

def create_outbreak_risk_target(df):
    """
    Create a binary target variable for 'High Risk Outbreak'
    Based on case fatality rate and case density
    """
    # Calculate case fatality rate
    df['case_fatality_rate'] = df['Deaths_Reported'] / df['Cases_Reported'].replace(0, 1)
    
    # Calculate cases per 100k population  
    df['cases_per_100k'] = (df['Cases_Reported'] / df['Population']) * 100000
    
    # Define high risk based on thresholds
    # High risk if case fatality rate > 1% OR cases per 100k > 100
    high_risk = (df['case_fatality_rate'] > 0.01) | (df['cases_per_100k'] > 100)
    
    return high_risk.astype(int)

def prepare_features(df):
    """Prepare features for modeling"""
    df = df.copy()
    
    # Create the target
    df['high_risk_outbreak'] = create_outbreak_risk_target(df)
    
    # Feature engineering
    df['case_fatality_rate'] = df['Deaths_Reported'] / df['Cases_Reported'].replace(0, 1)
    df['cases_per_100k'] = (df['Cases_Reported'] / df['Population']) * 100000
    df['recovery_rate'] = df['Recovered'] / df['Cases_Reported'].replace(0, 1)
    df['healthcare_vaccination_score'] = df['Healthcare_Expenditure_PctGDP'] * df['Vaccination_Coverage_Pct']
    
    # Select features
    feature_cols = [
        'Population', 'Cases_Reported', 'Deaths_Reported', 'Recovered',
        'Vaccination_Coverage_Pct', 'Healthcare_Expenditure_PctGDP', 
        'Urbanization_Rate_Pct', 'Avg_Temperature_C', 'Avg_Humidity_Pct',
        'case_fatality_rate', 'cases_per_100k', 'recovery_rate', 
        'healthcare_vaccination_score'
    ]
    
    categorical_cols = ['Country', 'Disease_Name']
    
    return df, feature_cols, categorical_cols

def main():
    print("Loading disease outbreak dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features and target
    df, feature_cols, categorical_cols = prepare_features(df)
    
    # All columns used for input
    all_input_cols = feature_cols + categorical_cols
    
    X = df[all_input_cols]
    y = df['high_risk_outbreak']
    
    print(f"Features: {len(all_input_cols)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create preprocessing pipeline
    from sklearn.preprocessing import OneHotEncoder
    
    numeric_features = feature_cols
    categorical_features = categorical_cols
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\\nModel Performance:")
    print(classification_report(y_test, y_pred))
    print("\\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save artifacts
    print(f"Saving model to {MODEL_PATH}")
    joblib.dump(pipeline, MODEL_PATH)
    
    # Save expected columns
    expected_cols = {
        "expected_input_cols": all_input_cols
    }
    print(f"Saving expected columns to {COLS_PATH}")
    with open(COLS_PATH, 'w') as f:
        json.dump(expected_cols, f, indent=2)
    
    # Create reference sample for dashboard
    ref_sample = X_train.sample(n=min(200, len(X_train)), random_state=42)
    # Get corresponding target values using the sampled indices
    ref_targets = y_train[ref_sample.index]
    ref_sample['high_risk_outbreak'] = ref_targets
    print(f"Saving reference sample to {REF_PATH}")
    ref_sample.to_csv(REF_PATH, index=False)
    
    print("\\nModel creation completed successfully!")
    print(f"Input features: {all_input_cols}")

if __name__ == "__main__":
    main()