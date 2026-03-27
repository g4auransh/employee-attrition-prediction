import sys
import os

# Adds the project root directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib
# ... (rest of your imports)


import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.preprocess import get_feature_lists, get_preprocessing_pipeline, load_and_split_data

def train_model(data_path):
    # 1. Load and Split
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    
    # 2. Get feature types
    num_features, cat_features = get_feature_lists(X_train)
    
    # 3. Create the full pipeline
    # We use ImbPipeline because standard sklearn Pipeline doesn't support SMOTE
    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', get_preprocessing_pipeline(num_features, cat_features)),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # 4. Train the model
    print("🚀 Training starting (Scaling -> Encoding -> SMOTE -> Random Forest)...")
    full_pipeline.fit(X_train, y_train)
    print("✅ Training complete!")
    
    # 5. Save the model and the test data (for evaluation later)
    joblib.dump(full_pipeline, 'models/best_model.pkl')
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    print("💾 Model saved to models/best_model.pkl")

if __name__ == "__main__":
    train_model('data/HR_Analytics.csv')