# Binary classification model using Gradient Boosting - GPU XGBoost

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib

# Paths to your dataset
DATA_FOLDER = "preprocessed_data/"

# Load and preprocess data
def load_and_preprocess_data(data_folder):
    print("Loading and preprocessing data...")
    data = []
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_path.endswith(".csv"):
            print(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            # Ensure no NaNs
            df = df.dropna()
            # Convert labels to binary (1 for malicious, 0 for benign)
            df['Label'] = df['Label'].apply(lambda x: 1 if x != 'Benign' else 0)
            data.append(df)

    # Combine all files into a single DataFrame
    full_data = pd.concat(data, ignore_index=True)

    # Feature and label separation
    X = full_data.drop(columns=['Label'])
    y = full_data['Label']

    return X, y

# Train the XGBoost model
def train_xgboost_model(X, y):
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Setting up XGBoost parameters...")
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',  # Use GPU acceleration
        'predictor': 'gpu_predictor',
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100
    }

    print("Training the XGBoost model...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Save the trained model
    print("Saving the model...")
    model_path = "xgboost_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_and_preprocess_data(DATA_FOLDER)

    # Train and evaluate the XGBoost model
    train_xgboost_model(X, y)
