# Binary classification model using Gradient Boosting - GPU XGBoost

import os
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
os.environ["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin" + os.pathsep + os.environ["PATH"]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import cupy as cp

# Paths to your dataset
DATA_FOLDER = "preprocessed_data/"

# Preprocess data
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['Timestamp', 'Flow ID', 'Src IP', 'Dst IP'], errors='ignore')

    # Replace `inf` values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Convert categorical columns to category type
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    return df

# Load and preprocess data
def load_and_preprocess_data(data_folder):
    print("Loading and preprocessing data...")
    data = []
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_path.endswith(".csv"):
            print(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            # Convert labels to binary (1 for malicious, 0 for benign)
            df['Label'] = df['Label'].apply(lambda x: 1 if x != 'Benign' else 0)
            # Preprocess the data
            df = preprocess_data(df)
            data.append(df)

    # Combine all files into a single DataFrame
    full_data = pd.concat(data, ignore_index=True)

    # Feature and label separation
    X = full_data.drop(columns=['Label'])
    y = full_data['Label']

    return X, y

# Train the XGBoost model
def train_xgboost_model(X, y):
    # Add CUDA verification
    print("\nVerifying CUDA setup:")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")
    print(f"CUDA in PATH: {r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin' in os.environ['PATH']}")

    print("\nStarting model training...")
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print("Setting up XGBoost parameters...")
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',  # Specify CUDA device, removed gpu_id
        'learning_rate': 0.1,
        'max_depth': 6,
        'max_bin': 256
    }

    print("Training the XGBoost model...")
    num_round = 100

    # Add early stopping to prevent overfitting
    early_stopping = xgb.callback.EarlyStopping(
        rounds=10,
        metric_name='auc',
        save_best=True
    )

    model = xgb.train(
        params,
        dtrain,
        num_round,
        evals=[(dtest, 'test')],
        verbose_eval=10,
        callbacks=[early_stopping]
    )

    print("\nEvaluating the model...")
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    try:
        print("\nSaving the model...")
        model_path = "xgboost_model.json"
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

    return model


if __name__ == "__main__":
    # Set correct CUDA path
    import os

    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"  # Main CUDA directory
    os.environ["CUDA_PATH"] = cuda_path

    # Add CUDA bin to PATH if not already there
    cuda_bin = os.path.join(cuda_path, 'bin')
    if cuda_bin not in os.environ['PATH']:
        os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']

    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data(DATA_FOLDER)

        # Train and evaluate the XGBoost model
        model = train_xgboost_model(X, y)
    except Exception as e:
        print(f"Error during execution: {str(e)}")