# Multi-classification model using Gradient Boosting - GPU XGBoost
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb
import json
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to your dataset
DATA_FOLDER = "preprocessed_data/"


def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['Timestamp', 'Flow ID', 'Src IP', 'Dst IP'], errors='ignore')

    # Replace `inf` values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Convert categorical columns to category type
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Label':  # Don't convert Label column yet
            df[col] = df[col].astype('category')

    return df


def load_and_preprocess_data(data_folder):
    print("Loading and preprocessing data...")
    data = []

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_path.endswith(".csv"):
            print(f"Loading file: {file_path}")
            df = pd.read_csv(file_path, low_memory=False)
            df = preprocess_data(df)
            data.append(df)

    # Combine all files into a single DataFrame
    full_data = pd.concat(data, ignore_index=True)

    # Print class distribution
    print("\nClass Distribution:")
    class_distribution = full_data['Label'].value_counts()
    total_samples = len(full_data)
    for label, count in class_distribution.items():
        percentage = (count / total_samples) * 100
        print(f"{label}: {count} samples ({percentage:.2f}%)")

    # Encode labels
    label_encoder = LabelEncoder()
    full_data['Label'] = label_encoder.fit_transform(full_data['Label'])

    # Feature and label separation
    X = full_data.drop(columns=['Label'])
    y = full_data['Label']

    return X, y, label_encoder


def plot_confusion_matrix(y_true, y_pred, label_encoder, output_path="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_xgboost_model(X, y, label_encoder):
    print("\nSplitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Setting up XGBoost parameters...")
    params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'eval_metric': ['mlogloss', 'merror'],
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_delta_step': 1
    }

    print("Converting data to DMatrix format...")
    # Calculate class weights for imbalanced data
    classes, counts = np.unique(y_train, return_counts=True)
    weights = len(y_train) / (len(classes) * counts)
    weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([weight_dict[label] for label in y_train])

    print("Creating DMatrix objects...")
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print("Training the XGBoost model...")
    num_round = 100
    evals_result = {}

    # Simplified early stopping callback
    early_stopping = xgb.callback.EarlyStopping(
        rounds=10,
        save_best=True
    )

    model = xgb.train(
        params,
        dtrain,
        num_round,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        evals_result=evals_result,
        verbose_eval=10,
        callbacks=[early_stopping]
    )

    print("\nEvaluating the model...")
    y_pred_proba = model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Convert numeric labels back to original classes for reporting
    y_test_orig = label_encoder.inverse_transform(y_test)
    y_pred_orig = label_encoder.inverse_transform(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_test_orig, y_pred_orig)
    precision_macro = precision_score(y_test_orig, y_pred_orig, average='macro')
    precision_weighted = precision_score(y_test_orig, y_pred_orig, average='weighted')
    recall_macro = recall_score(y_test_orig, y_pred_orig, average='macro')
    recall_weighted = recall_score(y_test_orig, y_pred_orig, average='weighted')
    f1_macro = f1_score(y_test_orig, y_pred_orig, average='macro')
    f1_weighted = f1_score(y_test_orig, y_pred_orig, average='weighted')

    # Calculate ROC AUC for multi-class
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    print("\nClassification Report:")
    print(classification_report(y_test_orig, y_pred_orig))

    print("\nDetailed Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    print(f"Recall (Weighted): {recall_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"ROC AUC (OvR): {roc_auc:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(y_test_orig, y_pred_orig, label_encoder)

    # Save the trained model and label encoder
    print("\nSaving the model and label encoder...")
    model_path = "xgboost_multi_class_model.json"
    encoder_path = "label_encoder.json"

    model.save_model(model_path)

    # Save label encoder classes
    with open(encoder_path, 'w') as f:
        json.dump({
            'classes': label_encoder.classes_.tolist()
        }, f, indent=4)

    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")

    return model, evals_result


if __name__ == "__main__":
    # Set up CUDA environment variables
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    os.environ["CUDA_PATH"] = cuda_path
    os.environ["PATH"] = cuda_path + os.pathsep + os.path.join(cuda_path, 'bin') + os.pathsep + os.environ["PATH"]

    # Print CUDA configuration
    print("CUDA Configuration:")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")
    print(f"CUDA in PATH: {'bin' in os.environ['PATH']}")

    # Load and preprocess data
    X, y, label_encoder = load_and_preprocess_data(DATA_FOLDER)

    # Train and evaluate the XGBoost model
    model, evals_result = train_xgboost_model(X, y, label_encoder)