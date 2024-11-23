# First model, no longer used. Replaced with Binary and multi-class classification pipeline

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Function to load, clean, and preprocess data
def load_and_preprocess_data(data_folder):
    data_frames = []
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder, file)
            print(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)

            # Handle infinite values and NaNs
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)

            # Normalize numeric features
            feature_columns = df.select_dtypes(include=["float64", "int64"]).columns
            scaler = StandardScaler()
            df[feature_columns] = scaler.fit_transform(df[feature_columns])

            data_frames.append(df)

    # Concatenate all dataframes
    full_data = pd.concat(data_frames, ignore_index=True)
    return full_data

# Function to train the model
def train_model(data_folder):
    # Load and preprocess data
    data = load_and_preprocess_data(data_folder)
    print(f"Data shape after loading and preprocessing: {data.shape}")

    # Separate features and labels
    X = data.drop(columns=["Label"])
    y = data["Label"]

    # Encode labels (if necessary)
    y = y.apply(lambda x: 1 if x != "Benign" else 0)  # Example: 1 for malicious, 0 for benign

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    return model

# Main execution
if __name__ == "__main__":
    data_folder = "C:/Users/Tom/Downloads/ProcessedCICIDS2018"
    trained_model = train_model(data_folder)
