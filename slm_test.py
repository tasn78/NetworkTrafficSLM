import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

# Load the SLM model for text classification
nlp_model = pipeline("text-classification", model="rdpahalavan/bert-network-packet-flow-header-payload", framework="pt")

# Define attack type mapping
ATTACK_TYPES = {
    0: 'Benign',
    1: 'DoS Hulk',
    2: 'DoS GoldenEye',
    3: 'DoS Slowloris',
    4: 'DoS Slowhttptest',
    5: 'DDoS',
    6: 'Bot',
    7: 'FTP-Patator',
    8: 'SSH-Patator',
    9: 'Web Attack Brute Force',
    10: 'Web Attack XSS',
    11: 'Web Attack Sql Injection',
    12: 'Infiltration',
    13: 'Heartbleed',
    14: 'PortScan'
}

def load_and_combine_data(folder_path):
    """
    Load all CSV files in the given folder and combine them into a single DataFrame.
    """
    print("Loading and combining CICIDS2018 dataset...")
    data_frames = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            print(f"Loading {file}...")
            df = pd.read_csv(os.path.join(folder_path, file))
            data_frames.append(df)
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

def preprocess_multiclass_labels(data, attack_mapping):
    """
    Map attack type strings in the 'Label' column to their numeric values based on attack_mapping.
    """
    print("Mapping attack types to numeric labels...")
    reverse_mapping = {v: k for k, v in attack_mapping.items()}
    data['Label'] = data['Label'].apply(lambda x: reverse_mapping.get(x, -1))  # Map unknown labels to -1
    data = data[data['Label'] != -1]  # Remove rows with unmapped labels
    return data

def analyze_payload_with_slm(payload):
    """Analyze the payload using the SLM model."""
    try:
        if payload:
            result = nlp_model(payload)
            label = result[0].get("label", "Unknown")
            score = result[0].get("score", 0.0)
            return label, score
        return None, None
    except Exception as e:
        print(f"Error during SLM analysis: {e}")
        return None, None

def evaluate_slm_multiclass(data, attack_mapping):
    """
    Evaluate the SLM model on the dataset for multi-class classification.
    """
    results = []
    y_true = []  # Ground truth labels
    y_pred = []  # SLM predictions

    print("Evaluating SLM model on CICIDS2018 dataset for multi-class classification...")
    for index, row in data.iterrows():
        # Simulate payload as concatenation of relevant text fields
        payload = f"Flow Bytes/s: {row.get('Flow Byts/s', 'Unknown')}, Flow Packets/s: {row.get('Flow Pkts/s', 'Unknown')}"

        # Perform SLM analysis
        label, score = analyze_payload_with_slm(payload)

        # Map SLM label to numeric value
        slm_prediction = next((k for k, v in attack_mapping.items() if v.lower() == label.lower()), -1)

        if slm_prediction == -1:
            # Handle cases where the SLM output does not map to a known attack type
            slm_prediction = len(attack_mapping)  # Assign a special class for unknown predictions

        ground_truth = row['Label']  # Already numeric after preprocessing

        # Append results for evaluation
        y_true.append(ground_truth)
        y_pred.append(slm_prediction)
        results.append({'Index': index, 'Ground Truth': ground_truth, 'SLM Prediction': slm_prediction, 'SLM Label': label, 'SLM Score': score})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("slm_multiclass_results.csv", index=False)
    print("Results saved to slm_multiclass_results.csv")

    # Dynamically determine target names based on `y_true`
    unique_classes = sorted(set(y_true))
    dynamic_target_names = [attack_mapping[c] for c in unique_classes if c in attack_mapping]

    # Add 'Unknown' if it's part of the predictions
    if len(attack_mapping) in y_pred:
        dynamic_target_names.append('Unknown')

    # Generate evaluation metrics
    print("\nSLM Multi-Class Model Evaluation:")
    print(classification_report(y_true, y_pred, target_names=dynamic_target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def enumerate_attack_types(data, label_column='Label'):
    """
    Print unique attack types and their counts in the dataset.

    Args:
        data (pd.DataFrame): The dataset to analyze.
        label_column (str): The name of the column containing attack labels.
    """
    print("\nEnumerating attack types in the dataset...")
    attack_counts = data[label_column].value_counts()
    for attack_type, count in attack_counts.items():
        print(f"{attack_type}: {count} rows")
    print(f"Total rows: {len(data)}\n")

# Main script
if __name__ == "__main__":
    DATA_FOLDER = "preprocessed_data"  # Update with the path to your dataset folder

    # Load and combine the dataset
    combined_dataset = load_and_combine_data(DATA_FOLDER)

    # Enumerate attack types before sampling
    print("\nBefore Sampling:")
    enumerate_attack_types(combined_dataset)

    # Preprocess the dataset for multi-class classification
    combined_dataset = preprocess_multiclass_labels(combined_dataset, ATTACK_TYPES)

    # Randomly sample 10,000 rows from the combined dataset
    sampled_dataset = combined_dataset.sample(n=10000, random_state=42)
    print(f"Sampled dataset size: {len(sampled_dataset)} rows")

    # Enumerate attack types after sampling
    print("\nAfter Sampling:")
    enumerate_attack_types(sampled_dataset)

    # Evaluate the SLM model for multi-class classification
    evaluate_slm_multiclass(sampled_dataset, ATTACK_TYPES)
