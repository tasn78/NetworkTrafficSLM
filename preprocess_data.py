# Processes data to be used for training
# Heartbleed and Portscan were not found on this dataset, so will not be used
import os
import pandas as pd
from datetime import datetime

# Input and output folder paths
input_folder = "C:/Users/Owner/Downloads/CICIDS2018"
output_folder = "C:/Users/Owner/Documents/NetworkTrafficSLM/preprocessed_data"

# Define expected attack types and their standardized names with all variants
ATTACK_MAPPING = {
    'BENIGN': 'Benign',
    'Benign': 'Benign',
    'Bot': 'Bot',
    'DDoS': 'DDoS',
    'DDoS attacks-LOIC-HTTP': 'DDoS',
    'DDOS attack-LOIC-UDP': 'DDoS',
    'DDOS attack-HOIC': 'DDoS',
    'DoS Hulk': 'DoS Hulk',
    'DoS attacks-Hulk': 'DoS Hulk',
    'DoS GoldenEye': 'DoS GoldenEye',
    'DoS attacks-GoldenEye': 'DoS GoldenEye',
    'DoS slowloris': 'DoS Slowloris',
    'DoS Slowloris': 'DoS Slowloris',
    'DoS attacks-Slowloris': 'DoS Slowloris',
    'DoS slowhttptest': 'DoS Slowhttptest',
    'DoS Slowhttptest': 'DoS Slowhttptest',
    'DoS attacks-SlowHTTPTest': 'DoS Slowhttptest',
    'FTP-Patator': 'FTP-Patator',
    'FTP-BruteForce': 'FTP-Patator',
    'SSH-Patator': 'SSH-Patator',
    'SSH-Bruteforce': 'SSH-Patator',
    'Web Attack Brute Force': 'Web Attack Brute Force',
    'Brute Force -Web': 'Web Attack Brute Force',
    'Web Attack XSS': 'Web Attack XSS',
    'Brute Force -XSS': 'Web Attack XSS',
    'Web Attack Sql Injection': 'Web Attack Sql Injection',
    'SQL Injection': 'Web Attack Sql Injection',
    'Infiltration': 'Infiltration',
    'Infilteration': 'Infiltration',
    'PortScan': 'PortScan',
    'Heartbleed': 'Heartbleed'
}

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


def analyze_labels(df, file_name):
    """Analyze and print label distribution in the dataset."""
    print(f"\nLabel distribution in {file_name}:")
    label_counts = df['Label'].value_counts()
    total_rows = len(df)
    for label, count in label_counts.items():
        percentage = (count / total_rows) * 100
        print(f"{label}: {count} rows ({percentage:.2f}%)")
    return label_counts


def clean_labels(df):
    """Clean and standardize labels, removing invalid entries."""
    # Remove rows where Label is literally 'Label'
    if 'Label' in df['Label'].values:
        print(f"Warning: Found {len(df[df['Label'] == 'Label'])} rows with 'Label' as the label value")
        df = df[df['Label'] != 'Label']

    # Remove empty or NaN labels
    df = df.dropna(subset=['Label'])

    return df


def preprocess_file(input_path, output_path, file_name):
    """Preprocess a single file with detailed logging."""
    print(f"\nProcessing file: {file_name}")

    # Read the CSV file with low_memory=False to avoid dtype warnings
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Initial row count: {len(df)}")

    # Clean labels before standardization
    df = clean_labels(df)

    # Print unique labels before mapping
    print("\nUnique labels before standardization:")
    print(df['Label'].unique())

    # Standardize labels
    df['Label'] = df['Label'].map(lambda x: ATTACK_MAPPING.get(x, x))

    # Print unique labels after mapping
    print("\nUnique labels after standardization:")
    print(df['Label'].unique())

    # Handle timestamp conversion and feature extraction
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        df['Second'] = df['Timestamp'].dt.second
        df['Day_of_Week'] = df['Timestamp'].dt.dayofweek

    # Analyze label distribution
    analyze_labels(df, file_name)

    # Remove rows with unknown attack types
    unknown_labels = set(df['Label'].unique()) - set(ATTACK_MAPPING.values())
    if unknown_labels:
        print(f"\nWarning: Found unknown labels: {unknown_labels}")
        original_count = len(df)
        df = df[df['Label'].isin(ATTACK_MAPPING.values())]
        removed_count = original_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with unknown labels")

    # Save the processed file
    df.to_csv(output_path, index=False)
    print(f"Processed file saved with {len(df)} rows")
    return df


# Initialize tracking variables
processed_files = []
skipped_files = []
total_label_distribution = pd.Series(dtype=int)

# Process all files
print("Starting preprocessing...")
for file_name in os.listdir(input_folder):
    if not file_name.endswith(".csv"):
        print(f"Skipping non-CSV file: {file_name}")
        continue

    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    try:
        df = preprocess_file(input_path, output_path, file_name)
        processed_files.append(file_name)

        # Update total label distribution
        total_label_distribution = total_label_distribution.add(
            df['Label'].value_counts(), fill_value=0)

    except Exception as e:
        skipped_files.append((file_name, str(e)))
        print(f"Error processing file: {file_name}")
        print(f"Error details: {str(e)}")

# Print final summary
print("\nFinal Processing Summary:")
print(f"Total files processed: {len(processed_files)}")
print("\nOverall label distribution:")
total_samples = total_label_distribution.sum()
for label, count in total_label_distribution.items():
    percentage = (count / total_samples) * 100
    print(f"{label}: {count} samples ({percentage:.2f}%)")

print("\nMissing attack types:")
missing_types = set(ATTACK_MAPPING.values()) - set(total_label_distribution.index)
for attack_type in missing_types:
    print(f"- {attack_type}")

if skipped_files:
    print("\nSkipped files:")
    for file, error in skipped_files:
        print(f"- {file}: {error}")