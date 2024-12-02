import os
import pandas as pd
from datetime import datetime
import time
from LLM_SLM_hybrid import (
    process_packet,
    predict_binary,
    predict_multiclass,
    generate_notification_with_llm,
    analyze_payload_with_slm,
    extract_packet_features,
    log_data,
    get_traffic_pattern
)

# Paths
DATA_FOLDER = "preprocessed_data/"
LOG_FILE = "test_results_log.csv"

# Preprocess data
def preprocess_data(df):
    """
    Preprocess the data by removing unnecessary columns, handling NaNs, and converting types.
    """
    df = df.drop(columns=['Timestamp', 'Flow ID', 'Src IP', 'Dst IP'], errors='ignore')
    df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    df.dropna(inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    return df

# Load and preprocess CICIDS2018 dataset
def load_cicids_data(folder):
    """
    Load and preprocess all CSV files in the given folder.
    """
    print("Loading and preprocessing CICIDS2018 dataset...")
    data_frames = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            print(f"Loading {file}...")
            df = pd.read_csv(os.path.join(folder, file))
            # Convert labels to binary (1 for malicious, 0 for benign)
            df['Label'] = df['Label'].apply(lambda x: 1 if x != 'Benign' else 0)
            data_frames.append(preprocess_data(df))
    combined_data = pd.concat(data_frames, ignore_index=True)

    # Reduce to 100,000 rows randomly
    sampled_data = combined_data.sample(n=10000, random_state=42)
    return sampled_data


def test_hybrid_model(data):
    """
    Test the hybrid LLM/SLM model on the preprocessed dataset with enhanced logging.
    """
    print("Testing Hybrid Model...")
    log_data = []
    test_start_time = datetime.now()

    for index, row in data.iterrows():
        # Prepare features
        features = row.to_dict()
        label = features.pop('Label', None)  # Remove actual label for testing

        # Run binary classification
        binary_pred = predict_binary(features)

        # Multi-class classification if binary indicates malicious
        multiclass_pred = None
        if binary_pred:
            multiclass_pred = predict_multiclass(features)

        # Perform SLM analysis
        payload = f"Test payload for row {index}"  # Replace with actual packet data if available
        slm_label, slm_score = analyze_payload_with_slm(payload)

        # Check if traffic is malicious
        is_malicious = binary_pred or (slm_label.lower() != "normal" and slm_score > 0.7)

        if is_malicious:
            # Generate notification with enhanced data
            try:
                notification = generate_notification_with_llm({
                    'Protocol': features.get('Protocol', 'Unknown'),
                    'Src Port': features.get('Src Port', 'Unknown'),
                    'Dst Port': features.get('Dst Port', 'Unknown'),
                    'Flow Byts/s': features.get('Flow Byts/s', 0),
                    'Flow Pkts/s': features.get('Flow Pkts/s', 0),
                    'Flow IAT Mean': features.get('Flow IAT Mean', 0),
                    'Fwd Header Len': features.get('Fwd Header Len', 0),
                    'Bwd Header Len': features.get('Bwd Header Len', 0),
                    'RST Flag Cnt': features.get('RST Flag Cnt', 0),
                    'SYN Flag Cnt': features.get('SYN Flag Cnt', 0),
                    'Pkt Len Min': features.get('Pkt Len Min', 0),
                    'Pkt Len Max': features.get('Pkt Len Max', 0),
                    'SLM Label': slm_label,
                    'SLM Score': slm_score,
                    'Binary Prediction': binary_pred,
                    'Multi-class Label': multiclass_pred
                })
            except Exception as e:
                notification = f"Error generating notification: {e}"
                print(notification)

            # Debug: Print notification
            print(f"Malicious Notification for Row {index}: {notification}")

            # Enhanced logging with detailed metrics
            log_entry = {
                'Row Index': index,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Actual Label': label,
                'Binary Prediction': binary_pred,
                'Multi-class Prediction': multiclass_pred,
                'SLM Label': slm_label,
                'SLM Score': slm_score,

                # Traffic Metrics
                'Flow Rate (bytes/s)': float(features.get('Flow Byts/s', 0)),
                'Packet Rate (pkts/s)': float(features.get('Flow Pkts/s', 0)),
                'IAT Mean (ms)': float(features.get('Flow IAT Mean', 0)),
                'Protocol': features.get('Protocol', 'Unknown'),
                'Source Port': features.get('Src Port', 'Unknown'),
                'Destination Port': features.get('Dst Port', 'Unknown'),

                # Header Information
                'Forward Header Length': float(features.get('Fwd Header Len', 0)),
                'Backward Header Length': float(features.get('Bwd Header Len', 0)),

                # Connection Flags
                'RST Flags': int(features.get('RST Flag Cnt', 0)),
                'SYN Flags': int(features.get('SYN Flag Cnt', 0)),

                # Analysis
                'Traffic Pattern': get_traffic_pattern(features),
                'Packet Length Min': float(features.get('Pkt Len Min', 0)),
                'Packet Length Max': float(features.get('Pkt Len Max', 0)),

                # Full Notification
                'Notification': notification
            }
            log_data.append(log_entry)
        else:
            print(f"Row {index} classified as benign. Skipping notification.")

    # Create detailed results DataFrame
    results_df = pd.DataFrame(log_data)
    results_df.to_csv(LOG_FILE, index=False, float_format='%.3f')
    print(f"Detailed results saved to {LOG_FILE}")

    # Generate summary statistics
    test_end_time = datetime.now()
    summary = pd.DataFrame({
        'Total Records Analyzed': len(data),
        'Malicious Records Detected': len(results_df),
        'Detection Rate (%)': (len(results_df) / len(data) * 100),
        'Unique Attack Types': results_df['Multi-class Prediction'].nunique(),
        'Average SLM Score': results_df['SLM Score'].mean(),
        'Test Duration (minutes)': (test_end_time - test_start_time).total_seconds() / 60,
        'Test Start Time': test_start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Test End Time': test_end_time.strftime('%Y-%m-%d %H:%M:%S')
    }, index=[0])

    # Save summary
    summary_file = 'test_summary2.csv'
    summary.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to {summary_file}")

    # Print summary to console
    print("\nTest Summary:")
    print(f"Total Records Analyzed: {len(data)}")
    print(f"Malicious Records Detected: {len(results_df)}")
    print(f"Detection Rate: {(len(results_df) / len(data) * 100):.2f}%")
    print(f"Unique Attack Types: {results_df['Multi-class Prediction'].nunique()}")
    print(f"Test Duration: {(test_end_time - test_start_time).total_seconds() / 60:.2f} minutes")


# Main Execution
if __name__ == "__main__":
    # Load the CICIDS2018 dataset
    dataset = load_cicids_data(DATA_FOLDER)
    print(f"Dataset size after sampling: {len(dataset)} rows")

    # Test the hybrid model
    test_hybrid_model(dataset)

