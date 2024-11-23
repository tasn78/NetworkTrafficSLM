import os
import pandas as pd

# Define the path to the dataset folder
data_folder = "C:/Users/Tom/Downloads/CICIDS2018"

# Define the output folder
output_folder = "C:/Users/Tom/Downloads/ProcessedCICIDS2018"
os.makedirs(output_folder, exist_ok=True)

# Define the required columns
required_columns = [
    'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration',
    'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Label'
]

# Process each file in the dataset folder
for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)
    if file_name.endswith('.csv'):
        print(f"Processing file: {file_path}")
        try:
            # Load the dataset
            df = pd.read_csv(file_path, low_memory=False)

            # Check if all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns in file: {missing_columns}")
                continue

            # Convert Timestamp column to datetime and handle errors
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)

            # Add new time-related features
            df['Hour'] = df['Timestamp'].dt.hour
            df['Minute'] = df['Timestamp'].dt.minute
            df['Second'] = df['Timestamp'].dt.second
            df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

            # Filter out rows with invalid timestamps
            df = df.dropna(subset=['Timestamp'])

            # Save the processed file
            output_file_path = os.path.join(output_folder, file_name)
            df.to_csv(output_file_path, index=False)
            print(f"Processed file saved to: {output_file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
