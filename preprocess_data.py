import os
import pandas as pd
from datetime import datetime

# Input and output folder paths
# Due to the size (~6 gb), the CICIDS2018 files were not added to the project
input_folder = "C:/Users/Tom/Downloads/CICIDS2018"
output_folder = "C:/Users/Tom/Downloads/ProcessedCICIDS2018"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize lists to track successes and failures
processed_files = []
skipped_files = []

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    # Skip if not a CSV file
    if not file_name.endswith(".csv"):
        print(f"Skipping non-CSV file: {file_name}")
        continue

    print(f"Processing file: {input_path}")
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)

        # Check for required columns
        required_columns = ['Dst Port', 'Protocol', 'Timestamp', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Label']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert Timestamp to datetime and extract new features
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
            df['Hour'] = df['Timestamp'].dt.hour
            df['Minute'] = df['Timestamp'].dt.minute
            df['Second'] = df['Timestamp'].dt.second
            df['Day_of_Week'] = df['Timestamp'].dt.dayofweek

        # Save the processed file
        df.to_csv(output_path, index=False)
        processed_files.append(file_name)
        print(f"Processed file saved to: {output_path}")

    except Exception as e:
        skipped_files.append((file_name, str(e)))
        print(f"Error processing file: {file_name}, Error: {e}")

# Summary of processing
print("\nProcessing Summary:")
print(f"Total files processed: {len(processed_files)}")
if processed_files:
    print("Successfully processed files:")
    for file in processed_files:
        print(f" - {file}")

print(f"Total files skipped: {len(skipped_files)}")
if skipped_files:
    print("Skipped files with errors:")
    for file, error in skipped_files:
        print(f" - {file}: {error}")
