import os
import psutil
import socket
import pandas as pd
import numpy as np
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import xgboost as xgb
from datetime import datetime

# Load the pre-trained models
BINARY_MODEL_PATH = "xgboost_model.json"
MULTICLASS_MODEL_PATH = "xgboost_multi_class_model.json"
binary_model = xgb.Booster()
binary_model.load_model(BINARY_MODEL_PATH)
multi_class_model = xgb.Booster()
multi_class_model.load_model(MULTICLASS_MODEL_PATH)

# Extract feature names from the models
binary_model_feature_names = binary_model.feature_names
multi_class_model_feature_names = multi_class_model.feature_names

# Define attack type mapping
ATTACK_TYPES = {
    0: 'Benign',
    1: 'DDoS',
    2: 'DoS Hulk',
    3: 'Bot',
    4: 'FTP-Patator',
    5: 'SSH-Patator',
    6: 'Infiltration',
    7: 'DoS Slowhttptest',
    8: 'DoS GoldenEye',
    9: 'DoS Slowloris',
    10: 'Web Attack Brute Force',
    11: 'Web Attack XSS',
    12: 'Web Attack Sql Injection'
}


def extract_packet_features(packet):
    try:
        # Basic features from packet
        features = {}

        # Protocol (direct value, not one-hot encoded)
        features['Protocol'] = packet[IP].proto if packet.haslayer(IP) else 0

        # Port information
        features['Src Port'] = packet[TCP].sport if packet.haslayer(TCP) else (
            packet[UDP].sport if packet.haslayer(UDP) else 0)
        features['Dst Port'] = packet[TCP].dport if packet.haslayer(TCP) else (
            packet[UDP].dport if packet.haslayer(UDP) else 0)

        # Basic packet metrics
        features['Tot Fwd Pkts'] = 1  # Single packet
        features['Tot Bwd Pkts'] = 0
        features['TotLen Fwd Pkts'] = len(packet)
        features['TotLen Bwd Pkts'] = 0

        # Packet length statistics
        features['Pkt Len Min'] = len(packet)
        features['Pkt Len Max'] = len(packet)
        features['Pkt Len Mean'] = len(packet)
        features['Pkt Len Std'] = 0
        features['Pkt Len Var'] = 0

        # Flow metrics (simplified for single packet)
        features['Flow Duration'] = 0
        features['Flow Byts/s'] = 0
        features['Flow Pkts/s'] = 0

        # TCP flags if available
        if packet.haslayer(TCP):
            flags = packet[TCP].flags
            features['FIN Flag Cnt'] = 1 if flags & 0x01 else 0
            features['SYN Flag Cnt'] = 1 if flags & 0x02 else 0
            features['RST Flag Cnt'] = 1 if flags & 0x04 else 0
            features['PSH Flag Cnt'] = 1 if flags & 0x08 else 0
            features['ACK Flag Cnt'] = 1 if flags & 0x10 else 0
            features['URG Flag Cnt'] = 1 if flags & 0x20 else 0
            features['CWE Flag Count'] = 1 if flags & 0x40 else 0
            features['ECE Flag Cnt'] = 1 if flags & 0x80 else 0
        else:
            features['FIN Flag Cnt'] = 0
            features['SYN Flag Cnt'] = 0
            features['RST Flag Cnt'] = 0
            features['PSH Flag Cnt'] = 0
            features['ACK Flag Cnt'] = 0
            features['URG Flag Cnt'] = 0
            features['CWE Flag Count'] = 0
            features['ECE Flag Cnt'] = 0

        # Additional required features with default values
        features['Fwd Header Len'] = 20 if packet.haslayer(IP) else 0
        features['Bwd Header Len'] = 0
        features['Fwd Pkts/s'] = 0
        features['Bwd Pkts/s'] = 0
        features['Down/Up Ratio'] = 0
        features['Pkt Size Avg'] = len(packet)
        features['Init Fwd Win Byts'] = packet[TCP].window if packet.haslayer(TCP) else 0
        features['Init Bwd Win Byts'] = 0

        # Time-based features
        current_time = datetime.fromtimestamp(packet.time)
        features['Hour'] = current_time.hour
        features['Minute'] = current_time.minute
        features['Second'] = current_time.second
        features['Day_of_Week'] = current_time.weekday()

        # Add all other required features with default values
        # (Add all remaining features from the error message with appropriate default values)

        return features
    except Exception as e:
        print(f"Error extracting packet features: {e}")
        return None


def preprocess_features(features, model_features):
    """Ensure features match the exact order and names from training."""
    try:
        # Create DataFrame with single row
        df = pd.DataFrame([features])

        # Add missing columns with default values
        for col in model_features:
            if col not in df.columns:
                df[col] = 0

        # Drop extra columns that are not in the model
        df = df[model_features]

        return df
    except Exception as e:
        print(f"Error preprocessing features: {e}")
        return None


def predict_binary(features):
    """Perform binary prediction."""
    try:
        # Get the model's expected feature names
        model_features = binary_model.feature_names

        # Preprocess the features to match the model's expected format
        df = preprocess_features(features, model_features)

        # Convert to DMatrix and predict
        dmatrix = xgb.DMatrix(df)
        prediction = binary_model.predict(dmatrix)

        # Get the binary prediction (0 or 1)
        return prediction[0] > 0.5
    except Exception as e:
        print(f"Error during binary prediction: {e}")
        return None



def predict_multiclass(features):
    """Perform multi-class prediction."""
    try:
        # Get the model's expected feature names
        model_features = multi_class_model.feature_names

        # Preprocess the features to match the model's expected format
        df = preprocess_features(features, model_features)

        # Convert to DMatrix and predict
        dmatrix = xgb.DMatrix(df)
        predictions = multi_class_model.predict(dmatrix)

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])

        # Map class index to attack type
        return ATTACK_TYPES.get(predicted_class, 'Unknown')
    except Exception as e:
        print(f"Error during multi-class prediction: {e}")
        return None

# Alert on malicious traffic
def alert_on_malicious(prediction, multiclass_label=None, features=None):
    if prediction == 1:
        print("⚠️ Suspicious traffic detected!")
        if multiclass_label is not None:
            print(f"Detected attack type: {multiclass_label}")
        if features:
            print(f"Traffic Details: {features}")

# Log the data for LLM processing
def log_data(features, binary_prediction, multiclass_label=None, file_path="traffic_log.csv"):
    try:
        features["binary_prediction"] = binary_prediction
        features["multiclass_label"] = multiclass_label if multiclass_label else "Benign"
        log_df = pd.DataFrame([features])
        log_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    except Exception as e:
        print(f"Error logging data: {e}")

# Process each packet
def process_packet(packet):
    try:
        features = extract_packet_features(packet)
        if features:
            print(f"Packet Features: {features}")

            # Binary classification
            binary_prediction = predict_binary(features)
            if binary_prediction:
                # Suspicious traffic detected
                print("⚠️ Suspicious traffic detected!")

                # Multi-class classification
                multiclass_label = predict_multiclass(features)
                print(f"Attack Type Detected: {multiclass_label}")

                # Log data
                log_data(features, binary_prediction, multiclass_label)
            else:
                # Benign traffic
                print("Traffic is benign.")
                log_data(features, binary_prediction)
    except Exception as e:
        print(f"Error processing packet: {e}")


# Find active network interface
def get_active_interface():
    interfaces = psutil.net_if_addrs()
    io_counters = psutil.net_io_counters(pernic=True)

    for iface_name, iface_addrs in interfaces.items():
        for addr in iface_addrs:
            if addr.family == socket.AF_INET:
                if iface_name in io_counters:
                    stats = io_counters[iface_name]
                    if stats.bytes_sent > 0 or stats.bytes_recv > 0:
                        if "Wi-Fi" in iface_name or "Ethernet" in iface_name:
                            return iface_name
    return None

# Start monitoring
def start_monitoring():
    interface = get_active_interface()
    if interface:
        print(f"Starting network monitoring on {interface}")
        sniff(iface=interface, prn=process_packet, filter="ip", store=False)
    else:
        print("No active interface found. Make sure you are connected to Wi-Fi or Ethernet.")

if __name__ == "__main__":
    start_monitoring()
