import os
import psutil
import socket
import pandas as pd
import numpy as np
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import xgboost as xgb
from datetime import datetime
from transformers import pipeline

# Load the pre-trained models
BINARY_MODEL_PATH = "xgboost_model.json"
MULTICLASS_MODEL_PATH = "xgboost_multi_class_model.json"
binary_model = xgb.Booster()
binary_model.load_model(BINARY_MODEL_PATH)
multi_class_model = xgb.Booster()
multi_class_model.load_model(MULTICLASS_MODEL_PATH)

# Load the SLM model for text classification
nlp_model = pipeline("text-classification", model="rdpahalavan/bert-network-packet-flow-header-payload", framework="pt")

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
    """Extract features from a network packet."""
    try:
        features = {
            'Protocol': packet[IP].proto if packet.haslayer(IP) else 0,
            'Src Port': packet[TCP].sport if packet.haslayer(TCP) else (packet[UDP].sport if packet.haslayer(UDP) else 0),
            'Dst Port': packet[TCP].dport if packet.haslayer(TCP) else (packet[UDP].dport if packet.haslayer(UDP) else 0),
            'Pkt Len Min': len(packet),
            'Pkt Len Max': len(packet),
            'Pkt Len Mean': len(packet),
            'Pkt Len Std': 0,
            'Flow Duration': 0,
            'Flow Byts/s': 0,
            'Flow Pkts/s': 0,
            'Hour': datetime.fromtimestamp(packet.time).hour,
            'Minute': datetime.fromtimestamp(packet.time).minute,
            'Second': datetime.fromtimestamp(packet.time).second,
            'Day_of_Week': datetime.fromtimestamp(packet.time).weekday(),
        }
        return features
    except Exception as e:
        print(f"Error extracting packet features: {e}")
        return None

def preprocess_features(features, model_features):
    """Ensure features match the model's training features."""
    try:
        df = pd.DataFrame([features])
        for col in model_features:
            if col not in df.columns:
                df[col] = 0
        df = df[model_features]  # Retain only features the model expects
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

        # Debugging log
        #log_debug_info(features, prediction, "Binary Model")

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

        # Debugging log
        #log_debug_info(features, predictions, "Multi-class Model")

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])

        # Map class index to attack type
        return ATTACK_TYPES.get(predicted_class, 'Unknown')
    except Exception as e:
        print(f"Error during multi-class prediction: {e}")
        return None


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


def log_data(features, binary_prediction, multiclass_label=None, file_path="traffic_log.csv"):
    """Log data to a CSV file."""
    try:
        features["binary_prediction"] = binary_prediction
        features["multiclass_label"] = multiclass_label if multiclass_label else "Benign"
        log_df = pd.DataFrame([features])
        log_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    except Exception as e:
        print(f"Error logging data: {e}")

def process_packet(packet):
    """Process a network packet."""
    try:
        features = extract_packet_features(packet)
        if features:
            print(f"Packet Features: {features}")

            payload = str(packet)
            label, score = analyze_payload_with_slm(payload)
            if label and score:
                print(f"SLM Analysis: Label={label}, Score={score:.4f}")

            binary_prediction = predict_binary(features)
            if binary_prediction:
                print("⚠️ Suspicious traffic detected!")
                multiclass_label = predict_multiclass(features)
                print(f"Attack Type Detected: {multiclass_label}")
                log_data(features, binary_prediction, multiclass_label)
            else:
                print("Traffic is benign.")
                log_data(features, binary_prediction)
    except Exception as e:
        print(f"Error processing packet: {e}")

def log_debug_info(features, prediction, model_type):
    print(f"Model: {model_type}")
    print(f"Features Passed: {features}")
    print(f"Prediction: {prediction}")


def get_active_interface():
    """Find an active network interface."""
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

def start_monitoring():
    """Start monitoring network traffic."""
    interface = get_active_interface()
    if interface:
        print(f"Starting network monitoring on {interface}")
        sniff(iface=interface, prn=process_packet, filter="ip", store=False)
    else:
        print("No active interface found. Make sure you are connected to Wi-Fi or Ethernet.")

if __name__ == "__main__":
    start_monitoring()
