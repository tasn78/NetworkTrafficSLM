import os
import psutil
import socket
import pandas as pd
import numpy as np
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import xgboost as xgb
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from collections import deque
from datetime import datetime, timedelta
import time
# Deque expires over time to reduce memory consumption
DEQUE_MAX_SIZE = 1000
DEQUE_EXPIRATION_SECONDS = 60

# Load the pre-trained models
BINARY_MODEL_PATH = "xgboost_model.json"
MULTICLASS_MODEL_PATH = "xgboost_multi_class_model.json"
binary_model = xgb.Booster()
binary_model.load_model(BINARY_MODEL_PATH)
multi_class_model = xgb.Booster()
multi_class_model.load_model(MULTICLASS_MODEL_PATH)

# Load the SLM model for text classification
nlp_model = pipeline("text-classification", model="rdpahalavan/bert-network-packet-flow-header-payload", framework="pt")

# Load the LLM model
# Load GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Initialize the pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

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
    """Extract features from a network packet with error handling."""
    try:
        # Initialize features with defaults
        features = {
            'Src Port': packet[TCP].sport if packet.haslayer(TCP) else (packet[UDP].sport if packet.haslayer(UDP) else 0),
            'Dst Port': packet[TCP].dport if packet.haslayer(TCP) else (packet[UDP].dport if packet.haslayer(UDP) else 0),
            'Protocol': packet[IP].proto if packet.haslayer(IP) else 0,
            'Flow Duration': 0,
            'Tot Fwd Pkts': 0,
            'Tot Bwd Pkts': 0,
            'TotLen Fwd Pkts': 0,
            'TotLen Bwd Pkts': 0,
            'Fwd Pkt Len Max': 0,
            'Fwd Pkt Len Min': 0,
            'Fwd Pkt Len Mean': 0,
            'Fwd Pkt Len Std': 0,
            'Bwd Pkt Len Max': 0,
            'Bwd Pkt Len Min': 0,
            'Bwd Pkt Len Mean': 0,
            'Bwd Pkt Len Std': 0,
            'Flow Byts/s': 0,
            'Flow Pkts/s': 0,
            'Flow IAT Mean': 0,
            'Flow IAT Std': 0,
            'Flow IAT Max': 0,
            'Flow IAT Min': 0,
            'Fwd IAT Tot': 0,
            'Fwd IAT Mean': 0,
            'Fwd IAT Std': 0,
            'Fwd IAT Max': 0,
            'Fwd IAT Min': 0,
            'Bwd IAT Tot': 0,
            'Bwd IAT Mean': 0,
            'Bwd IAT Std': 0,
            'Bwd IAT Max': 0,
            'Bwd IAT Min': 0,
            'Fwd PSH Flags': 0,
            'Bwd PSH Flags': 0,
            'Fwd URG Flags': 0,
            'Bwd URG Flags': 0,
            'Fwd Header Len': 0,
            'Bwd Header Len': 0,
            'Fwd Pkts/s': 0,
            'Bwd Pkts/s': 0,
            'Pkt Len Min': len(packet),
            'Pkt Len Max': len(packet),
            'Pkt Len Mean': len(packet),
            'Pkt Len Std': 0,
            'Pkt Len Var': 0,
            'FIN Flag Cnt': 0,
            'SYN Flag Cnt': 0,
            'RST Flag Cnt': 0,
            'PSH Flag Cnt': 0,
            'ACK Flag Cnt': 0,
            'URG Flag Cnt': 0,
            'CWE Flag Count': 0,
            'ECE Flag Cnt': 0,
            'Hour': datetime.fromtimestamp(packet.time).hour if hasattr(packet, 'time') else 0,
            'Minute': datetime.fromtimestamp(packet.time).minute if hasattr(packet, 'time') else 0,
            'Second': datetime.fromtimestamp(packet.time).second if hasattr(packet, 'time') else 0,
            'Day_of_Week': datetime.fromtimestamp(packet.time).weekday() if hasattr(packet, 'time') else 0,
        }

        # Extract TCP flags if the packet has a TCP layer
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

        # Add logic for flow-based features
        pkt_lengths = deque(maxlen=DEQUE_MAX_SIZE)
        pkt_lengths.append((len(packet), datetime.now()))
        pkt_lengths = [
            (length, timestamp)
            for length, timestamp in pkt_lengths
            if datetime.now() - timestamp <= timedelta(seconds=DEQUE_EXPIRATION_SECONDS)
        ]
        features['Pkt Len Std'] = np.std([length for length, _ in pkt_lengths])
        features['Pkt Len Var'] = np.var([length for length, _ in pkt_lengths])

        # Flow IAT calculation
        flow_iat = deque(maxlen=DEQUE_MAX_SIZE)
        flow_iat.append((time.time(), datetime.now()))
        flow_iat = [(iat, timestamp) for iat, timestamp in flow_iat if datetime.now() - timestamp <= timedelta(seconds=DEQUE_EXPIRATION_SECONDS)]
        features['Flow IAT Mean'] = np.mean([iat for iat, _ in flow_iat]) if flow_iat else 0
        features['Flow IAT Std'] = np.std([iat for iat, _ in flow_iat]) if flow_iat else 0
        features['Flow IAT Max'] = max([iat for iat, _ in flow_iat]) if flow_iat else 0
        features['Flow IAT Min'] = min([iat for iat, _ in flow_iat]) if flow_iat else 0

        return features

    except AttributeError as e:
        print(f"AttributeError: {e} - Check packet structure.")
        return None
    except Exception as e:
        print(f"Unexpected error in feature extraction: {e}")
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

        # Debugging log - uncomment for feature information
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

        if df is None or df.empty:
            raise ValueError("Preprocessed features are empty or invalid.")

        # Convert to DMatrix and predict
        dmatrix = xgb.DMatrix(df)
        predictions = multi_class_model.predict(dmatrix)

        if predictions is None or len(predictions) == 0:
            raise ValueError("Prediction output is empty or invalid.")

        # Get the predicted class index
        predicted_class = np.argmax(predictions[0])

        # Map class index to attack type
        attack_label = ATTACK_TYPES.get(predicted_class, 'Unknown')

        # Debugging log - Uncomment for feature details
        # print(f"Features: {features}")
        # print(f"Predictions: {predictions}")
        # print(f"Predicted Class: {predicted_class}, Label: {attack_label}")

        return attack_label
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
    """
    Process a network packet, analyze it, and generate notifications for malicious traffic.
    """
    DEBUG_MODE = True  # Set to True for debugging benign traffic
    try:
        # Step 1: Extract packet features
        features = extract_packet_features(packet)
        if not features:
            if DEBUG_MODE:
                print("⚠️ Skipping packet: Unable to extract features.")
            return

        if DEBUG_MODE:
            print(f"Extracted Packet Features: {features}")

        # Step 2: SLM analysis
        slm_label, slm_score = analyze_payload_with_slm(str(packet))
        if DEBUG_MODE:
            print(f"SLM Analysis: Label={slm_label}, Score={slm_score:.2f}")

        # Step 3: Binary classification
        binary_prediction = predict_binary(features)
        if DEBUG_MODE:
            print(f"Binary Classification: {'Malicious' if binary_prediction else 'Benign'}")

        # Step 4: Determine if traffic is malicious
        slm_trigger = slm_label.lower() != "normal" and slm_score > 0.9  # Adjust SLM threshold as needed
        binary_trigger = binary_prediction

        if slm_trigger or binary_trigger:
            if DEBUG_MODE:
                print("⚠️ Malicious traffic detected!")

            # Step 5: Multi-class classification (if binary triggers)
            multiclass_label = None
            if binary_trigger:
                predicted_class = predict_multiclass(features)
                multiclass_label = ATTACK_TYPES.get(predicted_class, 'Unknown')
                if DEBUG_MODE:
                    print(f"Multi-class Classification: {multiclass_label} (Class ID: {predicted_class})")

            # Aggregate results for LLM notification
            data = {
                'Protocol': features.get('Protocol', 'Unknown'),
                'Src Port': features.get('Src Port', 'Unknown'),
                'Dst Port': features.get('Dst Port', 'Unknown'),
                'Pkt Len Min': features.get('Pkt Len Min', 0),
                'Pkt Len Max': features.get('Pkt Len Max', 0),
                'SLM Label': slm_label,
                'SLM Score': slm_score,
                'Binary Prediction': binary_prediction,
                'Multi-class Label': multiclass_label,
            }

            # Step 6: Generate and send LLM notification
            notification = generate_notification_with_llm(data)
            print(f"LLM Notification: {notification}")

        else:
            if DEBUG_MODE:
                print("No malicious activity detected. Packet classified as benign.")

    except KeyError as ke:
        print(f"Error processing packet - missing key: {ke}")
    except Exception as e:
        print(f"Error processing packet: {e}")


def log_debug_info(features, prediction, model_type):
    print(f"Model: {model_type}")
    print(f"Features Passed: {features}")
    print(f"Prediction: {prediction}")


def generate_notification_with_llm(data):
    try:
        # Construct the prompt
        prompt = f"""
        Network Alert:
        - Protocol: {data['Protocol']}
        - Source Port: {data['Src Port']}
        - Destination Port: {data['Dst Port']}
        - SLM Analysis: {data['SLM Label']} (Score: {data['SLM Score']:.2f})
        - Binary Classification: {'Malicious' if data['Binary Prediction'] else 'Benign'}
        - Multi-class Classification: {data['Multi-class Label'] or 'Not Triggered'}

        Provide a **concise** summary of the detected threat:
        - Type of attack (e.g., DDoS, Exploits).
        - Suggested actions for mitigation.
        - Implications for network security if unresolved.
        """

        # Tokenize the input prompt
        tokenized_prompt = llm_pipeline.tokenizer(prompt, return_tensors="pt")
        input_length = len(tokenized_prompt["input_ids"][0])

        # Truncate if input exceeds a safe length (e.g., 512 tokens)
        if input_length > 512:
            print("Truncating prompt to fit model input capacity.")
            truncated_prompt = prompt[:1024]  # Truncate to a reasonable length
        else:
            truncated_prompt = prompt

        # Generate a response
        response = llm_pipeline(
            truncated_prompt,
            max_new_tokens=300,
            truncation=True,
            num_return_sequences=1
        )

        return response[0]['generated_text']
    except Exception as e:
        print(f"Error generating LLM notification: {e}")
        return "Error: Unable to generate notification."


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

def monitor_traffic():
    """
    Monitor network traffic, process packets, and analyze them for potential threats.
    """
    try:
        # Get the active network interface
        interface = get_active_interface()
        if not interface:
            print("No active interface found. Make sure you are connected to a network.")
            return

        print(f"Starting network monitoring on interface: {interface}")

        # Start sniffing traffic on the selected interface
        sniff(iface=interface, prn=process_packet, filter="ip", store=False)
    except Exception as e:
        print(f"Error in monitor_traffic: {e}")

if __name__ == "__main__":
    monitor_traffic()