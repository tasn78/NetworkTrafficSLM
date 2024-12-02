import os
import psutil
import socket
import pandas as pd
import numpy as np
from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import xgboost as xgb
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
from collections import deque
from datetime import datetime, timedelta
import time
import json

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

# Load the trained Flan LLM
model = AutoModelForSequenceClassification.from_pretrained("security_alert_model")
tokenizer = AutoTokenizer.from_pretrained("security_alert_model")

# Load the SLM model for text classification
nlp_model = pipeline("text-classification", model="rdpahalavan/bert-network-packet-flow-header-payload", framework="pt")

# Previous LLM model
# Load GPT-2 tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained("gpt2")

# Set pad_token to eos_token
#tokenizer.pad_token = tokenizer.eos_token

# Initialize the pipeline
#llm_pipeline = pipeline(
#    "text-generation",
#    model=model,
#    tokenizer=tokenizer
#)

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


def load_llm_model():
    """Load the fine-tuned classification model for notification generation."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained("security_alert_model")
        tokenizer = AutoTokenizer.from_pretrained("security_alert_model")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        return None, None


# Load models at startup
model, tokenizer = load_llm_model()


def get_traffic_pattern(data):
    """Determine traffic pattern with safe handling of missing data."""
    try:
        packet_rate = float(data.get('Flow Pkts/s', 0))
        iat_mean = float(data.get('Flow IAT Mean', 0))

        if packet_rate > 1000:
            return "High-frequency"
        elif iat_mean < 50:
            return "Burst"
        elif iat_mean > 1000:
            return "Slow and methodical"
        else:
            return "Moderate"
    except (ValueError, TypeError):
        return "Unknown pattern"


def get_impact_level(flow_rate, packet_rate):
    """Determine impact level based on traffic metrics."""
    if flow_rate > 1000 or packet_rate > 500:
        return "High"
    elif flow_rate > 500 or packet_rate > 100:
        return "Medium"
    else:
        return "Low"


def get_threat_description(attack_type, data):
    """Generate appropriate threat description."""
    # Get both source and destination ports
    src_port = data.get('Src Port', 'Unknown')
    dst_port = data.get('Dst Port', 'Unknown')

    # For HTTP-based attacks, verify if it's actually targeting HTTP ports
    if attack_type == 'DoS Slowhttptest':
        if dst_port in [80, 443, 8080]:  # Common HTTP ports
            port_info = f"HTTP port {dst_port}"
        else:
            port_info = f"port {dst_port} (unusual for HTTP)"
    else:
        port_info = f"port {dst_port}"

    # Create appropriate description
    if attack_type == 'DoS Slowhttptest':
        return f"DoS Slowhttptest attack detected targeting {port_info} with slow HTTP request pattern"
    else:
        return f"Suspicious {attack_type} activity detected on {port_info}"


def get_protocol_name(protocol):
    """Convert protocol number to name."""
    protocols = {
        1: "ICMP",
        6: "TCP",
        17: "UDP"
    }
    try:
        protocol_num = int(float(protocol))
        return protocols.get(protocol_num, f"Protocol {protocol_num}")
    except (ValueError, TypeError):
        return "Unknown Protocol"


def get_mitigation_steps(attack_type):
    """Get specific mitigation steps for attack type."""
    mitigations = {
        'DoS Slowhttptest': """• Configure aggressive timeout policies
• Implement request rate limiting
• Enable mod_reqtimeout
• Monitor server resources""",

        'Unknown': """• Enable enhanced monitoring
• Update security signatures
• Review system logs
• Implement additional filtering"""
    }
    return mitigations.get(attack_type, mitigations['Unknown'])


def get_impact_description(attack_type, data):
    """Generate impact description based on attack type and metrics."""
    if attack_type == 'DoS Slowhttptest':
        return "Web server resources being exhausted through connection pool depletion"
    else:
        return "Potential security risk requiring investigation and monitoring"


def create_fallback_notification(data):
    """Create a basic notification when main generation fails."""
    return f"""THREAT: Suspicious activity detected on port {data.get('Dst Port', 'Unknown')}

Attack Details:
• Protocol: {data.get('Protocol', 'Unknown')}
• Source Port: {data.get('Src Port', 'Unknown')}
• Destination Port: {data.get('Dst Port', 'Unknown')}
• Alert Type: {data.get('Multi-class Label', 'Unknown')}

MITIGATION:
• Enable enhanced monitoring
• Review security logs
• Update security rules

IMPACT: Potential security risk requiring investigation"""


def generate_notification_with_llm(data):
    """Generate notification with error handling for missing data."""
    try:
        # Debug prints
        print("\nDebug - Input Data:")
        print(f"Source Port: {data.get('Src Port')}")
        print(f"Destination Port: {data.get('Dst Port')}")
        print(f"Attack Type: {data.get('Multi-class Label')}")
        print(f"Protocol: {data.get('Protocol')}")

        # Ensure numeric values are properly handled
        data = {
            k: (0 if pd.isna(v) and isinstance(v, (int, float)) else v)
            for k, v in data.items()
        }

        # Add missing fields if they don't exist
        required_fields = [
            'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
            'Fwd Header Len', 'Bwd Header Len', 'RST Flag Cnt',
            'SYN Flag Cnt'
        ]
        for field in required_fields:
            if field not in data:
                data[field] = 0

        # Get attack type and ensure it's a string
        attack_type = str(data.get('Multi-class Label', 'Unknown'))
        if pd.isna(attack_type):
            attack_type = 'Unknown'

        # Calculate metrics safely
        flow_rate = float(data.get('Flow Byts/s', 0))
        packet_rate = float(data.get('Flow Pkts/s', 0))
        iat_mean = float(data.get('Flow IAT Mean', 0))

        # Port analysis
        src_port = data.get('Src Port', 'Unknown')
        dst_port = data.get('Dst Port', 'Unknown')
        port_info = get_port_analysis(src_port, dst_port, attack_type)

        # Create detailed attack information
        # Create detailed attack information
        attack_details = {
            'DoS Slowhttptest': f"""Attack Details:
        - Connection Hold Time: {iat_mean:.2f} ms
        - Request Rate: {packet_rate:.2f} requests/sec
        - Traffic Pattern: {'High-volume' if packet_rate > 1000 else 'Slow and Low'}
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Attack Vector: {'High-Rate DoS' if packet_rate > 10000 else 'Slow and Low DoS'}
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Attack Signature: Slow HTTP headers""",

            'DDoS': f"""Attack Details:
        - Flow Rate: {flow_rate:.2f} bytes/sec
        - Packet Rate: {packet_rate:.2f} packets/sec
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Connection Pattern: {get_traffic_pattern(data)}
        - SYN Flags: {data.get('SYN Flag Cnt', 0)}
        - RST Flags: {data.get('RST Flag Cnt', 0)}
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Attack Vector: {'SYN Flood' if data.get('SYN Flag Cnt', 0) > 100 else 'Volume-based DDoS'}""",

            'FTP-Patator': f"""Attack Details:
        - Attack Rate: {packet_rate:.2f} attempts/sec
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Failed Connections: {data.get('RST Flag Cnt', 0)}
        - Connection Pattern: {'Aggressive' if packet_rate > 100 else 'Stealthy'}
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Forward Packets: {data.get('Tot Fwd Pkts', 0)}
        - Backward Packets: {data.get('Tot Bwd Pkts', 0)}
        - Attack Vector: FTP Brute Force""",

            'SSH-Patator': f"""Attack Details:
        - Attack Rate: {packet_rate:.2f} attempts/sec
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Failed Connections: {data.get('RST Flag Cnt', 0)}
        - Connection Pattern: {'Aggressive' if packet_rate > 100 else 'Stealthy'}
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Forward Packets: {data.get('Tot Fwd Pkts', 0)}
        - Backward Packets: {data.get('Tot Bwd Pkts', 0)}
        - Attack Vector: SSH Brute Force""",

            'Web Attack Brute Force': f"""Attack Details:
        - Attack Rate: {packet_rate:.2f} attempts/sec
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Header Length: {data.get('Fwd Header Len', 0)} bytes
        - Response Size: {data.get('Bwd Pkt Len Mean', 0)} bytes
        - Connection Pattern: {'Aggressive' if packet_rate > 100 else 'Stealthy'}
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Attack Vector: Web Authentication Brute Force""",

            'Web Attack XSS': f"""Attack Details:
        - Request Rate: {packet_rate:.2f} requests/sec
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Payload Size: {data.get('Fwd Pkt Len Mean', 0)} bytes
        - Response Size: {data.get('Bwd Pkt Len Mean', 0)} bytes
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Pattern: {'Large Payload' if data.get('Fwd Pkt Len Mean', 0) > 1000 else 'Standard Request'}
        - Attack Vector: Cross-Site Scripting""",

            'Web Attack Sql Injection': f"""Attack Details:
        - Request Rate: {packet_rate:.2f} requests/sec
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Query Size: {data.get('Fwd Pkt Len Mean', 0)} bytes
        - Response Size: {data.get('Bwd Pkt Len Mean', 0)} bytes
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Pattern: {'Complex Query' if data.get('Fwd Pkt Len Mean', 0) > 500 else 'Simple Query'}
        - Attack Vector: SQL Injection""",

            'DoS Hulk': f"""Attack Details:
        - Flow Rate: {flow_rate:.2f} bytes/sec
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Request Rate: {packet_rate:.2f} requests/sec
        - Connection Pattern: {get_traffic_pattern(data)}
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Attack Vector: HTTP Flood""",

            'DoS GoldenEye': f"""Attack Details:
        - Flow Rate: {flow_rate:.2f} bytes/sec
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Request Rate: {packet_rate:.2f} requests/sec
        - Connection Pattern: {get_traffic_pattern(data)}
        - Keep-alive Connections: {data.get('SYN Flag Cnt', 0)}
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Attack Vector: Resource Exhaustion""",

            'DoS Slowloris': f"""Attack Details:
        - Connection Hold Time: {iat_mean:.2f} ms
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Active Connections: {data.get('SYN Flag Cnt', 0) - data.get('RST Flag Cnt', 0)}
        - Request Rate: {packet_rate:.2f} requests/sec
        - Impact Level: {get_impact_level(flow_rate, packet_rate)}
        - Attack Vector: Connection Pool Exhaustion""",

            'Unknown': f"""Attack Details:
        - Flow Rate: {flow_rate:.2f} bytes/sec
        - Packet Rate: {packet_rate:.2f} packets/sec
        - Protocol: {get_protocol_name(data.get('Protocol', 'Unknown'))}
        - {get_port_analysis(src_port, dst_port, attack_type, data.get('Protocol'))}
        - Connection Pattern: {get_traffic_pattern(data)}
        - Forward Packets: {data.get('Tot Fwd Pkts', 0)}
        - Backward Packets: {data.get('Tot Bwd Pkts', 0)}
        - Alert Confidence: {data.get('SLM Score', 0):.2f}"""
        }

        # Get appropriate template
        template = attack_details.get(attack_type, attack_details['Unknown'])

        # Create structured notification
        notification = f"""THREAT: {get_threat_description(attack_type, data)}

{template}

MITIGATION:
{get_mitigation_steps(attack_type)}

IMPACT: {get_impact_description(attack_type, data)}"""

        return notification

    except Exception as e:
        print(f"Error in generate_notification_with_llm: {str(e)}")
        return create_fallback_notification(data)


def get_port_analysis(src_port, dst_port, attack_type, protocol):
    """Enhanced port and protocol analysis with better handling of zeros and missing values."""
    try:
        common_ports = {
            20: "FTP-Data",
            21: "FTP-Control",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            80: "HTTP",
            443: "HTTPS",
            3389: "RDP",
            8080: "HTTP-Alt"
        }

        # Handle source port
        if pd.isna(src_port) or src_port == 0:
            src_desc = "Unknown Source"
        else:
            src = float(src_port)
            src_desc = f"Port {src} ({common_ports.get(int(src), 'Unknown')})"

        # Handle destination port
        if pd.isna(dst_port) or dst_port == 0:
            dst_desc = "Unknown Destination"
            dst = 0
        else:
            dst = float(dst_port)
            dst_service = common_ports.get(int(dst), "Unknown")
            dst_desc = f"Port {dst} ({dst_service})"

        protocol_name = get_protocol_name(protocol)

        if attack_type == 'DoS Slowhttptest':
            if dst == 21:
                return f"WARNING: HTTP DoS Attack targeting FTP port ({dst}) over {protocol_name}"
            elif dst == 0:
                return f"WARNING: HTTP DoS Attack targeting unknown port over {protocol_name}"
            elif dst not in [80, 443, 8080]:
                return f"WARNING: HTTP DoS Attack on non-HTTP port {dst} over {protocol_name}"
            else:
                return f"Target: {dst_desc} over {protocol_name}"

        return f"Connection: {src_desc} → {dst_desc} over {protocol_name}"

    except (ValueError, TypeError):
        return "Port Analysis: Unable to determine port information"


def create_notification(attack_type, confidence, data):
    """Create detailed notification with comprehensive attack-specific metrics."""
    # Extract common metrics
    flow_rate = float(data.get('Flow Byts/s', 0))
    packet_rate = float(data.get('Flow Pkts/s', 0))
    iat_mean = float(data.get('Flow IAT Mean', 0))
    fwd_header_len = float(data.get('Fwd Header Len', 0))
    bwd_header_len = float(data.get('Bwd Header Len', 0))
    rst_flags = int(data.get('RST Flag Cnt', 0))
    syn_flags = int(data.get('SYN Flag Cnt', 0))

    def get_traffic_pattern():
        if packet_rate > 1000:
            return "High-frequency"
        elif iat_mean < 50:
            return "Burst"
        elif iat_mean > 1000:
            return "Slow and methodical"
        else:
            return "Moderate"

    # Detailed attack-specific information
    attack_details = {
        'DDoS': f"""Attack Details:
• Flow Rate: {flow_rate:.2f} bytes/sec
• Packet Rate: {packet_rate:.2f} packets/sec
• Average Flow Interval: {iat_mean:.2f} ms
• Traffic Pattern: {get_traffic_pattern()}
• Protocol: {data['Protocol']}
• SYN Flags: {syn_flags} (potential SYN flood indicators)
• RST Flags: {rst_flags} (connection reset attempts)""",

        'DoS Hulk': f"""Attack Details:
• HTTP Request Rate: {packet_rate:.2f} requests/sec
• Traffic Volume: {flow_rate:.2f} bytes/sec
• Connection Pattern: {'Sustained' if iat_mean < 100 else 'Burst'} traffic
• Target Port: {data['Dst Port']}
• Header Size: {fwd_header_len} bytes (forward), {bwd_header_len} bytes (backward)
• Attack Signature: High-volume HTTP POST floods""",

        'Bot': f"""Attack Details:
• Bot Behavior Pattern: {get_traffic_pattern()}
• Traffic Volume: {flow_rate:.2f} bytes/sec
• Connection Interval: {iat_mean:.2f} ms
• Command Channel: Port {data['Src Port']}
• Connection Attempts: {syn_flags} SYN flags
• Failed Connections: {rst_flags} RST flags
• Automated Pattern: {'Yes' if packet_rate > 100 else 'Possible'}""",

        'FTP-Patator': f"""Attack Details:
• Authentication Attempts Rate: {packet_rate:.2f} tries/sec
• Connection Pattern: {'Aggressive' if iat_mean < 50 else 'Stealthy'}
• Target FTP Port: {data['Dst Port']}
• Protocol Violations: {rst_flags} RST flags
• Connection Attempts: {syn_flags} SYN flags
• Traffic Volume: {flow_rate:.2f} bytes/sec""",

        'SSH-Patator': f"""Attack Details:
• Login Attempt Rate: {packet_rate:.2f} attempts/sec
• Connection Pattern: {'Rapid' if iat_mean < 100 else 'Distributed'}
• Target SSH Port: {data['Dst Port']}
• Failed Connections: {rst_flags} RST flags
• New Connection Attempts: {syn_flags} SYN flags
• Average Attempt Interval: {iat_mean:.2f} ms""",

        'Infiltration': f"""Attack Details:
• Infiltration Pattern: {get_traffic_pattern()}
• Data Flow Rate: {flow_rate:.2f} bytes/sec
• Connection Behavior: {iat_mean:.2f} ms intervals
• Protocol Used: {data['Protocol']}
• Suspicious Headers: {fwd_header_len + bwd_header_len} total bytes
• Connection Attempts: {syn_flags} initiations
• Anomalous Terminations: {rst_flags} resets""",

        'DoS Slowhttptest': f"""Attack Details:
• Connection Hold Time: {iat_mean:.2f} ms
• Request Rate: {packet_rate:.2f} requests/sec
• Traffic Pattern: Low-and-slow attack signature
• Target Port: {data['Dst Port']}
• Header Manipulation: {fwd_header_len} bytes
• Active Connections: {syn_flags - rst_flags} maintained
• Connection Strategy: Partial HTTP requests""",

        'DoS GoldenEye': f"""Attack Details:
• Attack Intensity: {flow_rate:.2f} bytes/sec
• Request Frequency: {packet_rate:.2f} packets/sec
• Connection Pattern: {'Aggressive' if packet_rate > 500 else 'Moderate'}
• Target Service: Port {data['Dst Port']}
• HTTP Headers: {fwd_header_len} bytes
• Connection Floods: {syn_flags} attempts
• Resource Exhaustion: {'High' if flow_rate > 1000 else 'Moderate'}""",

        'DoS Slowloris': f"""Attack Details:
• Connection Hold Pattern: {iat_mean:.2f} ms intervals
• Active Connections: {syn_flags - rst_flags} maintained
• Target Web Port: {data['Dst Port']}
• Traffic Pattern: Low bandwidth, high connection count
• Header Size: {fwd_header_len} bytes
• Connection Strategy: Keep-alive abuse
• Resource Impact: Connection pool exhaustion""",

        'Web Attack Brute Force': f"""Attack Details:
• Authentication Attempt Rate: {packet_rate:.2f} tries/sec
• Request Pattern: {get_traffic_pattern()}
• Target Port: {data['Dst Port']}
• Request Size: {fwd_header_len} bytes
• Response Size: {bwd_header_len} bytes
• Failed Attempts: {rst_flags} connection resets
• Attack Signature: Repetitive POST requests""",

        'Web Attack XSS': f"""Attack Details:
• Injection Attempt Rate: {packet_rate:.2f} requests/sec
• Payload Size: {flow_rate:.2f} bytes/sec
• Target Service: Port {data['Dst Port']}
• Request Headers: {fwd_header_len} bytes
• Script Pattern: {'Large payload' if fwd_header_len > 1000 else 'Standard'}
• Attack Vector: HTTP GET/POST with script injection""",

        'Web Attack Sql Injection': f"""Attack Details:
• SQL Attempt Pattern: {get_traffic_pattern()}
• Payload Size: {flow_rate:.2f} bytes/sec
• Request Rate: {packet_rate:.2f} requests/sec
• Target Port: {data['Dst Port']}
• Query Length: {fwd_header_len} bytes
• Attack Vector: {'Complex' if fwd_header_len > 500 else 'Simple'} SQL patterns
• Response Size: {bwd_header_len} bytes""",

        'Benign': f"""Traffic Details:
• Flow Rate: {flow_rate:.2f} bytes/sec
• Packet Rate: {packet_rate:.2f} packets/sec
• Protocol: {data['Protocol']}
• Connection Pattern: Normal
• Port Usage: {data['Src Port']} → {data['Dst Port']}""",
    }

    # Get template with severity-based prefix
    def get_severity_prefix():
        if flow_rate > 1000 or packet_rate > 500 or rst_flags > 100:
            return "CRITICAL"
        elif flow_rate > 500 or packet_rate > 100 or rst_flags > 50:
            return "WARNING"
        else:
            return "NOTICE"

    templates = {
        # [Your existing templates here with severity]
    }

    template = templates.get(attack_type, templates.get('Unknown'))
    attack_info = attack_details.get(attack_type, f"""Attack Details:
• Flow Rate: {flow_rate:.2f} bytes/sec
• Packet Rate: {packet_rate:.2f} packets/sec
• Protocol: {data['Protocol']}
• Source Port: {data['Src Port']}
• Destination Port: {data['Dst Port']}
• Connection Pattern: {get_traffic_pattern()}""")

    severity = get_severity_prefix()

    return f"""{severity} - THREAT: {template['threat']} (Confidence: {confidence:.2f})

{attack_info}

MITIGATION:
{chr(10).join(f"• {step}" for step in template['mitigation'])}

IMPACT: {template['impact']}"""

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
        slm_trigger = slm_label.lower() != "normal" and slm_score > 0.7  # Adjust SLM threshold as needed
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


def clean_notification(notification):
    """
    Cleans and filters the raw notification generated by the LLM.
    Extracts relevant lines using stricter keyword filtering and ignores verbose or unrelated content.
    """
    try:
        keywords = ['attack', 'mitigation', 'implications', 'security']
        filtered_lines = []
        for line in notification.split('\n'):
            # Check if line contains any keywords and isn't too verbose or irrelevant
            if any(keyword in line.lower() for keyword in keywords) and len(line.split()) < 20:
                filtered_lines.append(line.strip())
        return '\n'.join(filtered_lines)
    except Exception as e:
        print(f"Error in cleaning notification: {e}")
        return "Error: Unable to clean notification."


def clean_and_validate_output(output, data):
    """Clean and validate the LLM output, ensuring it meets our format requirements."""
    try:
        # Split into sections
        sections = output.split('\n\n')
        cleaned_sections = []

        for section in sections:
            section = section.strip()
            if section.startswith(('THREAT:', 'MITIGATION:', 'IMPACT:')):
                # Remove any numbered lists beyond 3 items in mitigation
                if section.startswith('MITIGATION:'):
                    mitigation_lines = section.split('\n')
                    header = mitigation_lines[0]
                    steps = [line for line in mitigation_lines[1:] if
                             line.strip() and not line.strip().startswith('4')][:3]
                    section = '\n'.join([header] + steps)
                cleaned_sections.append(section)

        # If we don't have all three sections, use fallback
        if len(cleaned_sections) != 3:
            return create_fallback_notification(data)

        return '\n\n'.join(cleaned_sections)

    except Exception as e:
        print(f"Error cleaning output: {e}")
        return create_fallback_notification(data)


def create_fallback_notification(data):
    """Create a detailed template-based notification based on attack type."""
    attack_type = data['Multi-class Label']
    port_info = f"port {data['Dst Port']}" if data[
                                                  'Dst Port'] == 80 else f"ports {data['Src Port']} → {data['Dst Port']}"

    templates = {
        'DDoS': {
            'threat': f"A DDoS attack has been detected targeting {port_info} with a high confidence score of {data['SLM Score']:.2f}.",
            'mitigation': """1. Activate DDoS mitigation services and traffic scrubbing
2. Implement rate limiting and connection throttling
3. Scale infrastructure capacity and enable redundant systems""",
            'impact': "Service availability is at risk, potentially affecting customer operations and business continuity."
        },
        'Bot': {
            'threat': f"Automated bot activity detected on {port_info} showing suspicious behavioral patterns.",
            'mitigation': """1. Deploy advanced bot detection mechanisms
2. Implement IP-based rate limiting and CAPTCHA
3. Update WAF rules to filter malicious bot signatures""",
            'impact': "Bot activity may degrade service performance and consume valuable system resources."
        },
        'Web Attack': {
            'threat': f"Web-based attack detected targeting {port_info} with suspicious payload characteristics.",
            'mitigation': """1. Enable enhanced WAF filtering
2. Update input validation rules
3. Patch vulnerable components and systems""",
            'impact': "Potential data breach risk and system compromise if attack succeeds."
        }
    }

    # Get template or use generic if attack type not found
    template = templates.get(attack_type, {
        'threat': f"Suspicious {attack_type} activity detected on {port_info} with anomalous traffic patterns.",
        'mitigation': """1. Isolate affected systems and enable enhanced monitoring
2. Update security rules and signatures
3. Review and restrict network access policies""",
        'impact': "Security breach could lead to system compromise and data exposure."
    })

    return f"""THREAT: {template['threat']}

MITIGATION:
{template['mitigation']}

IMPACT: {template['impact']}"""

if __name__ == "__main__":
    monitor_traffic()