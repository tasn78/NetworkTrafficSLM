import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder


def create_llm_training_data(data_folder):
    """Create training examples from preprocessed CICIDS2018 dataset."""

    print("Loading preprocessed data for LLM training...")
    data = []

    # Load the preprocessed data
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_folder, file_name)
            print(f"Loading file: {file_path}")
            df = pd.read_csv(file_path, low_memory=False)
            data.append(df)

    full_data = pd.concat(data, ignore_index=True)

    # Attack templates matching your exact attack types, plus Unknown
    attack_templates = {
        'Benign': {
            'description': "Normal network traffic pattern observed with {flow_rate:.2f} Flow Bytes/s",
            'mitigations': [
                "Continue standard monitoring",
                "Maintain security baselines",
                "Regular system health checks"
            ],
            'impact': "No security impact - normal operations"
        },
        'DDoS': {
            'description': "Distributed Denial of Service attack detected with {flow_rate:.2f} Flow Bytes/s targeting port {dst_port}",
            'mitigations': [
                "Activate DDoS mitigation services",
                "Enable traffic scrubbing",
                "Scale infrastructure capacity"
            ],
            'impact': "Service availability compromised, potential system overload"
        },
        'DoS Hulk': {
            'description': "DoS Hulk attack detected with {pkt_rate:.2f} packets/sec targeting HTTP services",
            'mitigations': [
                "Deploy HTTP-specific filtering",
                "Implement request rate limiting",
                "Enable application layer protection"
            ],
            'impact': "Web server resources exhausted, service degradation likely"
        },
        'Bot': {
            'description': "Bot activity identified with unusual IAT patterns (mean: {iat_mean:.2f})",
            'mitigations': [
                "Deploy bot detection mechanisms",
                "Implement behavioral analysis",
                "Block suspicious automation patterns"
            ],
            'impact': "System resources being consumed by automated threats"
        },
        'FTP-Patator': {
            'description': "FTP brute force attack detected on port {dst_port} with high connection rate",
            'mitigations': [
                "Enable FTP access controls",
                "Implement login attempt limiting",
                "Monitor FTP authentication logs"
            ],
            'impact': "FTP service at risk of unauthorized access"
        },
        'SSH-Patator': {
            'description': "SSH brute force attack detected targeting port {dst_port}",
            'mitigations': [
                "Enable SSH key-based authentication",
                "Implement IP blacklisting",
                "Set maximum auth attempts"
            ],
            'impact': "SSH service facing unauthorized access attempts"
        },
        'Infiltration': {
            'description': "Infiltration attempt detected with suspicious traffic pattern on port {dst_port}",
            'mitigations': [
                "Isolate affected systems",
                "Enable deep packet inspection",
                "Update intrusion prevention rules"
            ],
            'impact': "Potential system compromise and data breach risk"
        },
        'DoS Slowhttptest': {
            'description': "Slowhttptest DoS attack detected targeting web services",
            'mitigations': [
                "Adjust connection timeouts",
                "Implement request limiting",
                "Configure maximum connection settings"
            ],
            'impact': "Web server resources being slowly exhausted"
        },
        'DoS GoldenEye': {
            'description': "GoldenEye DoS attack detected with {flow_rate:.2f} Flow Bytes/s",
            'mitigations': [
                "Enable application layer filtering",
                "Implement connection limiting",
                "Deploy anti-DoS measures"
            ],
            'impact': "HTTP service availability being degraded"
        },
        'DoS Slowloris': {
            'description': "Slowloris DoS attack identified targeting HTTP services",
            'mitigations': [
                "Configure connection timeouts",
                "Enable mod_reqtimeout",
                "Implement concurrent connection limits"
            ],
            'impact': "Web server connections being exhausted"
        },
        'Web Attack Brute Force': {
            'description': "Web authentication brute force attack detected on port {dst_port}",
            'mitigations': [
                "Enable account lockout policies",
                "Implement CAPTCHA protection",
                "Monitor authentication attempts"
            ],
            'impact': "Web application authentication at risk"
        },
        'Web Attack XSS': {
            'description': "Cross-site scripting attack detected in web traffic",
            'mitigations': [
                "Enable XSS filtering",
                "Implement input sanitization",
                "Update WAF rules"
            ],
            'impact': "Web application vulnerable to client-side attacks"
        },
        'Web Attack Sql Injection': {
            'description': "SQL injection attempt detected in web traffic",
            'mitigations': [
                "Enable SQL injection protection",
                "Implement prepared statements",
                "Update database access controls"
            ],
            'impact': "Database security at risk of compromise"
        },
        'Unknown': {
            'description': "Unknown malicious activity detected on port {dst_port} with anomalous characteristics (Flow Rate: {flow_rate:.2f}, Packet Rate: {pkt_rate:.2f})",
            'mitigations': [
                "Enable enhanced monitoring and logging",
                "Isolate suspicious traffic for analysis",
                "Update security signatures and rules"
            ],
            'impact': "Potential new or variant attack requiring immediate investigation"
        }
    }

    training_data = []

    # Create examples for each attack type
    for label in full_data['Label'].unique():
        attack_data = full_data[full_data['Label'] == label].sample(
            n=min(100, len(full_data[full_data['Label'] == label]))
        )

        for _, row in attack_data.iterrows():
            input_data = {
                'Protocol': int(row.get('Protocol', 0)),
                'Src Port': int(row.get('Source Port', 0)),
                'Dst Port': int(row.get('Destination Port', 0)),
                'Flow Rate': float(row.get('Flow Bytes/s', 0)),
                'Packet Rate': float(row.get('Flow Packets/s', 0)),
                'IAT Mean': float(row.get('Flow IAT Mean', 0)),
                'Attack Type': label
            }

            template = attack_templates.get(label, attack_templates['Unknown'])

            description = template['description'].format(
                flow_rate=input_data['Flow Rate'],
                src_port=input_data['Src Port'],
                dst_port=input_data['Dst Port'],
                pkt_rate=input_data['Packet Rate'],
                iat_mean=input_data['IAT Mean']
            )

            example = {
                'input': input_data,
                'output': {
                    'threat': description,
                    'mitigation': template['mitigations'],
                    'impact': template['impact']
                }
            }

            training_data.append(example)

            # Add example for Unknown attack with same metrics
            if label != 'Unknown':
                unknown_example = {
                    'input': {**input_data, 'Attack Type': 'Unknown'},
                    'output': {
                        'threat': attack_templates['Unknown']['description'].format(
                            flow_rate=input_data['Flow Rate'],
                            src_port=input_data['Src Port'],
                            dst_port=input_data['Dst Port'],
                            pkt_rate=input_data['Packet Rate'],
                            iat_mean=input_data['IAT Mean']
                        ),
                        'mitigation': attack_templates['Unknown']['mitigations'],
                        'impact': attack_templates['Unknown']['impact']
                    }
                }
                training_data.append(unknown_example)

    return training_data


def save_training_data(training_data, output_path="llm_training_data.json"):
    """Save training data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Saved {len(training_data)} training examples to {output_path}")


if __name__ == "__main__":
    DATA_FOLDER = "preprocessed_data/"
    training_data = create_llm_training_data(DATA_FOLDER)
    save_training_data(training_data)