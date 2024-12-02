from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm.auto import tqdm
import json


class SecurityAlertsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mapping = {
            'DDoS': 0,
            'DoS Hulk': 1,
            'Bot': 2,
            'FTP-Patator': 3,
            'SSH-Patator': 4,
            'Infiltration': 5,
            'DoS Slowhttptest': 6,
            'DoS GoldenEye': 7,
            'DoS Slowloris': 8,
            'Web Attack Brute Force': 9,
            'Web Attack XSS': 10,
            'Web Attack Sql Injection': 11,
            'Benign': 12,
            'Unknown': 13
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format input text
        input_text = f"""Network Alert:
Protocol: {item['input']['Protocol']}
Source Port: {item['input']['Src Port']}
Destination Port: {item['input']['Dst Port']}
Attack Type: {item['input']['Attack Type']}"""

        # Convert attack type to numeric label
        label = self.label_mapping.get(item['input']['Attack Type'], self.label_mapping['Unknown'])

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None  # Return as lists rather than tensors
        )

        # Add label to encoding
        encoding['labels'] = label

        # Convert to tensors
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(label)
        }


def train_model():
    # Initialize model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize model with correct number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=14  # 13 attack types + Unknown
    )

    # Load dataset
    dataset = SecurityAlertsDataset("llm_training_data.json", tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=data_collator
    )

    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

    # Save the fine-tuned model
    model.save_pretrained("security_alert_model")
    tokenizer.save_pretrained("security_alert_model")

    # Save label mapping
    with open("security_alert_model/label_mapping.json", 'w') as f:
        json.dump(dataset.label_mapping, f)

    print("Model and label mapping saved to security_alert_model/")


def generate_notification(alert_data):
    """Generate notification using fine-tuned model."""
    # Load fine-tuned model and label mapping
    model = AutoModelForSequenceClassification.from_pretrained("security_alert_model")
    tokenizer = AutoTokenizer.from_pretrained("security_alert_model")

    with open("security_alert_model/label_mapping.json", 'r') as f:
        label_mapping = json.load(f)

    # Reverse the label mapping for prediction
    label_to_attack = {v: k for k, v in label_mapping.items()}

    # Format input
    input_text = f"""Network Alert:
Protocol: {alert_data['Protocol']}
Source Port: {alert_data['Src Port']}
Destination Port: {alert_data['Dst Port']}
Attack Type: {alert_data['Multi-class Label']}"""

    # Tokenize and predict
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_label].item()

    # Get predicted attack type
    predicted_attack = label_to_attack[predicted_label]

    # Generate notification based on prediction
    notification = create_notification(predicted_attack, confidence, alert_data)
    return notification


def create_notification(attack_type, confidence, alert_data):
    """Create a structured notification based on the predicted attack type."""
    templates = {
        'DDoS': {
            'threat': f"DDoS attack detected targeting port {alert_data['Dst Port']}",
            'mitigation': [
                "Enable DDoS protection",
                "Implement rate limiting",
                "Scale infrastructure"
            ],
            'impact': "Service availability at risk"
        },
        # Add templates for other attack types...
        'Unknown': {
            'threat': f"Unknown malicious activity detected on port {alert_data['Dst Port']}",
            'mitigation': [
                "Enable enhanced monitoring",
                "Isolate suspicious traffic",
                "Update security rules"
            ],
            'impact': "Potential security risk requiring investigation"
        }
    }

    template = templates.get(attack_type, templates['Unknown'])

    return f"""THREAT: {template['threat']} (Confidence: {confidence:.2f})

MITIGATION:
{chr(10).join(f"• {step}" for step in template['mitigation'])}

IMPACT: {template['impact']}"""


if __name__ == "__main__":
    # First, train the model
    train_model()

    # Example usage after training
    test_alert = {
        'Protocol': 6,
        'Src Port': 443,
        'Dst Port': 80,
        'Multi-class Label': 'DDoS'
    }

    notification = generate_notification(test_alert)
    print("\nGenerated Notification:")
    print(notification)