# slm_model.py

from transformers import GPT2Tokenizer, GPT2Model
import torch

# Load the GPT-2 model and tokenizer from Hugging Face's Model Hub
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

def analyze_packet_slm(src_ip, dst_ip, payload):
    # Tokenize the packet data
    input_text = f"Source: {src_ip}, Destination: {dst_ip}, Payload: {payload}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Use the model to analyze the traffic
    with torch.no_grad():
        output = model(input_ids)

    # Determine if the packet is suspicious
    is_suspicious = (output.mean().item() > 0.5)  # Example threshold

    return {
        'is_suspicious': is_suspicious,
        'details': "Suspicious pattern detected" if is_suspicious else "Normal traffic"
    }
