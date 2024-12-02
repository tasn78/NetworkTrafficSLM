from transformers import pipeline

# Load your SLM model
nlp_model = pipeline("text-classification", model="rdpahalavan/bert-network-packet-flow-header-payload", framework="pt")

# Print model metadata
print(nlp_model.model.config.id2label)
