from LLM_SLM_hybrid import generate_notification_with_llm

# Mock data
mock_data = {
    'Protocol': 6,
    'Src Port': 443,
    'Dst Port': 80,
    'SLM Label': 'Malicious',
    'SLM Score': 0.95,
    'Binary Prediction': True,
    'Multi-class Label': 'DDoS'
}

# Test the LLM notification
notification = generate_notification_with_llm(mock_data)
print(f"Generated Notification:\n{notification}")
