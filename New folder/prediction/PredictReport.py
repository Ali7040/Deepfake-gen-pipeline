import os
import requests

def generate_qwen_report(fake_confidence):
    """
    Explainable AI (XAI) Cloud Microservice.
    Uses Qwen 2.5 72B to generate a professional forensic text report.
    """
    # Set your Hugging Face API Token (Settings -> Access Tokens) as an
    # environment variable — never hardcode it here.
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
    
    # Combined Optimized ChatML Prompt Format
    prompt = f"""<|im_start|>system
You are DeepTrace, a cybersecurity forensic AI. Write a professional, 2-sentence security report based on the deepfake probability score provided. Be direct, highly technical, and do not use greetings or filler words.<|im_end|>
<|im_start|>user
Image Fake Probability: {fake_confidence}%<|im_end|>
<|im_start|>assistant
"""

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.2, # Low temp for analytical tone
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        
        # Keep the robust validation from your before code
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text'].strip()
        elif isinstance(result, dict) and 'error' in result:
            return f"Qwen API Error: {result['error']}"
        else:
            return "Forensic analysis inconclusive."
    except Exception as e:
        return f"Microservice offline: {str(e)}"