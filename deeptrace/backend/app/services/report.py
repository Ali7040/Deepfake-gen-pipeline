"""Forensic report generation (explainable-AI layer).

Calls the Qwen instruct model via the Hugging Face Inference API to turn a raw
fake-probability score into a short professional security report. Requires
`HF_TOKEN` in the environment; degrades gracefully if unset/unreachable.
"""

from __future__ import annotations

import requests

from app.config import settings


def generate_report(fake_confidence: int) -> str:
    if not settings.hf_token:
        return (
            "Forensic report unavailable: no HF_TOKEN configured. "
            f"Detection score: {fake_confidence}% probability of synthetic media."
        )

    prompt = (
        "<|im_start|>system\n"
        "You are DeepTrace, a cybersecurity forensic AI. Write a professional, "
        "2-sentence security report based on the deepfake probability score "
        "provided. Be direct, highly technical, and do not use greetings or "
        "filler words.<|im_end|>\n"
        f"<|im_start|>user\nImage Fake Probability: {fake_confidence}%<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }
    headers = {
        "Authorization": f"Bearer {settings.hf_token}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            settings.report_model_url, headers=headers, json=payload, timeout=60
        )
        result = resp.json()
    except Exception as exc:  # network / decode errors
        return f"Forensic microservice offline: {exc}"

    if isinstance(result, list) and result and "generated_text" in result[0]:
        return result[0]["generated_text"].strip()
    if isinstance(result, dict) and "error" in result:
        return f"Qwen API error: {result['error']}"
    return "Forensic analysis inconclusive."
