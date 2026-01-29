#!/usr/bin/env python3
"""
Download Face Enhancer model
"""

import os
import sys
import urllib.request
from pathlib import Path

# Base URL
BASE_URL = "https://github.com/facefusion/facefusion-assets/releases/download"

# Additional required models
MODELS = {
    # Face Enhancer - GFPGAN 1.4
    "gfpgan_1.4.onnx": f"{BASE_URL}/models-3.0.0/gfpgan_1.4.onnx",
}

def download_file(url: str, dest_path: str):
    """Download a file with progress"""
    print(f"Downloading: {os.path.basename(dest_path)}")
    print(f"  From: {url}")
    
    try:
        def report_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            print(f"\r  Progress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
        print(f"\n  Saved to: {dest_path}")
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False

def main():
    # Create models directory
    models_dir = Path(__file__).parent / ".assets" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DeepTrace Face Enhancer Model Downloader")
    print("=" * 60)
    
    success_count = 0
    for model_name, model_url in MODELS.items():
        dest_path = models_dir / model_name
        
        if dest_path.exists():
            print(f"✓ Already exists: {model_name}")
            success_count += 1
            continue
        
        if download_file(model_url, str(dest_path)):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Downloaded: {success_count}/{len(MODELS)} models")

if __name__ == "__main__":
    main()
