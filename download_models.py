#!/usr/bin/env python3
"""
Download required models for DeepTrace
"""

import os
import sys
import urllib.request
from pathlib import Path

# Base URL for FaceFusion assets (original source)
BASE_URL = "https://github.com/facefusion/facefusion-assets/releases/download"

# Required models for basic face swapping
MODELS = {
    # Face Detection - YOLO Face
    "yolo_face_8n.onnx": f"{BASE_URL}/models-3.0.0/yolo_face_8n.onnx",
    
    # Face Recognition - ArcFace
    "arcface_w600k_r50.onnx": f"{BASE_URL}/models-3.0.0/arcface_w600k_r50.onnx",
    
    # Face Landmarks - 2DFAN4
    "2dfan4.onnx": f"{BASE_URL}/models-3.0.0/2dfan4.onnx",
    
    # Face Swapper - InSwapper
    "inswapper_128_fp16.onnx": f"{BASE_URL}/models-3.0.0/inswapper_128_fp16.onnx",
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
    print("DeepTrace Model Downloader")
    print("=" * 60)
    print(f"\nModels directory: {models_dir}")
    print(f"Models to download: {len(MODELS)}\n")
    
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
    
    if success_count == len(MODELS):
        print("✓ All models ready!")
        return 0
    else:
        print("⚠ Some models failed to download")
        return 1

if __name__ == "__main__":
    sys.exit(main())
