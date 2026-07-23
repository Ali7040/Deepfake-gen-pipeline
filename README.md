# DeepTrace

<p align="center">
<h2 align="center">
AI-Powered Deepfake Detection & Generation Platform
</h2>

<p align="center">
A production-style full-stack platform that combines deepfake generation, AI-powered detection, forensic analysis, and modern web technologies into a unified application.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![React](https://img.shields.io/badge/React-19-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6)
![PyTorch](https://img.shields.io/badge/PyTorch-AI-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

# Overview

DeepTrace is an end-to-end AI platform that integrates both **deepfake generation** and **deepfake detection** into a single modern web application.

The platform provides:

- AI-powered deepfake generation
- AI-powered image & video detection
- Modern React frontend
- Production-style FastAPI backend
- JWT Authentication
- Detection history
- AI forensic reports
- Modular research pipeline

Originally developed as a research project, DeepTrace has evolved into a complete full-stack application demonstrating modern AI system design and scalable software engineering.

---

# Features

## Deepfake Detection

- Image Detection
- Video Detection
- Multi-face Detection
- AI Confidence Scores
- Detection History
- Annotated Results
- AI Forensic Reports

## Deepfake Generation

- Image Face Swapping
- Video Face Swapping
- Multiple Generation Models
- Async Processing
- Face Enhancement
- Lip Synchronization

## Platform

- Modern Dashboard
- JWT Authentication
- User Management
- REST APIs
- Responsive UI
- Secure File Upload
- Modular Architecture

---

# Screenshots

## Landing Page
<img width="948" height="471" alt="Screenshot 2026-07-23 223745" src="https://github.com/user-attachments/assets/c38169ee-0b9a-4d5e-984a-74bcc3a29cdf" />
<img width="950" height="473" alt="Screenshot 2026-07-23 223902" src="https://github.com/user-attachments/assets/ee37d8b5-d016-4684-933c-867b8f81e692" />



---

## Dashboard

![Dashboard](./screenshots/dashboard.png)

---

## Deepfake Detection

![Detection](./screenshots/detection.png)
<img width="951" height="475" alt="Screenshot 2026-07-23 223957" src="https://github.com/user-attachments/assets/f7040ee9-fa35-40a4-bcd2-3efae9b905d4" />



## Deepfake Generation








---

# System Architecture

```
                React Frontend
                       │
                       ▼
                FastAPI Backend
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
 Detection Service          Generation Service
        │                             │
        ▼                             ▼
 MTCNN + Transformers      FaceFusion Research Engine
        │                             │
        └──────────────┬──────────────┘
                       ▼
                 Processed Output
```

---

# Deepfake Generation Pipeline

```
                     Source Face
                          │
                          ▼
             Face Detection & Alignment
        (YOLO • RetinaFace • SCRFD)
                          │
                          ▼
             Facial Landmark Extraction
                          │
                          ▼
               ArcFace Identity Encoding
                          │
                          ▼
             Face Swap Model Selection
(InSwapper • GhostFace • SimSwap • HyperSwap)
                          │
                          ▼
               Initial Face Generation
                          │
                          ▼
         Temporal Consistency Module
 (One-Euro Filter + Optical Flow Blending)
                          │
                          ▼
      Closed-Loop Adaptive Quality Controller
                          │
        ArcFace Identity Verification
                          │
              Identity Score Evaluation
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      Quality Accepted          Increase Pixel Boost
             │                  Re-run Generation
             └────────────┬────────────┘
                          ▼
           Face Restoration & Enhancement
             (GFPGAN • Real-ESRGAN)
                          │
                          ▼
           Color Matching & Seamless Blend
                          │
                          ▼
      Optional Lip Synchronization (Wav2Lip)
                          │
                          ▼
              Video Reconstruction
                          │
                          ▼
                   Final Output
```

---

# Detection Pipeline

```
             Uploaded Image / Video
                     │
                     ▼
            Face Detection (MTCNN)
                     │
                     ▼
             Face Crop Extraction
                     │
                     ▼
      Vision Transformer Classification
         (ViT / SigLIP Deepfake Models)
                     │
                     ▼
          Fake Probability Estimation
                     │
                     ▼
       Confidence Threshold Evaluation
                     │
                     ▼
        AI Forensic Report Generation
                     │
                     ▼
          Detection History Storage
                     │
                     ▼
               Final Prediction
```

---

# Research Contributions

DeepTrace extends traditional face-swapping systems with original research contributions focused on improving both quality and efficiency.

### Temporal Consistency

Frame-to-frame stabilization using:

- One-Euro Filter
- Optical Flow
- Motion-aware blending

Reduces temporal flickering while preserving identity.

---

### Closed-Loop Adaptive Quality Controller

Unlike traditional pipelines that generate a result once, DeepTrace continuously evaluates output quality.

The generated face is verified using ArcFace embeddings.

If identity similarity is below a configurable threshold, the pipeline automatically:

- Increases processing quality
- Re-runs face generation
- Re-evaluates identity
- Stops when the desired quality is achieved

This adaptive feedback loop improves identity preservation while avoiding unnecessary computation.

---

# Tech Stack

## Frontend

- React
- TypeScript
- Tailwind CSS
- Framer Motion
- GSAP

## Backend

- FastAPI
- Python
- SQLAlchemy
- SQLite
- JWT Authentication
- REST API

## AI & Machine Learning

- PyTorch
- Hugging Face Transformers
- ONNX Runtime
- CUDA
- OpenCV

### Detection

- MTCNN
- Vision Transformers
- SigLIP
- ArcFace

### Generation

- FaceFusion
- InSwapper
- GhostFace
- SimSwap
- BlendSwap
- HyperSwap
- GFPGAN
- Real-ESRGAN
- Wav2Lip

---

# Installation

```bash
git clone https://github.com/Ali7040/Deepfake-gen-pipeline.git

cd Deepfake-gen-pipeline

python install.py

python deeptrace.py run
```

---

# Roadmap

- Cloud Deployment
- Docker Support
- Kubernetes
- Live Webcam Detection
- Video Streaming
- Explainable AI
- Multi-user Collaboration
- Fine-tuned Detection Models

---

# License

MIT License

---

# Author

## Ali Haider

Full-Stack Engineer • AI Engineer • Open Source Contributor

Passionate about AI, distributed systems, full-stack development, and building products that solve real-world problems.

---

⭐ If you found this project useful, consider giving it a star.
