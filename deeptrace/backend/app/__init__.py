"""DeepTrace unified backend.

A single FastAPI application that serves the frontend two capability areas:

    /api/detection/*   native deepfake detection (image, video, forensic report)
    /api/generation/*  proxy to the existing DeepTrace face-swap engine

Auth is stateless JWT. Heavy ML libraries (torch/transformers) are imported
lazily inside the detection services, so the app boots fast and the generation
proxy works even before the detection stack is installed.
"""

__version__ = "1.0.0"
