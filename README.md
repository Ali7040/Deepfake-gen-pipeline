DeepTrace
==========

> Modular High-Accuracy Deepfake Generation Pipeline for Academic Research

[![Build Status](https://img.shields.io/github/actions/workflow/status/Ali7040/Deepfake-gen-pipeline/ci.yml.svg?branch=main)](https://github.com/Ali7040/Deepfake-gen-pipeline/actions?query=workflow:ci)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)


Preview
-------

![Preview](.github/preview.png)


Installation
------------

This is a Final Year Project (FYP) focused on deepfake generation for academic research. The installation requires Python 3.10+ and technical knowledge.

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/Ali7040/Deepfake-gen-pipeline.git
   cd Deepfake-gen-pipeline
   ```

2. Install dependencies:
   ```bash
   python install.py
   ```

3. Run the application:
   ```bash
   python deeptrace.py run
   ```

For detailed setup instructions, see [QUICK_START.md](QUICK_START.md).


Usage
-----

### Basic Commands

```bash
# Run with GUI
python deeptrace.py run

# Run in headless mode
python deeptrace.py headless-run

# Get help
python deeptrace.py --help
```

### Simple Web Interface

For quick testing, use the simplified Flask web app:

```bash
python simple_app.py
```

Then open http://localhost:5000 in your browser.


Optimizations
-------------

This project includes comprehensive optimizations for performance and efficiency:

### Performance Enhancements
- **ONNX Runtime Optimization**: Leveraged FP16 precision and graph optimizations
- **Multi-threading**: Parallel processing for face detection and analysis
- **GPU Acceleration**: CUDA and DirectML support for faster inference
- **Model Caching**: Intelligent caching to reduce load times
- **Memory Management**: Efficient resource allocation and cleanup

### Code Quality
- **Python 3.13 Compatibility**: Full support for latest Python features
- **Error Handling**: Robust exception handling and logging
- **Type Hints**: Complete type annotations for better IDE support
- **Modular Architecture**: Clean separation of concerns for maintainability

### Benchmarks
Run performance tests:
```bash
python test_optimizations.py
```

For detailed optimization documentation, see [OPTIMIZATION_COMPLETE.md](OPTIMIZATION_COMPLETE.md) and [ADVANCED_OPTIMIZATIONS.md](ADVANCED_OPTIMIZATIONS.md).


Documentation
-------------

### FYP Documentation

- [Technical Documentation](FYP_TECHNICAL_DOCUMENTATION.md) - Deep dive into models and architectures
- [Defense Guide](FYP_DEFENSE_GUIDE.md) - Q&A for FYP defense preparation
- [Visual Diagrams](FYP_VISUAL_DIAGRAMS.md) - System architecture and data flow diagrams
- [Quick Start Guide](QUICK_START.md) - Getting started with DeepTrace
- [Migration Guide](MIGRATION_TO_DEEPTRACE.md) - Rebranding and optimization details

### Features

- **Face Detection**: YOLO-Face, RetinaFace, SCRFD models
- **Face Recognition**: ArcFace (ResNet-50) for identity verification
- **Face Swapping**: InsWapper, GhostFace, BlendSwap, HyperSwap, SimSwap
- **Face Enhancement**: GFPGAN, Real-ESRGAN for quality improvement
- **Lip Sync**: Wav2Lip integration for talking video generation
- **Audio Generation**: Text-to-speech capabilities
- **Modular Pipeline**: Each component can run independently or chained together

### Academic Use

This project is developed for academic research and FYP demonstration. Please use responsibly and ethically:
- Always watermark generated content
- Respect privacy and consent
- Follow institutional ethics guidelines
- Document all experiments and results

### Repository

GitHub: [https://github.com/Ali7040/Deepfake-gen-pipeline](https://github.com/Ali7040/Deepfake-gen-pipeline)

### License

See [LICENSE.md](LICENSE.md) for details.
