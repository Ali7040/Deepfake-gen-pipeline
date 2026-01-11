# DeepTrace - Quick Start Guide

## What Changed?

Your application has been successfully migrated from "FaceFusion" to "DeepTrace" with the following improvements:

### 🎯 Rebranding
- All references changed from "FaceFusion" to "DeepTrace"
- Consistent naming across all files and UI

### ⚡ Performance Optimizations
1. **Thread Management**
   - Optimized CPU thread allocation
   - Better memory management
   - Reduced resource contention

2. **Error Handling**
   - Comprehensive exception handling
   - Better user feedback
   - Graceful degradation

3. **Code Quality**
   - Improved documentation
   - Better code organization
   - Enhanced maintainability

## 📋 Quick Start

### 1. Run the Main Application
```bash
python deeptrace.py --help
```

### 2. Start the Web Interface
```bash
python simple_app.py
```
Then open: http://localhost:5000

### 3. Use Startup Scripts

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

## 🔧 Configuration

Edit `deeptrace.ini` to customize settings:
- Face detection models
- Processing parameters
- Output settings
- Performance options

## 📁 Key Files

- `deeptrace.py` - Main application entry point
- `deeptrace.ini` - Configuration file
- `simple_app.py` - Web interface
- `install.py` - Dependency installer
- `config.py` - Advanced configuration

## 🧪 Testing

Run performance tests:
```bash
python test_optimizations.py
```

Run unit tests:
```bash
pytest tests/
```

Verify migration:
```bash
python verify_migration.py
```

## 📊 Performance Tips

1. **Use GPU** - CUDA support for 10-50x faster processing
2. **Optimize Settings** - Adjust model sizes in config
3. **Batch Processing** - Process multiple files at once
4. **Quality vs Speed** - Balance settings for your needs

## 🚀 Advanced Usage

### Command Line
```bash
# Face swap
python deeptrace.py run --source source.jpg --target target.jpg

# Batch processing
python deeptrace.py run --source-path ./sources --target target.mp4

# Custom config
python deeptrace.py run --config custom_config.ini
```

### Python API
```python
from deeptrace import state_manager, core

# Initialize
state_manager.init_item('face_swapper_model', 'inswapper_128_fp16')

# Process
core.process_swap(source='face.jpg', target='video.mp4', output='result.mp4')
```

## 📖 Documentation

See additional documentation:
- `MIGRATION_TO_DEEPTRACE.md` - Complete migration details
- `OPTIMIZATION_README.md` - Performance optimization guide
- `QUICK_START.md` - Original quick start guide

## ⚙️ Environment Variables

The application uses these optimizations automatically:
- `OMP_NUM_THREADS=1` - OpenMP optimization
- `MKL_NUM_THREADS=1` - Intel MKL optimization
- `NUMEXPR_NUM_THREADS=1` - NumPy optimization

## 🔍 Troubleshooting

### Import Errors
If you see import errors, make sure you're in the correct directory:
```bash
cd C:\Users\DELL\FYP\facefusion
```

### Missing Dependencies
Run the installer:
```bash
python install.py
```

### Performance Issues
1. Check GPU is being used: `nvidia-smi`
2. Reduce model sizes in config
3. Lower output quality settings

## 📝 Notes

- The folder is still named `facefusion` but contains `deeptrace` code
- All functionality is preserved
- Configuration format is unchanged
- Backward compatible with existing configs

## 🎉 What's Next?

1. ✅ Migration complete
2. ✅ All files updated
3. ✅ Optimizations applied
4. 🚀 Ready to use!

Enjoy your optimized DeepTrace application!
