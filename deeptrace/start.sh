#!/bin/bash
# DeepTrace - Fast Face Swap - Startup Script for Linux/Mac

echo "========================================"
echo "  DeepTrace - Starting..."
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.10 or later"
    exit 1
fi

echo "[OK] Python found"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Virtual environment not found. Creating..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "[OK] Virtual environment created"
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate
echo "[OK] Virtual environment activated"
echo ""

# Check if dependencies are installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Installing dependencies..."
    pip install -r requirements_simple.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
    echo "[OK] Dependencies installed"
fi

echo ""
echo "========================================"
echo "  Starting DeepTrace Application"
echo "========================================"
echo ""
echo "  Open your browser and go to:"
echo "  http://localhost:5000"
echo ""
echo "  Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start the application
python simple_app.py

# Deactivate virtual environment on exit
deactivate
