@echo off
REM DeepTrace - Fast Face Swap - Startup Script for Windows

echo ========================================
echo   DeepTrace - Starting...
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10 or later
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Check if dependencies are installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements_simple.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
)

echo.
echo ========================================
echo   Starting DeepTrace Application
echo ========================================
echo.
echo   Open your browser and go to:
echo   http://localhost:5000
echo.
echo   Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the application
python simple_app.py

REM Deactivate virtual environment on exit
deactivate

pause
