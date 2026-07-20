@echo off
cd /d "%~dp0"

echo ========================================
echo   DeepTrace v2 - Clean Restart
echo ========================================
echo.

echo [INFO] Stopping any running Python processes...
taskkill /f /im python.exe /t >nul 2>&1
taskkill /f /im pythonw.exe /t >nul 2>&1
timeout /t 2 /nobreak >nul
echo [OK] Old processes cleared
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Activate virtual environment
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Install deps if missing
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements_simple.txt
)

echo.
echo ========================================
echo   Starting DeepTrace v2
echo ========================================
echo.
echo   Open your browser and go to:
echo   http://localhost:5000
echo.
echo   From another device on same WiFi:
echo   http://192.168.18.78:5000
echo.
echo   Press Ctrl+C to stop
echo ========================================
echo.

python simple_app.py

deactivate
pause
