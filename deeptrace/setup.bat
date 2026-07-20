@echo off
REM ===========================================================================
REM  DeepTrace - one-time developer setup (consolidated repo).
REM  This repo holds BOTH the generation engine (api/, deeptrace/, simple_app.py)
REM  and the unified backend (backend/). Run this ONCE after cloning. It:
REM    1. creates the ENGINE venv (venv\) and installs its deps
REM    2. creates the BACKEND venv (backend\.venv) and installs its deps
REM    3. downloads the ONNX face-swap models (~2.3 GB, one time)
REM    4. creates backend\.env from the example
REM  Then run  start.bat  to launch everything.
REM ===========================================================================
setlocal
cd /d "%~dp0"

echo(
echo === [1/4] Engine venv (venv\) + deps ================================
if not exist "venv\Scripts\python.exe" (
    python -m venv venv
    if errorlevel 1 ( echo   ERROR: could not create venv. Is Python 3.11+ installed? & goto :fail )
)
call venv\Scripts\python -m pip install --upgrade pip
call venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 ( echo   ERROR: engine dependency install failed. & goto :fail )

echo(
echo === [2/4] Backend venv (backend\.venv) + deps ======================
if not exist "backend\.venv\Scripts\python.exe" (
    python -m venv backend\.venv
    if errorlevel 1 ( echo   ERROR: could not create backend\.venv. & goto :fail )
)
call backend\.venv\Scripts\python -m pip install --upgrade pip
call backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
call backend\.venv\Scripts\python -m pip install -r backend\requirements-ml.txt
if errorlevel 1 ( echo   ERROR: backend dependency install failed. & goto :fail )
REM facenet-pytorch (MTCNN) must go in with --no-deps: its stale numpy<1.25 pin
REM would otherwise force a from-source numpy build and fail.
call backend\.venv\Scripts\python -m pip install --no-deps "facenet-pytorch>=2.6"
if errorlevel 1 ( echo   ERROR: facenet-pytorch install failed. & goto :fail )

echo(
echo === [3/4] Download face-swap models (~2.3 GB, one time) =============
call venv\Scripts\python download_hyperswap.py
call venv\Scripts\python deeptrace.py force-download
echo   (If any model failed, re-run this step - it resumes/skips existing files.)

echo(
echo === [4/4] Backend config ===========================================
if not exist "backend\.env" (
    copy /Y "backend\.env.example" "backend\.env" >nul
    echo   Created backend\.env  (edit SECRET_KEY / HF_TOKEN as needed).
) else (
    echo   backend\.env already exists - leaving it.
)

echo(
echo =====================================================================
echo  SETUP COMPLETE.  Now run:   start.bat
echo =====================================================================
goto :eof

:fail
echo(
echo SETUP FAILED - see the error above. Fix it and re-run setup.bat.
exit /b 1
