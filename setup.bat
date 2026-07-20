@echo off
REM ===========================================================================
REM  DeepTrace - one-time developer setup.
REM  Run this ONCE after cloning. It:
REM    1. clones the generation engine (Deepfake-gen-pipeline) into deeptrace\
REM    2. creates the BACKEND venv (.venv) and installs its deps
REM    3. creates the ENGINE venv (deeptrace\venv) and installs its deps
REM    4. downloads the ONNX face-swap models (~2.3 GB, one time)
REM    5. creates backend\.env from the example
REM  After this finishes, run  start.bat  to launch everything.
REM ===========================================================================
setlocal
cd /d "%~dp0"

echo(
echo === [1/5] Generation engine (deeptrace\) =============================
if exist "deeptrace\api\main.py" (
    echo   deeptrace\ already present - skipping clone.
) else (
    echo   Cloning Deepfake-gen-pipeline into deeptrace\ ...
    git clone https://github.com/Ali7040/Deepfake-gen-pipeline.git deeptrace
    if errorlevel 1 ( echo   ERROR: git clone failed. & goto :fail )
)

echo(
echo === [2/5] Backend venv (.venv) + deps ================================
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 ( echo   ERROR: could not create .venv. Is Python 3.11+ installed? & goto :fail )
)
call .venv\Scripts\python -m pip install --upgrade pip
call .venv\Scripts\python -m pip install -r backend\requirements.txt
call .venv\Scripts\python -m pip install -r backend\requirements-ml.txt
if errorlevel 1 ( echo   ERROR: backend dependency install failed. & goto :fail )
REM facenet-pytorch (MTCNN) must go in separately with --no-deps: its stale
REM numpy<1.25 pin would otherwise force a from-source numpy build and fail.
call .venv\Scripts\python -m pip install --no-deps "facenet-pytorch>=2.6"
if errorlevel 1 ( echo   ERROR: facenet-pytorch install failed. & goto :fail )

echo(
echo === [3/5] Engine venv (deeptrace\venv) + deps =======================
if not exist "deeptrace\venv\Scripts\python.exe" (
    python -m venv deeptrace\venv
    if errorlevel 1 ( echo   ERROR: could not create deeptrace\venv. & goto :fail )
)
call deeptrace\venv\Scripts\python -m pip install --upgrade pip
call deeptrace\venv\Scripts\python -m pip install -r deeptrace\requirements_simple.txt
if errorlevel 1 ( echo   ERROR: engine dependency install failed. & goto :fail )

echo(
echo === [4/5] Download face-swap models (~2.3 GB, one time) =============
pushd deeptrace
call venv\Scripts\python download_hyperswap.py
call venv\Scripts\python deeptrace.py force-download
popd
echo   (If any model failed, re-run this step - it resumes/skips existing files.)

echo(
echo === [5/5] Backend config =============================================
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
