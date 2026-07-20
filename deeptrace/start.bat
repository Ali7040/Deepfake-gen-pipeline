@echo off
REM ===========================================================================
REM  DeepTrace - run the backend. It AUTO-STARTS the generation engine, so this
REM  single command brings up BOTH detection and generation.
REM    Frontend / API : http://localhost:8080      (this is what the frontend uses)
REM    API docs        : http://localhost:8080/docs
REM    Gen engine      : http://localhost:8000      (spawned automatically)
REM  First time? Run  setup.bat  before this.
REM  (Ctrl+C stops the backend AND the engine it started.)
REM  Legacy Flask UI is still available via:  venv\Scripts\python simple_app.py
REM ===========================================================================
cd /d "%~dp0backend"

if not exist ".venv\Scripts\python.exe" (
  echo Backend venv missing. Run setup.bat first.
  pause & exit /b 1
)

echo Starting DeepTrace backend on :8080 (auto-starting gen engine on :8000) ...
echo Detection is ready immediately; generation is ready once the engine finishes
echo loading its models (a few seconds after boot).
echo(
.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
