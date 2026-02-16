@echo off
REM Windows batch script to run security scans locally
REM This script runs the same security checks as the CI/CD pipeline

echo 🔍 Running local security and quality checks for puffinflow
echo ============================================================

REM Change to project root
cd /d "%~dp0\.."

REM Run the Python security scan script
python scripts\run-security-scan.py

REM Pause to see results (optional, remove if running in CI)
if "%1" neq "--no-pause" (
    echo.
    echo Press any key to continue...
    pause >nul
)
