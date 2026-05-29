@echo off
REM SynthID Remover - Quick Start Script (Windows)

echo ==================================
echo SynthID Remover - Quick Start
echo ==================================
echo.

REM Check Python
python --version

REM Create venv if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check models
echo.
echo Checking models...
python download_models.py --check

if errorlevel 1 (
    echo.
    echo Models missing. Download now? (y/n)
    set /p response=
    if "%response%"=="y" python download_models.py
)

REM Launch
echo.
echo Launching SynthID Remover...
python main.py

pause
