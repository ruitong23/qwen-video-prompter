@echo off
cd /d %~dp0

echo [1/5] Creating virtual environment...
python -m venv venv

echo [2/5] Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

echo [3/5] Installing Python dependencies...
"%~dp0venv\Scripts\pip.exe" install --upgrade pip
"%~dp0venv\Scripts\pip.exe" install -r requirements.txt

echo [4/5] Initializing Qwen2.5-VL submodule...
git submodule update --init --recursive

echo [5/5] Setup complete!
pause
exit /b
