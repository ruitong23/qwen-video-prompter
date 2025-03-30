@echo off
cd /d %~dp0

echo [1/4] Creating virtual environment...
python -m venv venv

echo [2/4] Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

echo [3/4] Installing requirements...
"%~dp0venv\Scripts\pip.exe" install --upgrade pip
"%~dp0venv\Scripts\pip.exe" install -r requirements.txt

echo [4/4] Setup complete!
pause
exit /b
