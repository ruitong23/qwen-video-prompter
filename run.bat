@echo off
cd /d %~dp0

call "%~dp0venv\Scripts\activate.bat"
"%~dp0venv\Scripts\python.exe" batch_video_prompt_qwen2.5.py
pause
exit /b
