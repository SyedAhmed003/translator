@echo off
setlocal
cd /d "%~dp0"
set /p MODEL=Enter OpenRouter model ID: 
python -m src.cli translate sample_input\info_20260807_jp.pdf --model "%MODEL%" --source Japanese --target English
pause
