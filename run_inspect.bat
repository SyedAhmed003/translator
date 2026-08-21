@echo off
setlocal
cd /d "%~dp0"
python translate.py inspect sample_input\info_20260807_jp.pdf --source Japanese
pause
