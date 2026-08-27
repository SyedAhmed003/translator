@echo off
setlocal
cd /d "%~dp0"
echo Installing OCR extras...
python -m pip install -r requirements-ocr.txt
if errorlevel 1 (
  echo.
  echo OCR installation failed. Check the Python version and PaddlePaddle Windows support for your environment.
  pause
  exit /b 1
)
echo.
echo OCR extras installed.
echo Run check_ocr.py to verify the backend.
pause
