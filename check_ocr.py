import os
from pathlib import Path

print("OCR diagnostics")
print("POPPLER_PATH:", os.getenv("POPPLER_PATH", "<not set>"))
print("TESSERACT_CMD:", os.getenv("TESSERACT_CMD", "<not set>"))
try:
    import paddlex
    print("PaddleX: OK", getattr(paddlex, "__version__", ""))
except Exception as e:
    print("PaddleX: NOT AVAILABLE", e)
try:
    import paddle
    print("PaddlePaddle: OK", getattr(paddle, "__version__", ""))
except Exception as e:
    print("PaddlePaddle: NOT AVAILABLE", e)
try:
    import pytesseract
    print("pytesseract: OK")
except Exception as e:
    print("pytesseract: NOT AVAILABLE", e)
