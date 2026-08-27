import sys
import pymupdf
import openai

sys.path.insert(0, '.')
from src.renderer import render_translated_pdf
from src.image_translator import OpenRouterVisionTranslator
from src.pipeline import translate_document_with_options

print("Python:", sys.version)
print("PyMuPDF:", pymupdf.__doc__.splitlines()[0])
print("OpenAI SDK:", openai.__version__)
print("Project imports: OK")
print("OpenRouter text + vision modules: OK")
print("Renderer image-preservation path: OK")
