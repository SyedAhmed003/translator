from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_http_referer: str = os.getenv("OPENROUTER_HTTP_REFERER", "")
    openrouter_app_name: str = os.getenv("OPENROUTER_APP_NAME", "PDF Translator Studio")
    min_font_scale: float = float(os.getenv("MIN_FONT_SCALE", "0.60"))
    font_scale_step: float = float(os.getenv("FONT_SCALE_STEP", "0.02"))
    text_margin: float = float(os.getenv("TEXT_MARGIN", "0.0"))
    max_translation_chars: int = int(os.getenv("MAX_TRANSLATION_CHARS", "12000"))
    max_image_edge: int = int(os.getenv("MAX_IMAGE_EDGE", "2400"))
    cache_db: Path = ROOT / os.getenv("CACHE_DB", "data/translations.sqlite3")


settings = Settings()
