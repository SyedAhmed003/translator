
from __future__ import annotations

import os


DEFAULT_NATIVE_TEXT_MODEL = os.getenv(
    "NATIVE_TEXT_MODEL",
    "google/gemini-3.7-flash",
)

DEFAULT_IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL",
    "google/gemini-3.1-flash-lite-image",
)


def native_model(model_from_ui: str | None) -> str:
    value = (model_from_ui or "").strip()
    return value or DEFAULT_NATIVE_TEXT_MODEL


def image_model(model_from_ui: str | None) -> str:
    value = (model_from_ui or "").strip()
    return value or DEFAULT_IMAGE_MODEL
