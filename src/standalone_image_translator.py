from __future__ import annotations

import base64
import io
from typing import Any

import requests
from openai import OpenAI
from PIL import Image

DEFAULT_IMAGE_MODEL = "google/gemini-2.5-flash-image"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _client(api_key: str) -> OpenAI:
    if not api_key or not api_key.strip():
        raise RuntimeError("OPENROUTER_API_KEY is missing.")
    return OpenAI(
        api_key=api_key.strip(),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://translator.local",
            "X-Title": "Document Translator Studio",
        },
        max_retries=0,
    )


def _png_data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _normalize_image(data: bytes) -> tuple[bytes, tuple[int, int]]:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Uploaded file is not a readable image: {exc}") from exc

    size = image.size
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), size


def _extract_image(response: Any) -> bytes:
    if not getattr(response, "choices", None):
        raise RuntimeError("OpenRouter returned no choices.")

    message = response.choices[0].message
    images = getattr(message, "images", None) or []

    for item in images:
        if isinstance(item, dict):
            url = (item.get("image_url") or {}).get("url")
        else:
            image_url = getattr(item, "image_url", None)
            url = getattr(image_url, "url", None) if image_url else None

        if not url:
            continue

        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])

        if url.startswith(("http://", "https://")):
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            return r.content

    content = getattr(message, "content", None)
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            url = (part.get("image_url") or {}).get("url")
            if not url:
                continue
            if url.startswith("data:"):
                return base64.b64decode(url.split(",", 1)[1])
            if url.startswith(("http://", "https://")):
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                return r.content

    raise RuntimeError(
        "The selected image model returned text/no image instead of an edited image."
    )


def translate_image(
    input_bytes: bytes,
    api_key: str,
    source_language: str,
    target_language: str,
    model: str = DEFAULT_IMAGE_MODEL,
) -> bytes:
    """Translate all visible source-language text in one standalone image."""
    source_png, source_size = _normalize_image(input_bytes)

    prompt = f"""
You are translating an existing document/image, not creating a new picture.

SOURCE LANGUAGE: {source_language}
TARGET LANGUAGE: {target_language}
ORIGINAL IMAGE SIZE: {source_size[0]} x {source_size[1]} pixels

TASK
Translate EVERY readable {source_language} phrase in the supplied image into
accurate, natural {target_language} and return the COMPLETE edited image.

PRESERVE THE ORIGINAL IMAGE AS THE VISUAL MASTER.
Do not redesign it.
Do not recreate it from memory.
Do not change the composition.
Do not change the canvas aspect ratio.
Do not crop, rotate, zoom, reframe or add margins.
Do not move, resize or remove non-text objects.
Preserve exactly:
- tables and cell geometry
- diagrams and schematics
- arrows and connectors
- borders and lines
- logos and icons
- photos and illustrations
- colors and backgrounds
- numbers and symbols

TEXT COVERAGE
Translate all visible source-language text, including:
- headings
- paragraphs
- table text
- captions
- diagram labels
- callouts
- annotations
- legends
- small labels
- footer/header text

TECHNICAL FIDELITY
Keep unchanged unless it is ordinary-language text:
- numbers
- dates
- units
- equations
- component/reference identifiers
- model/part numbers
- figure/table numbers
- chemical notation
- URLs
- filenames

TEXT PLACEMENT
Keep each translation in the same visual location as the source text.
Match approximate font size, weight, alignment, line spacing and hierarchy.
If translated text is longer, reduce font size modestly or wrap within the
same original text area. Never move the surrounding artwork to make room.

OUTPUT
Return ONE complete edited image only.
No explanation. No Markdown. No watermark.
""".strip()

    client = _client(api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(source_png)},
                    },
                ],
            }
        ],
        modalities=["image", "text"],
    )

    result = _extract_image(response)

    # Ensure the returned artifact is a valid image.
    try:
        Image.open(io.BytesIO(result)).verify()
    except Exception as exc:
        raise RuntimeError(f"Image model returned invalid image data: {exc}") from exc

    return result
