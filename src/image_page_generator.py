
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Optional

import requests
import pymupdf as fitz
from PIL import Image


@dataclass
class GeneratedPageImage:
    png: bytes
    width: int
    height: int
    model: str
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    passes: int


class OpenRouterImagePageTranslator:
    """
    Full-page scanned-PDF translator.

    IMPORTANT:
    This implementation uses OpenRouter CHAT COMPLETIONS with the
    Gemini image model and requests BOTH image input and image output
    in the same call. It does not use OCR JSON or bounding boxes.

    Source page image:
        PDF page -> PNG -> Gemini image input

    Output:
        Gemini edited PNG -> exact source raster canvas -> PDF page
    """

    DEFAULT_MODEL = "google/gemini-3.1-flash-image-preview"
    CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str | None,
        source_language: str,
        target_language: str,
        dpi: int = 220,
        timeout: int = 300,
        max_retries: int = 2,
        edit_passes: int = 2,
        http_referer: str = "",
        app_name: str = "PDF Translator Studio",
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing.")

        self.api_key = api_key
        self.model = (model or self.DEFAULT_MODEL).strip()
        self.source_language = source_language
        self.target_language = target_language
        self.dpi = max(180, min(300, int(dpi)))
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.edit_passes = max(1, min(2, int(edit_passes)))

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if http_referer:
            self.headers["HTTP-Referer"] = http_referer
        if app_name:
            self.headers["X-Title"] = app_name

        self._cache: dict[str, dict] = {}

    @staticmethod
    def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
        scale = dpi / 72.0
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            alpha=False,
        )
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    @staticmethod
    def _png_bytes(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    @staticmethod
    def _data_url(png: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")

    @staticmethod
    def _aspect_ratio(width: int, height: int) -> str:
        ratio = width / max(height, 1)
        choices = {
            "1:1": 1.0,
            "4:5": 4 / 5,
            "3:4": 3 / 4,
            "2:3": 2 / 3,
            "9:16": 9 / 16,
            "16:9": 16 / 9,
            "3:2": 3 / 2,
            "5:4": 5 / 4,
        }
        return min(choices, key=lambda k: abs(choices[k] - ratio))

    def _prompt(self, verification: bool = False) -> str:
        if verification:
            return f"""
EDIT THE PROVIDED IMAGE. DO NOT CREATE A NEW DOCUMENT.

The attached image is a scanned technical PDF page.

Source language: {self.source_language}
Target language: {self.target_language}

This is a second proofreading pass. Inspect the ENTIRE image carefully.
Find every remaining readable {self.source_language} text, including:
- paragraphs
- headings
- table cells
- table headers
- diagram labels
- screenshots
- callouts
- annotations
- captions
- small labels
- footer/header text

Translate every remaining readable source-language phrase into {self.target_language}
and replace it directly in the image.

STRICT:
- Keep the original page as the visual canvas.
- Preserve every table, border, line, arrow, diagram, screenshot, logo,
  icon, color, photograph and drawing.
- Do not crop, rotate, redesign, or recompose.
- Do not add a watermark.
- Do not remove non-text artwork.
- Do not merely return the original image.
- The visible Japanese/source-language text must actually be replaced.
- Keep translated text in the same locations.
- Match the original font size, weight, alignment and line spacing as closely as possible.
- For text that does not fit, reduce its font size rather than moving the surrounding artwork.

Return the COMPLETE edited page image only.
""".strip()

        return f"""
EDIT THE PROVIDED IMAGE DIRECTLY. THIS IS AN IMAGE-EDITING TASK, NOT A
TEXT-TO-IMAGE TASK.

The attached image is a scanned technical document page.
Treat the attached image as the IMMUTABLE VISUAL CANVAS.

SOURCE LANGUAGE: {self.source_language}
TARGET LANGUAGE: {self.target_language}

Translate EVERY readable {self.source_language} text visible in the image
into accurate natural {self.target_language}, and replace that source text
DIRECTLY IN THE IMAGE.

You MUST visibly change the image by replacing the source-language text.
Do NOT simply reproduce, copy, summarize, describe, or return the original.

READ THE WHOLE PAGE:
- top header
- section headings
- paragraphs
- bullet points
- tables and table cells
- diagram labels
- arrows/callouts
- screenshots
- captions
- small annotations
- footer/header

LAYOUT PRESERVATION:
- Keep EXACTLY the same page composition and margins.
- Keep every table, border, line, arrow, diagram, screenshot, logo, icon,
  photograph and illustration in exactly the same position.
- Do not crop.
- Do not rotate.
- Do not zoom or reframe.
- Do not redesign the document.
- Do not invent graphics.
- Replace TEXT ONLY.
- Put each translated phrase in the same visual region as its source.
- Preserve alignment and visual hierarchy.
- Reduce translated font size when English is longer than the source.
- Do not move tables or diagrams to make room for text.

IMPORTANT FOR DENSE JAPANESE DOCUMENTS:
Do not stop after translating the large paragraph near the top.
Continue through every table, diagram, screenshot and small label.
Japanese text remaining anywhere on the page means the edit is incomplete.

Return ONE COMPLETE EDITED IMAGE of the page.
""".strip()

    @staticmethod
    def _extract_image_from_message(data: dict) -> bytes:
        # OpenRouter's chat image-output response uses:
        # choices[0].message.images[].image_url.url
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"OpenRouter returned no choices. Response: {str(data)[:3000]}"
            )

        message = (choices[0].get("message") or {})
        images = message.get("images") or []

        for item in images:
            image_url = item.get("image_url") if isinstance(item, dict) else None
            url = image_url.get("url") if isinstance(image_url, dict) else None

            if not url:
                continue

            if url.startswith("data:"):
                try:
                    _, encoded = url.split(",", 1)
                    return base64.b64decode(encoded)
                except Exception as exc:
                    raise RuntimeError(
                        "OpenRouter returned an invalid base64 image data URL."
                    ) from exc

            # Some providers may return a normal URL.
            if url.startswith("http://") or url.startswith("https://"):
                r = requests.get(url, timeout=self.timeout)
                r.raise_for_status()
                return r.content

        # Some response variants may expose image data directly.
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                image_url = part.get("image_url")
                if isinstance(image_url, dict) and image_url.get("url"):
                    url = image_url["url"]
                    if url.startswith("data:"):
                        _, encoded = url.split(",", 1)
                        return base64.b64decode(encoded)

        raise RuntimeError(
            "Gemini returned a text response but no generated image. "
            f"finish_reason={choices[0].get('finish_reason')!r}, "
            f"message_keys={list(message.keys())}, "
            f"content_preview={str(content)[:1200]!r}"
        )

    def _request(self, image: Image.Image, verification: bool = False) -> bytes:
        source_png = self._png_bytes(image)
        data_url = self._data_url(source_png)
        aspect = self._aspect_ratio(image.width, image.height)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._prompt(verification=verification),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            # This is the critical change:
            # Gemini receives the source image and is explicitly requested
            # to return an image in the SAME chat request.
            "modalities": ["image", "text"],
            "image_config": {
                "aspect_ratio": aspect,
                "image_size": "2K",
            },
        }

        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.CHAT_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code >= 400:
                    raise RuntimeError(
                        f"OpenRouter chat image-edit failed "
                        f"({response.status_code}) for {self.model}: "
                        f"{response.text[:4000]}"
                    )

                return self._extract_image_from_message(response.json())

            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(8, 2 ** attempt))

        raise RuntimeError(
            f"OpenRouter Gemini image-edit request failed for {self.model!r}: "
            f"{last_error}"
        ) from last_error

    @staticmethod
    def _fit_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        image = image.convert("RGB")
        if image.size == size:
            return image
        return image.resize(size, Image.Resampling.LANCZOS)

    def translate_page(self, page: fitz.Page, page_index: int) -> dict:
        source = self._render_page(page, self.dpi)
        source_size = source.size

        cache_key = (
            f"{page.rect.width:.3f}x{page.rect.height:.3f}:"
            f"{source_size}:{self.model}:{self.source_language}:"
            f"{self.target_language}:{self.edit_passes}"
        )

        if cache_key in self._cache:
            return self._cache[cache_key]

        current = source
        passes = 0

        for pass_index in range(self.edit_passes):
            raw = self._request(
                current,
                verification=(pass_index == 1),
            )

            try:
                current = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:
                raise RuntimeError(
                    "Gemini/OpenRouter returned data that is not a readable image."
                ) from exc

            passes += 1

        final = self._fit_canvas(current, source_size)
        png = self._png_bytes(final)

        result = {
            "png": png,
            "width": final.width,
            "height": final.height,
            "dpi": self.dpi,
            "model": self.model,
            "source_size": source_size,
            "output_size": current.size,
            "passes": passes,
            "mode": "gemini-chat-image-edit",
        }

        self._cache[cache_key] = result
        return result
