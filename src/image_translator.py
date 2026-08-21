from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import pymupdf as fitz
from openai import OpenAI, RateLimitError, APIStatusError
from PIL import Image


@dataclass
class ImageTextRegion:
    region_id: str
    page_index: int
    xref: int
    occurrence_index: int
    image_rect: tuple[float, float, float, float]
    image_width: int
    image_height: int
    bbox_px: tuple[float, float, float, float]
    source_text: str
    translation: str
    confidence: float = 0.0

    @property
    def page_bbox(self):
        ix0, iy0, ix1, iy1 = self.bbox_px
        rx0, ry0, rx1, ry1 = self.image_rect
        sx = (rx1 - rx0) / max(1, self.image_width)
        sy = (ry1 - ry0) / max(1, self.image_height)
        return (
            rx0 + ix0 * sx,
            ry0 + iy0 * sy,
            rx0 + ix1 * sx,
            ry0 + iy1 * sy,
        )


def _image_payload(image_bytes: bytes, ext: str, max_edge: int):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    scale = 1.0
    if max(original_size) > max_edge:
        scale = max_edge / float(max(original_size))
        size = (max(1, int(original_size[0] * scale)), max(1, int(original_size[1] * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    buff = io.BytesIO()
    image.save(buff, format="PNG", optimize=True)
    payload = base64.b64encode(buff.getvalue()).decode("ascii")
    return payload, "image/png", original_size, image.size, scale


def _message_content(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content or "")


def _parse_json(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        parts = content.split("\n", 1)
        content = parts[1] if len(parts) == 2 else content
        if content.endswith("```"):
            content = content[:-3].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise


def _valid_box(box, width, height):
    try:
        x0, y0, x1, y1 = [float(v) for v in box]
    except Exception:
        return None
    x0, x1 = sorted((max(0.0, x0), min(float(width), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(height), y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


class OpenRouterVisionTranslator:
    """Extract text from embedded PDF images and translate it with a vision model."""

    def __init__(
        self,
        api_key: str,
        model: str,
        source_language: str,
        target_language: str,
        max_retries: int = 4,
        max_image_edge: int = 2400,
        http_referer: str = "",
        app_name: str = "PDF Translator Studio",
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Put it in your .env file.")
        headers = {}
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if app_name:
            headers["X-Title"] = app_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers,
            max_retries=0,
        )
        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.max_retries = max_retries
        self.max_image_edge = max_image_edge
        self._seen = {}

    def _prompt(self, width: int, height: int) -> str:
        return f"""
You are doing OCR and document translation on a PDF image.

Source language: {self.source_language}
Target language: {self.target_language}
Image pixel size: {width} x {height}

Detect ONLY text that is visibly present in this image. Ignore photographs, logos, decorative
shapes, borders, icons, and non-text artwork unless they contain readable text.

For every readable text region, return its bounding box in ORIGINAL IMAGE PIXEL COORDINATES
(x0, y0, x1, y1), where (0,0) is the top-left corner. Keep boxes tight around the text.

Translate every detected text region faithfully. Preserve numbers, dates, codes, URLs, names,
currencies and symbols. Keep short labels short. Do not summarize.

Return ONLY valid JSON using exactly this top-level shape:
{{
  "regions": [
    {{
      "bbox": [x0, y0, x1, y1],
      "source_text": "...",
      "translation": "...",
      "confidence": 0.0
    }}
  ]
}}

Use an empty regions array when no readable text is present.
""".strip()

    def _call(self, image_b64: str, mime: str, width: int, height: int):
        data_url = f"data:{mime};base64,{image_b64}"
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert OCR and publication-quality document translator. Return JSON only.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self._prompt(width, height)},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        },
                    ],
                    temperature=0,
                    max_tokens=6000,
                )
            except RateLimitError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(30, 2 ** attempt))
            except APIStatusError as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(30, 2 ** attempt))
                    continue
                raise
        raise RuntimeError(
            f"OpenRouter vision model '{self.model}' could not analyze the image after "
            f"{self.max_retries + 1} attempts. Make sure the selected model supports image input."
        ) from last_error

    def analyze_image(self, image_bytes: bytes, ext: str) -> list[dict[str, Any]]:
        digest = sha256(image_bytes).hexdigest()
        if digest in self._seen:
            return self._seen[digest]
        payload, mime, original_size, sent_size, scale = _image_payload(
            image_bytes, ext, self.max_image_edge
        )
        response = self._call(payload, mime, sent_size[0], sent_size[1])
        parsed = _parse_json(_message_content(response.choices[0].message))
        result = []
        for region in parsed.get("regions", []):
            box = _valid_box(region.get("bbox"), sent_size[0], sent_size[1])
            text = str(region.get("source_text", "")).strip()
            translation = str(region.get("translation", "")).strip()
            if not box or not text or not translation:
                continue
            # Model sees a resized version, so map coordinates back to the exact source image.
            inv = 1.0 / max(scale, 1e-9)
            mapped = tuple(v * inv for v in box)
            mapped = _valid_box(mapped, original_size[0], original_size[1])
            if not mapped:
                continue
            try:
                conf = float(region.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            result.append(
                {
                    "bbox_px": mapped,
                    "source_text": text,
                    "translation": translation,
                    "confidence": max(0.0, min(1.0, conf)),
                }
            )
        self._seen[digest] = result
        return result

    def extract_page_regions(self, page: fitz.Page, page_index: int) -> list[ImageTextRegion]:
        regions: list[ImageTextRegion] = []
        seen_xrefs = set()
        for image_index, image_info in enumerate(page.get_images(full=True)):
            xref = int(image_info[0])
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            extracted = page.parent.extract_image(xref)
            if not extracted or not extracted.get("image"):
                continue
            image_bytes = extracted["image"]
            ext = extracted.get("ext", "png")
            try:
                with Image.open(io.BytesIO(image_bytes)) as im:
                    width, height = im.size
            except Exception:
                continue
            occurrences = page.get_image_rects(xref)
            boxes = self.analyze_image(image_bytes, ext)
            for occurrence_index, rect in enumerate(occurrences):
                for region_index, item in enumerate(boxes, 1):
                    regions.append(
                        ImageTextRegion(
                            region_id=f"p{page_index + 1}_img{xref}_{occurrence_index}_r{region_index}",
                            page_index=page_index,
                            xref=xref,
                            occurrence_index=occurrence_index,
                            image_rect=(rect.x0, rect.y0, rect.x1, rect.y1),
                            image_width=width,
                            image_height=height,
                            bbox_px=tuple(item["bbox_px"]),
                            source_text=item["source_text"],
                            translation=item["translation"],
                            confidence=item["confidence"],
                        )
                    )
        return regions


class NullVisionTranslator:
    def extract_page_regions(self, page, page_index):
        return []
