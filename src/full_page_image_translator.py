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
from PIL import Image, ImageDraw, ImageFont


@dataclass
class FullPageRegion:
    bbox_px: tuple[float, float, float, float]
    source_text: str
    translation: str
    confidence: float = 0.0


def _message_content(message) -> str:
    """Extract assistant text across OpenAI/OpenRouter SDK response variants."""
    if message is None:
        return ""

    # Standard Chat Completions text.
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content

    # Chat Completions content-part arrays.
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(part, "text", None) or getattr(part, "content", None)
                if text:
                    parts.append(str(text))
        joined = "".join(parts).strip()
        if joined:
            return joined

    # Some SDK/provider variants expose parsed structured output.
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        try:
            if isinstance(parsed, str):
                return parsed
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            pass

    # Responses-style compatibility fields.
    output_text = getattr(message, "output_text", None)
    if output_text:
        return str(output_text)

    # Tool-call fallback: accept function arguments if a provider returns
    # the structured payload through a tool call instead of content.
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            fn = getattr(call, "function", None)
            args = getattr(fn, "arguments", None) if fn is not None else None
            if args:
                return str(args)

    return ""


def _response_debug(response) -> str:
    """Small diagnostic summary for empty/malformed model output."""
    try:
        choice = response.choices[0]
        msg = choice.message
        finish = getattr(choice, "finish_reason", None)
        refusal = getattr(msg, "refusal", None)
        content = getattr(msg, "content", None)
        return (
            f"finish_reason={finish!r}, refusal={refusal!r}, "
            f"content_type={type(content).__name__}, content_len={len(content) if isinstance(content, str) else None}"
        )
    except Exception:
        return "Unable to inspect OpenRouter response."


def _extract_json(content: str) -> dict:
    """Parse JSON from raw JSON, markdown fences, or surrounding prose."""
    content = (content or "").strip()
    if not content:
        raise json.JSONDecodeError("Empty model response", "", 0)

    # Strip common markdown fences.
    if content.startswith("```"):
        lines = content.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Find a balanced JSON object while respecting quoted strings/escapes.
    start = content.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(content)):
            ch = content[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[start:i + 1]
                    return json.loads(candidate)

    raise json.JSONDecodeError("No valid JSON object found", content, 0)


def _valid_box(box, width, height):
    try:
        if isinstance(box, dict):
            if all(k in box for k in ("x0", "y0", "x1", "y1")):
                vals = [box["x0"], box["y0"], box["x1"], box["y1"]]
            elif all(k in box for k in ("x", "y", "width", "height")):
                vals = [box["x"], box["y"], float(box["x"]) + float(box["width"]), float(box["y"]) + float(box["height"])]
            elif all(k in box for k in ("left", "top", "right", "bottom")):
                vals = [box["left"], box["top"], box["right"], box["bottom"]]
            else:
                return None
        else:
            vals = list(box)
        if len(vals) != 4:
            return None
        x0, y0, x1, y1 = [float(v) for v in vals]
    except Exception:
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    # Accept normalized 0..1 or Google-style 0..1000 coordinates as well as pixels.
    vmax = max(abs(x0), abs(y0), abs(x1), abs(y1))
    if vmax <= 1.00001:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    elif vmax <= 1000.5 and (width > 1100 or height > 1100):
        x0, x1 = x0 * width / 1000.0, x1 * width / 1000.0
        y0, y1 = y0 * height / 1000.0, y1 * height / 1000.0
    x0, x1 = sorted((max(0.0, x0), min(float(width), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(height), y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _region_items(data):
    if not isinstance(data, dict):
        return []
    for key in ("regions", "text_regions", "items", "translations", "elements", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    # Some models return a single region object.
    if any(k in data for k in ("bbox", "bounding_box", "box_2d", "source_text", "translation", "translated_text")):
        return [data]
    return []


def _normalize_region(region):
    if not isinstance(region, dict):
        return None
    box = (region.get("bbox") or region.get("bounding_box") or region.get("box") or
           region.get("box_2d") or region.get("coordinates"))
    source = (region.get("source_text") or region.get("source") or region.get("original_text") or
              region.get("text") or region.get("ocr_text") or "").strip()
    translation = (region.get("translation") or region.get("translated_text") or
                   region.get("target_text") or region.get("translated") or
                   region.get("target") or "").strip()
    if not box or not translation:
        return None
    try:
        confidence = float(region.get("confidence", region.get("score", 0.0)))
    except Exception:
        confidence = 0.0
    return box, source, translation, confidence


def _render_page(page: fitz.Page, dpi: int) -> tuple[Image.Image, float]:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"), scale


def _encode_image(image: Image.Image, max_edge: int):
    original_size = image.size
    scale = 1.0
    if max(original_size) > max_edge:
        scale = max_edge / float(max(original_size))
        size = (
            max(1, int(original_size[0] * scale)),
            max(1, int(original_size[1] * scale)),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)
    buff = io.BytesIO()
    image.save(buff, format="PNG", optimize=True)
    return base64.b64encode(buff.getvalue()).decode("ascii"), image.size, scale


class FullPageVisionTranslator:
    """Translate an image-only PDF page as one complete visual canvas."""

    def __init__(self, api_key, model, source_language, target_language, max_retries=3,
                 max_image_edge=3000, dpi=220, http_referer="", app_name="PDF Translator Studio"):
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
        self.dpi = dpi
        self._seen = {}

    def _prompt(self, width: int, height: int) -> str:
        return f"""
Analyze this entire PDF page image as a publication-quality OCR and translation task.

Source language: {self.source_language}
Target language: {self.target_language}
Image size: {width} x {height} pixels.

This is a COMPLETE PAGE, not an isolated photograph. Read all visible source-language text,
including:
- normal paragraphs and headings
- table cells
- labels inside diagrams
- text inside screenshots/figures
- small annotations, callouts, arrows and legends
- dates, numbers, codes and technical identifiers

Do NOT translate logos, decorative shapes, lines, icons or artwork unless they contain readable text.
Do NOT summarize. Do not omit readable text merely because it is small.

For each distinct readable text region, return a tight bounding box in THIS IMAGE'S PIXEL COORDINATES.
The origin is top-left. Do not use PDF points and do not use normalized 0..1 coordinates.
Translate each region faithfully from {self.source_language} to {self.target_language}.
Preserve numbers, codes, names, units and symbols.

The output will be used to rebuild the entire page image. Therefore boxes must be accurate and
translations must stay short enough to fit the original visual region.
""".strip()

    @staticmethod
    def _schema():
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "pdf_page_translation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "regions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                    "source_text": {"type": "string"},
                                    "translation": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["bbox", "source_text", "translation", "confidence"],
                            },
                        }
                    },
                    "required": ["regions"],
                },
            },
        }

    def _call(self, image_b64: str, size: tuple[int, int], structured="schema"):
        """
        structured:
          - "schema": strict JSON schema
          - "json_object": generic JSON object mode
          - False: plain text fallback
        """
        data_url = f"data:image/png;base64,{image_b64}"
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert OCR and document translator. "
                                "Return ONLY the requested JSON object. "
                                "Do not add explanations, markdown or commentary."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self._prompt(size[0], size[1])},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        },
                    ],
                    temperature=0,
                    max_tokens=16000,
                )
                if structured == "schema":
                    kwargs["response_format"] = self._schema()
                elif structured == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}

                return self.client.chat.completions.create(**kwargs)

            except (RateLimitError, APIStatusError) as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)

                # If strict schema is rejected, retry with generic JSON and
                # finally plain text. This is useful across OpenRouter providers.
                if status in (400, 404, 422):
                    if structured == "schema":
                        return self._call(image_b64, size, structured="json_object")
                    if structured == "json_object":
                        return self._call(image_b64, size, structured=False)

                retryable = isinstance(exc, RateLimitError) or status in (429, 500, 502, 503, 504)
                if retryable and attempt < self.max_retries:
                    time.sleep(min(20, 2 ** attempt))
                    continue
                raise

        raise RuntimeError(
            f"OpenRouter vision request failed for model {self.model!r}."
        ) from last_error

    def _analyze(self, image: Image.Image):
        b64, sent_size, scale = _encode_image(image, self.max_image_edge)

        # Try strict schema first, then generic JSON, then plain text.
        attempts = ["schema", "json_object", False]
        errors = []
        last_debug = ""

        for mode in attempts:
            try:
                response = self._call(b64, sent_size, structured=mode)
                message = response.choices[0].message
                content = _message_content(message)
                last_debug = _response_debug(response)

                # Empty content is not a JSON parse error; it is an empty
                # model response and should trigger the next request mode.
                if not content.strip():
                    errors.append(f"{mode}: empty content ({last_debug})")
                    continue

                try:
                    data = _extract_json(content)
                    if not isinstance(data, dict):
                        errors.append(f"{mode}: JSON root was {type(data).__name__}, expected object")
                        continue
                    break
                except json.JSONDecodeError as exc:
                    errors.append(f"{mode}: {exc}")
                    continue

            except Exception as exc:
                errors.append(f"{mode}: request failed: {exc}")
        else:
            # Last-resort prompt: ask for the smallest possible JSON response.
            # This reduces the chance that a lightweight multimodal model
            # spends all output on prose instead of returning the payload.
            fallback_prompt = (
                f"Read all readable {self.source_language} text in this page image and translate it "
                f"to {self.target_language}. Return ONLY valid JSON in exactly this shape: "
                '{"regions":[{"bbox":[x0,y0,x1,y1],"source_text":"...","translation":"...","confidence":0.0}]} '
                f"Image size is {sent_size[0]}x{sent_size[1]} pixels. "
                "Use top-left pixel coordinates. Do not use markdown."
            )
            try:
                data_url = f"data:image/png;base64,{b64}"
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return only valid JSON."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": fallback_prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        },
                    ],
                    temperature=0,
                    max_tokens=16000,
                )
                content = _message_content(response.choices[0].message)
                last_debug = _response_debug(response)
                data = _extract_json(content)
            except Exception as exc:
                errors.append(f"fallback: {exc}")
                raise RuntimeError(
                    "Gemini/OpenRouter did not return usable JSON for the full-page vision request. "
                    f"Attempts: {' | '.join(errors)}. Last response: {last_debug}"
                ) from exc

        result = []
        inv = 1.0 / max(scale, 1e-9)
        for raw_region in _region_items(data):
            normalized = _normalize_region(raw_region)
            if not normalized:
                continue
            raw_box, source, translation, confidence = normalized
            box = _valid_box(raw_box, sent_size[0], sent_size[1])
            if not box or not translation:
                continue
            mapped = _valid_box(tuple(v * inv for v in box), image.width, image.height)
            if not mapped:
                continue
            confidence = max(0.0, min(1.0, confidence))
            result.append(FullPageRegion(mapped, source, translation, confidence))

        if not result:
            # One final schema deliberately uses the most common Gemini OCR field names.
            compact_prompt = (
                f"Read the page image and translate every readable {self.source_language} text region "
                f"into {self.target_language}. Return ONLY JSON. Use this exact shape: "
                '{"regions":[{"bbox":[x0,y0,x1,y1],"source_text":"...","translation":"..."}]} ' 
                f"Use pixel coordinates for an image of {sent_size[0]} by {sent_size[1]} pixels. "
                "Do not return markdown or prose. For bbox, x0,y0 is top-left and x1,y1 is bottom-right."
            )
            try:
                data_url = f"data:image/png;base64,{b64}"
                retry = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "OCR translator. Return only JSON."},
                        {"role": "user", "content": [
                            {"type": "text", "text": compact_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]},
                    ],
                    temperature=0,
                    max_tokens=16000,
                )
                retry_content = _message_content(retry.choices[0].message)
                retry_debug = _response_debug(retry)
                retry_data = _extract_json(retry_content)
                for raw_region in _region_items(retry_data):
                    normalized = _normalize_region(raw_region)
                    if not normalized:
                        continue
                    raw_box, source, translation, confidence = normalized
                    box = _valid_box(raw_box, sent_size[0], sent_size[1])
                    if not box or not translation:
                        continue
                    mapped = _valid_box(tuple(v * inv for v in box), image.width, image.height)
                    if mapped:
                        result.append(FullPageRegion(mapped, source, translation, max(0.0, min(1.0, confidence))))
                if not result:
                    raise RuntimeError(f"retry returned no usable regions ({retry_debug})")
            except Exception as exc:
                raise RuntimeError(
                    "Gemini/OpenRouter returned JSON but no usable translated text regions were found. "
                    f"Initial response: {last_debug}. Final retry: {exc}"
                ) from exc
        return result

    @staticmethod
    def _font_path():
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for p in candidates:
            try:
                if __import__("pathlib").Path(p).exists():
                    return p
            except Exception:
                pass
        return None

    @staticmethod
    def _wrap(draw, text, font, width):
        text = " ".join((text or "").split())
        if not text:
            return []
        tokens = list(text) if " " not in text else text.split()
        sep = "" if " " not in text else " "
        lines, cur = [], ""
        for token in tokens:
            candidate = token if not cur else cur + sep + token
            if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
                cur = candidate
                continue
            if cur:
                lines.append(cur)
            cur = token
        if cur:
            lines.append(cur)
        return lines

    def _inpaint(self, image, regions):
        try:
            import cv2
            import numpy as np
        except Exception:
            cv2 = np = None
        if cv2 is None:
            # No OpenCV: use a conservative local median fill, never white.
            out = image.copy()
            px = out.load()
            for r in regions:
                x0, y0, x1, y1 = [int(round(v)) for v in r.bbox_px]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(out.width, x1), min(out.height, y1)
                samples = []
                for x in range(x0, x1, max(1, (x1-x0)//8 or 1)):
                    if y0 > 0: samples.append(px[x, y0-1])
                    if y1 < out.height: samples.append(px[x, y1])
                if not samples: continue
                bg = tuple(sorted(c[i] for c in samples)[len(samples)//2] for i in range(3))
                for yy in range(y0, y1):
                    for xx in range(x0, x1): px[xx, yy] = bg
            return out
        rgb = np.array(image.convert("RGB"))
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for r in regions:
            x0, y0, x1, y1 = [int(round(v)) for v in r.bbox_px]
            pad = max(2, int(min(image.width, image.height) * 0.0015))
            cv2.rectangle(mask, (max(0, x0-pad), max(0, y0-pad)),
                          (min(image.width-1, x1+pad), min(image.height-1, y1+pad)), 255, -1)
        restored = cv2.inpaint(rgb, mask, 3.0, cv2.INPAINT_TELEA)
        return Image.fromarray(restored).convert("RGB")

    def _draw_translations(self, image, regions):
        draw = ImageDraw.Draw(image)
        font_path = self._font_path()
        for region in sorted(regions, key=lambda r: (r.bbox_px[1], r.bbox_px[0])):
            x0, y0, x1, y1 = region.bbox_px
            width = max(8, int(x1-x0-6))
            height = max(8, int(y1-y0-6))
            start = max(8, int(height * 0.78))
            chosen = None
            for size in range(start, 5, -1):
                font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
                lines = self._wrap(draw, region.translation, font, width)
                if not lines:
                    continue
                boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
                line_h = max(1, max(b[3]-b[1] for b in boxes))
                spacing = max(1, int(line_h * 0.08))
                if len(lines)*line_h + (len(lines)-1)*spacing <= height:
                    chosen = (font, lines, line_h, spacing)
                    break
            if chosen is None:
                font = ImageFont.truetype(font_path, 5) if font_path else ImageFont.load_default()
                lines = self._wrap(draw, region.translation, font, width)[:max(1, height//7)]
                boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
                line_h = max(1, max((b[3]-b[1] for b in boxes), default=5))
                spacing = 1
                chosen = (font, lines, line_h, spacing)
            font, lines, line_h, spacing = chosen
            y = int(y0 + 2)
            for line in lines:
                draw.text((int(x0 + 2), y), line, font=font, fill=(0, 0, 0))
                y += line_h + spacing
        return image

    def translate_page(self, page: fitz.Page, page_index: int):
        source_image, scale = _render_page(page, self.dpi)
        digest = sha256(source_image.tobytes()).hexdigest()
        if digest in self._seen:
            return self._seen[digest]
        regions = self._analyze(source_image)
        rebuilt = self._inpaint(source_image, regions)
        rebuilt = self._draw_translations(rebuilt, regions)
        buff = io.BytesIO()
        rebuilt.save(buff, format="PNG", optimize=True)
        result = {"png": buff.getvalue(), "width": rebuilt.width, "height": rebuilt.height,
                  "dpi": self.dpi, "regions": regions}
        self._seen[digest] = result
        return result
