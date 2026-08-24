
from __future__ import annotations

import base64
import copy
import io
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image


@dataclass
class ImageTranslationResult:
    sheet: str
    index: int
    model: str
    original_size: tuple[int, int]
    generated_size: tuple[int, int]


def _data_url(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")


def _media_type(name: str) -> str:
    ext = Path(name).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/png")


class ExcelImageTranslator:
    """
    Generate a translated version of each Excel image and put it back using
    the original anchor and displayed size.

    The workbook layout is therefore preserved even if Gemini returns different
    raster dimensions.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        source_language: str,
        target_language: str,
        http_referer: str = "",
        app_name: str = "PDF Translator Studio",
        timeout: int = 180,
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing.")
        if not model:
            raise RuntimeError("An OpenRouter image model is required for Excel images.")

        self.model = model
        self.source_language = source_language
        self.target_language = target_language
        self.timeout = timeout

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if http_referer:
            self.headers["HTTP-Referer"] = http_referer
        if app_name:
            self.headers["X-Title"] = app_name

    def _prompt(self) -> str:
        return f"""
Edit this EXACT spreadsheet image.

Source language: {self.source_language}
Target language: {self.target_language}

Translate all visible {self.source_language} text to accurate {self.target_language}.

STRICT PRESERVATION:
- This is an image embedded inside an Excel workbook.
- Preserve the exact composition and geometry.
- Do not create new tables.
- Do not add, remove, reorder, or redesign rows/columns or visual elements.
- Preserve borders, colors, icons, diagrams, logos, screenshots, arrows and graphics.
- Replace TEXT ONLY.
- Keep translated text in the same visual area as the source text.
- Preserve alignment and hierarchy.
- Do not crop.
- Do not add whitespace around the original image.
- Do not return an explanation; return the edited image.

The output will be inserted back into Excel using the ORIGINAL image anchor and
the ORIGINAL displayed width and height.
""".strip()

    def translate_image_bytes(self, image_bytes: bytes, filename: str) -> bytes:
        url = "https://openrouter.ai/api/v1/images"
        payload = {
            "model": self.model,
            "prompt": self._prompt(),
            "input_references": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_url(image_bytes, _media_type(filename))
                    },
                }
            ],
        }

        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter image generation failed ({response.status_code}) "
                f"for {self.model}: {response.text[:4000]}"
            )

        data = response.json()
        items = data.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise RuntimeError("OpenRouter image response did not contain b64_json image data.")

        raw = base64.b64decode(items[0]["b64_json"])
        # Validate that the returned bytes are an actual image.
        Image.open(io.BytesIO(raw)).verify()
        return raw

    def replace_sheet_images(self, workbook):
        results: list[ImageTranslationResult] = []

        for ws in workbook.worksheets:
            originals = list(getattr(ws, "_images", []))
            if not originals:
                continue

            new_images = []

            for index, original in enumerate(originals):
                try:
                    source_bytes = original._data()
                    filename = getattr(original, "path", None) or f"{ws.title}_{index}.png"
                    generated = self.translate_image_bytes(source_bytes, str(filename))

                    from openpyxl.drawing.image import Image as XLImage
                    new_image = XLImage(io.BytesIO(generated))

                    # Preserve displayed dimensions and exact anchor.
                    new_image.width = original.width
                    new_image.height = original.height
                    new_image.anchor = copy.copy(original.anchor)

                    new_images.append(new_image)

                    try:
                        original._data()
                    except Exception:
                        pass

                    results.append(
                        ImageTranslationResult(
                            sheet=ws.title,
                            index=index,
                            model=self.model,
                            original_size=(int(original.width), int(original.height)),
                            generated_size=Image.open(io.BytesIO(generated)).size,
                        )
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed translating image {index + 1} on sheet '{ws.title}': {exc}"
                    ) from exc

            # Replace only after every image on the sheet has succeeded.
            ws._images = new_images

        return results
