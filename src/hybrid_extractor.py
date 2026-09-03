from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz

from .extractor import extract_lines
from .models import TextLine, TextSpan


@dataclass
class ExtractionResult:
    lines: list[TextLine]
    mode: str
    ocr_used: bool
    confidence: float
    note: str = ""


def _native_quality(lines: list[TextLine], page: fitz.Page) -> dict[str, float]:
    chars = sum(len(x.text.strip()) for x in lines)
    nonempty = sum(1 for x in lines if x.text.strip())
    page_area = max(1.0, page.rect.width * page.rect.height)

    occupied = sum(
        max(0.0, x.bbox[2] - x.bbox[0])
        * max(0.0, x.bbox[3] - x.bbox[1])
        for x in lines
    )
    area_ratio = min(1.0, occupied / page_area)

    unique = len({x.text.strip() for x in lines if x.text.strip()})
    duplicate_ratio = 1.0 - (unique / max(1, nonempty))

    score = 0.0
    if chars >= 30:
        score += 0.55
    if nonempty >= 3:
        score += 0.20
    if area_ratio >= 0.0003:
        score += 0.15
    if duplicate_ratio < 0.25:
        score += 0.10

    image_area = 0.0

    for img in page.get_images(full=True):
        try:
            rects = page.get_image_rects(img[0])
            image_area += sum(
                max(0.0, r.width * r.height)
                for r in rects
            )
        except Exception:
            pass

    image_ratio = min(1.0, image_area / page_area)

    return {
        "chars": float(chars),
        "lines": float(nonempty),
        "area_ratio": area_ratio,
        "duplicate_ratio": duplicate_ratio,
        "image_ratio": image_ratio,
        "score": score,
    }


def _source_key(source_language: str) -> str:
    s = (source_language or "").lower().strip()

    if "simplified chinese" in s:
        return "ch"
    if "traditional chinese" in s:
        return "chinese"
    if "chinese" in s:
        return "chinese"
    if "japanese" in s:
        return "japan"
    if "korean" in s:
        return "korean"
    if "arabic" in s:
        return "arabic"
    if "hindi" in s:
        return "hindi"
    if "tamil" in s:
        return "tamil"
    if "telugu" in s:
        return "telugu"
    if "malayalam" in s:
        return "malayalam"
    if "kannada" in s:
        return "kannada"
    if "thai" in s:
        return "thai"

    return "latin"


def _paddle_available() -> bool:
    try:
        import paddlex  # noqa: F401
        return True
    except Exception:
        return False


def _configure_tesseract() -> str | None:
    """
    Locate and configure the Tesseract executable for pytesseract.

    Priority:
      1. TESSERACT_CMD from .env/environment
      2. tesseract found on PATH
      3. Standard Windows installation paths
    """
    candidates = [
        os.getenv("TESSERACT_CMD", "").strip(),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    try:
        import pytesseract
    except Exception:
        return next(
            (
                str(Path(candidate))
                for candidate in candidates
                if candidate and Path(candidate).is_file()
            ),
            None,
        )

    for candidate in candidates:
        if not candidate:
            continue

        path = Path(candidate)

        if path.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            return str(path)

    return None


def _tesseract_available() -> bool:
    return _configure_tesseract() is not None


def _render_page_image(
    page: fitz.Page,
    dpi: int,
    temp_dir: Path | None = None,
) -> tuple[Any, float, float]:
    """Render page with MuPDF. Poppler is optional."""
    scale = dpi / 72.0
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
    )

    try:
        from PIL import Image
        import io

        image = Image.open(
            io.BytesIO(pix.tobytes("png"))
        ).convert("RGB")

        return image, scale, scale
    finally:
        pix = None


def _parse_paddlex_result(
    result: Any,
    scale_x: float,
    scale_y: float,
) -> list[TextLine]:
    """Normalize PaddleX OCR output into the project's TextLine model."""
    data = getattr(result, "json", None)

    if callable(data):
        data = data()

    if data is None:
        data = getattr(result, "res", None)

    if data is None and isinstance(result, dict):
        data = result

    if data is None:
        return []

    def find(obj):
        if isinstance(obj, dict):
            if "rec_texts" in obj and (
                "rec_boxes" in obj or "rec_polys" in obj
            ):
                return obj

            for value in obj.values():
                found = find(value)
                if found is not None:
                    return found

        elif isinstance(obj, list):
            for value in obj:
                found = find(value)
                if found is not None:
                    return found

        return None

    payload = find(data)

    if payload is None:
        return []

    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or [1.0] * len(texts)

    boxes = payload.get("rec_boxes")
    if boxes is None:
        boxes = payload.get("rec_polys") or []

    lines: list[TextLine] = []

    for text, score, box in zip(texts, scores, boxes):
        text = str(text).strip()

        if not text:
            continue

        try:
            points = box.tolist() if hasattr(box, "tolist") else box

            if (
                len(points) == 4
                and isinstance(points[0], (list, tuple))
            ):
                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]
            else:
                x0, y0, x1, y1 = map(float, points[:4])
                xs, ys = [x0, x1], [y0, y1]

            x0, x1 = min(xs) / scale_x, max(xs) / scale_x
            y0, y1 = min(ys) / scale_y, max(ys) / scale_y

        except Exception:
            continue

        size = max(5.0, (y1 - y0) * 0.78)

        span = TextSpan(
            text=text,
            bbox=(x0, y0, x1, y1),
            font="ocr",
            size=size,
            color=0,
            flags=0,
        )

        lines.append(
            TextLine(
                text=text,
                bbox=(x0, y0, x1, y1),
                spans=[span],
            )
        )

    lines.sort(
        key=lambda line: (
            round(
                line.bbox[1]
                / max(
                    3.0,
                    line.bbox[3] - line.bbox[1],
                )
            ),
            line.bbox[0],
        )
    )

    return lines


def _ocr_with_paddle(
    page: fitz.Page,
    source_language: str,
    dpi: int,
) -> ExtractionResult:
    try:
        from paddlex import create_pipeline
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR/PaddleX is not installed. "
            "Run: pip install -r requirements-ocr.txt"
        ) from exc

    image, sx, sy = _render_page_image(page, dpi)

    pipeline = create_pipeline(
        pipeline="OCR",
        device="cpu",
    )

    outputs = pipeline.predict(
        input=image,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    result = next(iter(outputs), None)

    if result is None:
        raise RuntimeError(
            "PaddleOCR returned no result for the page."
        )

    lines = _parse_paddlex_result(result, sx, sy)

    if not lines:
        raise RuntimeError(
            "PaddleOCR completed but returned no text regions."
        )

    return ExtractionResult(
        lines,
        "paddleocr",
        True,
        0.85,
        f"PaddleOCR OCR, dpi={dpi}",
    )


def _ocr_with_tesseract(
    page: fitz.Page,
    source_language: str,
    dpi: int,
) -> ExtractionResult:
    try:
        import pytesseract
    except Exception as exc:
        raise RuntimeError(
            "pytesseract is not installed in the active Python environment."
        ) from exc

    cmd = _configure_tesseract()

    if not cmd:
        raise RuntimeError(
            "Tesseract executable could not be located. "
            "Set TESSERACT_CMD or install Tesseract OCR."
        )

    try:
        languages = pytesseract.get_languages(config="")
    except Exception as exc:
        raise RuntimeError(
            "Tesseract executable was found but could not be queried: "
            f"{exc}"
        ) from exc

    lang_map = {
        "japan": "jpn+eng",
        "ch": "chi_sim+eng",
        "chinese": "chi_tra+eng",
        "korean": "kor+eng",
        "arabic": "ara+eng",
        "hindi": "hin+eng",
        "tamil": "tam+eng",
        "telugu": "tel+eng",
        "malayalam": "mal+eng",
        "kannada": "kan+eng",
        "thai": "tha+eng",
        "latin": "eng",
    }

    lang = lang_map.get(
        _source_key(source_language),
        "eng",
    )

    requested_languages = {
        item.strip()
        for item in lang.split("+")
        if item.strip()
    }

    missing_languages = sorted(
        requested_languages - set(languages)
    )

    if missing_languages:
        raise RuntimeError(
            "Tesseract is installed, but the required language "
            f"data is missing: {', '.join(missing_languages)}. "
            "Install the corresponding .traineddata files in "
            r"C:\Program Files\Tesseract-OCR\tessdata\."
        )

    image, sx, sy = _render_page_image(page, dpi)

    data = pytesseract.image_to_data(
        image,
        lang=lang,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )

    lines: list[TextLine] = []

    for i, text in enumerate(data.get("text", [])):
        text = str(text).strip()

        if not text:
            continue

        raw_conf = str(data["conf"][i]).strip()

        conf = (
            float(raw_conf)
            if raw_conf not in {"", "-1"}
            else 0.0
        )

        if conf < 20:
            continue

        x = float(data["left"][i]) / sx
        y = float(data["top"][i]) / sy
        w = float(data["width"][i]) / sx
        h = float(data["height"][i]) / sy

        size = max(5.0, h * 0.78)

        span = TextSpan(
            text=text,
            bbox=(x, y, x + w, y + h),
            font="ocr",
            size=size,
            color=0,
            flags=0,
        )

        lines.append(
            TextLine(
                text=text,
                bbox=(x, y, x + w, y + h),
                spans=[span],
            )
        )

    if not lines:
        raise RuntimeError(
            f"Tesseract returned no usable text using language '{lang}'."
        )

    lines.sort(
        key=lambda line: (
            round(
                line.bbox[1]
                / max(
                    3.0,
                    line.bbox[3] - line.bbox[1],
                )
            ),
            line.bbox[0],
        )
    )

    valid_confidences = [
        float(data["conf"][i])
        for i in range(len(data.get("text", [])))
        if str(data["conf"][i]).strip() not in {"", "-1"}
    ]

    avg = sum(valid_confidences)
    count = max(1, len(valid_confidences))

    confidence = max(
        0.0,
        min(1.0, avg / count / 100.0),
    )

    return ExtractionResult(
        lines,
        "tesseract",
        True,
        confidence,
        f"Tesseract OCR, language={lang}, dpi={dpi}",
    )


def extract_page_hybrid(
    page: fitz.Page,
    source_language: str,
    mode: str = "auto",
    ocr_dpi: int = 300,
) -> ExtractionResult:
    native = extract_lines(page)
    q = _native_quality(native, page)

    if mode == "native":
        if not native:
            raise RuntimeError(
                "Native MuPDF extraction found no text on this page."
            )

        return ExtractionResult(
            native,
            "native",
            False,
            q["score"],
            "Native MuPDF extraction",
        )

    suspicious = (
        q["chars"] < 20
        or (
            q["chars"] < 80
            and q["image_ratio"] > 0.45
            and q["score"] < 0.75
        )
    )

    if mode != "ocr" and not suspicious:
        return ExtractionResult(
            native,
            "native",
            False,
            q["score"],
            "Native MuPDF extraction",
        )

    errors: list[str] = []

    if _paddle_available():
        try:
            return _ocr_with_paddle(
                page,
                source_language,
                ocr_dpi,
            )
        except Exception as exc:
            errors.append(f"PaddleOCR: {exc}")

    if _tesseract_available():
        try:
            return _ocr_with_tesseract(
                page,
                source_language,
                ocr_dpi,
            )
        except Exception as exc:
            errors.append(f"Tesseract: {exc}")

    if native:
        return ExtractionResult(
            native,
            "native-fallback",
            False,
            q["score"],
            "OCR unavailable; native extraction retained. "
            + " | ".join(errors),
        )

    raise RuntimeError(
        "This page is scanned/image-only and no OCR backend "
        "completed successfully. "
        + (
            "Details: " + " | ".join(errors)
            if errors
            else "No OCR backend was detected."
        )
    )
