from __future__ import annotations

import json
from pathlib import Path

import pymupdf as fitz

from config import settings
from .cache import TranslationCache
from .hybrid_extractor import extract_page_hybrid
from .pdf_analyzer import analyze_pdf
from .renderer import render_translated_pdf, detect_table_grids
from .segmenter import segment_lines
from .translator import OpenRouterTranslator
from .image_translator import OpenRouterVisionTranslator, ImageTextRegion


def _mode(value: str) -> str:
    return {
        "Auto (MuPDF → OCR when needed)": "auto",
        "Native MuPDF only": "native",
        "OCR": "ocr",
    }.get(value, "auto")


def _extract(input_path, source_language, extraction_mode, ocr_dpi, vision_translator=None):
    doc = fitz.open(input_path)
    all_units = []
    all_page_lines = {}
    all_image_regions: list[ImageTextRegion] = []
    modes = []
    analysis = analyze_pdf(input_path)
    for page_index, page in enumerate(doc):
        # If a page is image-dominant, prefer OpenRouter vision OCR over a
        # local OCR engine. This handles full-page scans and text embedded in
        # imported artwork with the same model selected in the UI.
        page_text_chars = len(page.get_text("text").strip())
        page_area = max(1.0, page.rect.width * page.rect.height)
        image_area = 0.0
        for img in page.get_images(full=True):
            for rect in page.get_image_rects(img[0]):
                image_area += max(0.0, rect.width * rect.height)
        image_ratio = min(1.0, image_area / page_area)
        image_dominant = vision_translator is not None and image_ratio >= 0.72 and page_text_chars < 80

        if image_dominant:
            from .hybrid_extractor import ExtractionResult
            extraction = ExtractionResult(
                lines=[], mode="openrouter-vision", ocr_used=True, confidence=0.0,
                note=f"Image-dominant page ({image_ratio:.0%}); OpenRouter vision OCR used."
            )
        else:
            extraction = extract_page_hybrid(page, source_language, mode=_mode(extraction_mode), ocr_dpi=ocr_dpi)
        lines = extraction.lines
        all_page_lines[page_index] = lines
        grids = detect_table_grids(page)
        units = segment_lines(lines, page_index, source_language, grids, page.rect.width)
        if vision_translator is not None:
            page_image_regions = vision_translator.extract_page_regions(page, page_index)
            all_image_regions.extend(page_image_regions)
        all_units.extend(units)
        modes.append({
            "page": page_index + 1,
            "mode": extraction.mode,
            "ocr_used": extraction.ocr_used,
            "confidence": extraction.confidence,
            "note": extraction.note,
            "units": len(units),
            "image_text_regions": sum(1 for r in all_image_regions if r.page_index == page_index),
            "image_ratio": round(image_ratio, 4),
        })
    doc.close()
    return analysis, all_units, all_page_lines, modes, all_image_regions


def inspect_document(input_path: str, report_path: str | None = None, source_language: str = "Japanese"):
    analysis, units, _, modes, image_regions = _extract(input_path, source_language, "Auto (MuPDF → OCR when needed)", 300, None)
    result = {
        "analysis": analysis,
        "extraction": modes,
        "image_text_regions": [
            {"id": r.region_id, "page": r.page_index + 1, "source": r.source_text, "translation": r.translation, "bbox": list(r.page_bbox)}
            for r in image_regions
        ],
        "units": [
            {"id": u.unit_id, "page": u.page_index + 1, "kind": u.kind, "text": u.text, "bbox": list(u.bbox)}
            for u in units
        ],
    }
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def translate_document_with_options(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str,
    source_language: str,
    target_language: str,
    min_font_scale: float = 0.90,
    text_margin: float = 0.0,
    use_cache: bool = True,
    extraction_mode: str = "Auto (MuPDF → OCR when needed)",
    ocr_dpi: int = 300,
    use_vision_images: bool = True,
    max_image_edge: int = 2400,
):
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Put it in the project's .env file.")
    if not model:
        raise RuntimeError("No OpenRouter model was selected in the UI.")
    if not source_language or not target_language:
        raise RuntimeError("Source and target languages must be selected in the UI.")
    if source_language == target_language:
        raise RuntimeError("Source and target languages must be different.")

    vision = None
    if use_vision_images:
        vision = OpenRouterVisionTranslator(
            api_key=api_key,
            model=model,
            source_language=source_language,
            target_language=target_language,
            max_image_edge=max_image_edge,
            http_referer=settings.openrouter_http_referer,
            app_name=settings.openrouter_app_name,
        )
    analysis, units, all_page_lines, extraction, image_regions = _extract(
        input_path, source_language, extraction_mode, ocr_dpi, vision
    )
    if not units and not image_regions:
        raise RuntimeError("No translatable source-language text or image text was found in the PDF.")

    cache = TranslationCache(settings.cache_db)
    try:
        translator = OpenRouterTranslator(
            api_key=api_key,
            model=model,
            source_language=source_language,
            target_language=target_language,
            cache=cache,
            http_referer=settings.openrouter_http_referer,
            app_name=settings.openrouter_app_name,
        )
        translations = translator.translate_batch(units, use_cache=use_cache)
        report, report_path = render_translated_pdf(
            input_path=input_path,
            output_path=output_path,
            units=units,
            translations=translations,
            all_page_lines=all_page_lines,
            image_regions=image_regions,
            min_font_scale=min_font_scale,
            font_scale_step=settings.font_scale_step,
            text_margin=text_margin,
        )
        return {
            "output": output_path,
            "report": report_path,
            "unit_count": len(units),
            "image_text_region_count": len(image_regions),
            "warning_count": len(report.get("warnings", [])),
            "validation_pass": report.get("validation_pass", False),
            "extraction": extraction,
            "analysis": analysis,
            "model": model,
            "source_language": source_language,
            "target_language": target_language,
        }
    finally:
        cache.close()


def translate_document(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str,
    source_language: str = "Japanese",
    target_language: str = "English",
    use_cache: bool = True,
):
    return translate_document_with_options(
        input_path=input_path,
        output_path=output_path,
        api_key=api_key,
        model=model,
        source_language=source_language,
        target_language=target_language,
        use_cache=use_cache,
    )
