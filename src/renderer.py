from __future__ import annotations

import json
from pathlib import Path
import io
import math
import gc
from typing import TYPE_CHECKING
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

from .fonts import choose_font, is_bold
from .validator import validate_page
if TYPE_CHECKING:
    from .image_translator import ImageTextRegion


def _font_spec(fontfile):
    if fontfile and fontfile != "helv" and Path(fontfile).exists():
        return "helv", str(fontfile)
    return "helv", None


def _font_obj(fontfile):
    if fontfile and fontfile != "helv" and Path(fontfile).exists():
        return fitz.Font(fontfile=str(fontfile))
    return fitz.Font(fontname="helv")


def measure(text, fontfile, size):
    return _font_obj(fontfile).text_length(text, fontsize=size)


def _metrics(fontfile, size):
    f = _font_obj(fontfile)
    return f.ascender * size, f.descender * size


def _insert_text(page, point, text, size, fontfile, color=(0, 0, 0)):
    fontname, embedded_font = _font_spec(fontfile)
    kwargs = dict(fontname=fontname, fontsize=size, color=color)
    if embedded_font:
        kwargs["fontfile"] = embedded_font
    return page.insert_text(point, text, **kwargs)


def _cluster(values, tol=1.5):
    out = []
    for v in sorted(values):
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
        else:
            out[-1] = (out[-1] + v) / 2
    return out


def detect_table_grids(page):
    horizontal, vertical = [], []
    for d in page.get_drawings():
        r = d.get("rect")
        if not r:
            continue
        if r.width >= 30 and r.height <= 1.5:
            horizontal.append((r.y0, r.x0, r.x1))
        elif r.height >= 10 and r.width <= 1.5:
            vertical.append((r.x0, r.y0, r.y1))

    grids = []
    ys = _cluster([y for y, _, _ in horizontal], 1.5)
    for i in range(len(ys) - 1):
        top, bottom = ys[i], ys[i + 1]
        if bottom - top < 12:
            continue
        vx = _cluster(
            [x for x, y0, y1 in vertical if y0 <= top + 2 and y1 >= bottom - 2],
            1.5,
        )
        if len(vx) >= 3:
            grids.append((top, bottom, vx))
    return grids


def _line_overlap(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def _is_table_border(rect, grids):
    for top, bottom, xs in grids:
        if abs(rect.y0 - top) < 1.5 or abs(rect.y0 - bottom) < 1.5:
            if _line_overlap((rect.x0, 0, rect.x1), (min(xs), 0, max(xs))) > 30:
                return True
        for x in xs:
            if abs(rect.x0 - x) < 1.5 and rect.height > 10:
                return True
    return False


def _decoration_lines_to_remove(page, translated_lines, grids):
    """
    Find thin horizontal rules that run through/under translated text.
    Table borders are explicitly protected.
    """
    remove = []
    for d in page.get_drawings():
        r = d.get("rect")
        if not r or r.height > 1.5 or r.width < 25:
            continue
        if _is_table_border(r, grids):
            continue

        for line in translated_lines:
            lr = fitz.Rect(line.bbox)
            overlap = _line_overlap((r.x0, 0, r.x1), (lr.x0, 0, lr.x1))
            if overlap < 0.55 * min(r.width, lr.width):
                continue

            # If the rule lies within ~1.5 pt of the text box, it is an
            # underline/decoration associated with that text, not page artwork.
            if r.y0 >= lr.y0 - 1.5 and r.y0 <= lr.y1 + 1.5:
                remove.append(r)
                break
    return remove


def _wrap(text, width, fontfile, size):
    words = text.replace("\n", " ").split()
    if not words:
        return []
    lines, cur = [], ""
    for word in words:
        candidate = word if not cur else cur + " " + word
        if measure(candidate, fontfile, size) <= width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
            if measure(cur, fontfile, size) > width:
                # Break long identifiers character-wise.
                piece = ""
                for ch in cur:
                    if measure(piece + ch, fontfile, size) <= width:
                        piece += ch
                    else:
                        if piece:
                            lines.append(piece)
                        piece = ch
                cur = piece
    if cur:
        lines.append(cur)
    return lines


def _fit_line_text(text, line, fontfile, start_size, min_scale):
    width = max(8.0, line.bbox[2] - line.bbox[0])
    size = start_size
    minimum = max(5.2, start_size * min_scale)
    while size >= minimum - 1e-6:
        if measure(text, fontfile, size) <= width:
            return size, False
        size -= 0.15
    return minimum, measure(text, fontfile, minimum) > width


def _baseline_for_line(line, fontfile, size):
    # Use a conservative baseline anchored to the original top edge.
    # PDF font metrics are not always identical to the bbox metrics returned
    # by the source font, especially for Latin descenders. A slightly higher
    # baseline prevents English glyphs from touching/crossing source underlines.
    baseline = line.bbox[1] + size * 0.82
    return baseline


def _insert_line(page, line, text, size, fontfile, align=0):
    width = measure(text, fontfile, size)
    x0, _, x1, _ = line.bbox
    if align == 1:
        x = (x0 + x1 - width) / 2
    elif align == 2:
        x = x1 - width
    else:
        x = x0
    x = max(x0, min(x, x1 - width))
    baseline = _baseline_for_line(line, fontfile, size)
    return _insert_text(page, (x, baseline), text, size, fontfile)


def _fit_unit_lines(unit, translation, fontfile, min_scale):
    widths = [max(8, l.bbox[2] - l.bbox[0]) for l in unit.lines]
    size = unit.size
    minimum = max(5.2, unit.size * min_scale)

    while size >= minimum - 1e-6:
        words = translation.replace("\n", " ").split()
        slots, idx, cur = [], 0, ""
        ok = True
        for word in words:
            if idx >= len(widths):
                ok = False
                break
            candidate = word if not cur else cur + " " + word
            if measure(candidate, fontfile, size) <= widths[idx]:
                cur = candidate
            else:
                if cur:
                    slots.append(cur)
                    idx += 1
                if idx >= len(widths) or measure(word, fontfile, size) > widths[idx]:
                    ok = False
                    break
                cur = word
        if cur:
            slots.append(cur)
        if ok and len(slots) <= len(widths):
            while len(slots) < len(widths):
                slots.append("")
            return slots, size, False
        size -= 0.15

    # At minimum scale, use the exact original line count and wrap as tightly
    # as possible. Do NOT create extra lines outside the original region.
    size = minimum
    slots = [""] * len(widths)
    idx = 0
    for word in translation.replace("\n", " ").split():
        if idx >= len(widths):
            return slots, size, True
        candidate = word if not slots[idx] else slots[idx] + " " + word
        if measure(candidate, fontfile, size) <= widths[idx]:
            slots[idx] = candidate
        else:
            idx += 1
            if idx >= len(widths) or measure(word, fontfile, size) > widths[idx]:
                return slots, size, True
            slots[idx] = word
    return slots, size, False


def _insert_cell(page, rect, text, fontfile, original_size, min_scale):
    pad_x = 6
    pad_y = 2.5
    inner = fitz.Rect(rect[0] + pad_x, rect[1] + pad_y, rect[2] - pad_x, rect[3] - pad_y)
    size = original_size
    minimum = max(5.2, original_size * min_scale)

    while size >= minimum - 1e-6:
        lines = _wrap(text, inner.width, fontfile, size)
        asc, desc = _metrics(fontfile, size)
        total_h = len(lines) * (asc - desc) * 1.05
        if lines and total_h <= inner.height:
            # Center vertically, but keep the full glyph box inside the cell.
            first_baseline = inner.y0 + (inner.height - total_h) / 2 + asc
            for i, line_text in enumerate(lines):
                _insert_text(page, (inner.x0, first_baseline + i * (asc - desc) * 1.05),
                             line_text, size, fontfile)
            return size, False
        size -= 0.15

    # Final safe fallback: single line, clipped to cell width via font scaling.
    size = minimum
    if measure(text, fontfile, size) > inner.width:
        while size > 4.8 and measure(text, fontfile, size) > inner.width:
            size -= 0.1
    asc, desc = _metrics(fontfile, size)
    baseline = inner.y0 + (inner.height - (asc - desc)) / 2 + asc
    _insert_text(page, (inner.x0, baseline), text, size, fontfile)
    return size, True


def _image_text_font_size(region: ImageTextRegion, translation: str, fontfile: str, min_scale: float) -> tuple[float, bool]:
    x0, y0, x1, y1 = region.page_bbox
    rect = fitz.Rect(x0, y0, x1, y1)
    original_h = max(3.0, rect.height)
    size = max(5.2, original_h * 0.78)
    minimum = max(4.8, size * min_scale)
    width = max(6.0, rect.width - 2.0)
    while size >= minimum - 1e-6:
        lines = _wrap(translation, width, fontfile, size)
        asc, desc = _metrics(fontfile, size)
        line_h = max(1.0, asc - desc) * 1.02
        if lines and len(lines) <= max(1, int(rect.height / max(1.0, line_h))):
            if len(lines) * line_h <= max(1.0, rect.height + 0.5):
                return size, False
        size -= 0.15
    return max(4.8, minimum), True


def _image_font_path():
    candidates = [
        Path(__file__).resolve().parents[1] / "fonts" / "NotoSans-Regular.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _wrap_pixels(draw, text, font, width):
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return []
    if " " not in text:
        words = list(text)
    else:
        words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + ("" if " " not in text else " ") + word
        box = draw.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        # Break a single over-wide token/character sequence.
        piece = ""
        for ch in word:
            candidate2 = piece + ch
            box2 = draw.textbbox((0, 0), candidate2, font=font)
            if box2[2] - box2[0] <= width:
                piece = candidate2
            else:
                if piece:
                    lines.append(piece)
                piece = ch
        current = piece
    if current:
        lines.append(current)
    return lines


def _fit_image_text(draw, text, bbox, font_path, min_scale=0.90):
    x0, y0, x1, y1 = bbox
    width = max(8, x1 - x0 - 4)
    height = max(8, y1 - y0 - 4)
    start = max(8, int(height * 0.78))
    minimum = max(7, int(start * min_scale))
    chosen = None
    for size in range(start, minimum - 1, -1):
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        lines = _wrap_pixels(draw, text, font, width)
        if not lines:
            continue
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        line_h = max(1, max(b[3] - b[1] for b in boxes))
        spacing = max(1, int(line_h * 0.08))
        total_h = len(lines) * line_h + (len(lines) - 1) * spacing
        if total_h <= height:
            chosen = (font, lines, line_h, spacing, False)
            break
    if chosen is not None:
        return chosen
    size = minimum
    font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    lines = _wrap_pixels(draw, text, font, width)[:max(1, int(height / max(1, size)))]
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_h = max(1, max((b[3] - b[1] for b in boxes), default=size))
    spacing = max(1, int(line_h * 0.05))
    return font, lines, line_h, spacing, True


def _inpaint_image(image: Image.Image, regions):
    """Remove only the source-text pixels; keep all other image content intact."""
    rgba = image.convert("RGBA")
    rgb = np.array(rgba.convert("RGB")) if np is not None else None
    mask = None
    if cv2 is not None and np is not None:
        mask = np.zeros((rgba.height, rgba.width), dtype=np.uint8)
        for region in regions:
            x0, y0, x1, y1 = [int(round(v)) for v in region.bbox_px]
            pad = max(1, int(round(min(rgba.width, rgba.height) * 0.0025)))
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(rgba.width - 1, x1 + pad); y1 = min(rgba.height - 1, y1 + pad)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        if np.any(mask):
            restored = cv2.inpaint(rgb, mask, 3.0, cv2.INPAINT_TELEA)
            out = Image.fromarray(restored).convert("RGBA")
            out.putalpha(rgba.getchannel("A"))
            return out

    # Dependency-free fallback for simple document images: use a local median
    # border colour instead of a hard white rectangle.
    out = rgba.copy()
    px = out.load()
    for region in regions:
        x0, y0, x1, y1 = [int(round(v)) for v in region.bbox_px]
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(out.width, x1); y1 = min(out.height, y1)
        samples = []
        for x in range(x0, min(x1, x0 + max(1, (x1-x0)//8))):
            if y0 > 0: samples.append(px[x, y0-1][:3])
            if y1 < out.height: samples.append(px[x, y1][:3])
        for y in range(y0, min(y1, y0 + max(1, (y1-y0)//8))):
            if x0 > 0: samples.append(px[x0-1, y][:3])
            if x1 < out.width: samples.append(px[x1, y][:3])
        if not samples:
            continue
        bg = tuple(sorted(c[i] for c in samples)[len(samples)//2] for i in range(3))
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                alpha = px[xx, yy][3]
                px[xx, yy] = (*bg, alpha)
    return out


def _render_image_occurrence(doc, page, xref, rect, regions):
    extracted = doc.extract_image(xref)
    if not extracted or not extracted.get("image"):
        return False
    try:
        image = Image.open(io.BytesIO(extracted["image"]))
    except Exception:
        return False
    source_image = image.convert("RGBA")
    image = _inpaint_image(source_image, regions)
    draw = ImageDraw.Draw(image)
    font_path = _image_font_path()
    for region in sorted(regions, key=lambda r: (r.bbox_px[1], r.bbox_px[0])):
        bbox = region.bbox_px
        font, lines, line_h, spacing, _ = _fit_image_text(draw, region.translation, bbox, font_path, 0.88)
        x0, y0, x1, y1 = bbox
        pad = max(1, int(round(min(image.width, image.height) * 0.004)))
        x = max(0, int(round(x0)) + pad)
        y = max(0, int(round(y0)) + pad)
        for line in lines:
            draw.text((x, y), line, font=font, fill=(0, 0, 0, 255))
            y += line_h + spacing
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    page.insert_image(rect, stream=buf.getvalue(), overlay=True)
    return True


def render_image_text_regions(page, regions, min_font_scale=0.60):
    if not regions:
        return [], []
    rendered = []
    warnings = []
    grouped = {}
    for region in regions:
        grouped.setdefault((region.xref, region.occurrence_index), []).append(region)

    doc = page.parent
    for (xref, occurrence_index), group in grouped.items():
        rects = page.get_image_rects(xref)
        if occurrence_index >= len(rects):
            continue
        rect = rects[occurrence_index]
        ok = _render_image_occurrence(doc, page, xref, rect, group)
        if not ok:
            warnings.append({
                "page": page.number + 1,
                "unit_id": f"p{page.number + 1}_img{xref}_{occurrence_index}",
                "warning": "Image text was detected but the image could not be rebuilt in-place.",
            })
            continue
        for region in group:
            rendered.append({
                "id": region.region_id,
                "source": region.source_text,
                "translation": region.translation,
                "bbox": list(region.page_bbox),
                "font_size": None,
                "confidence": region.confidence,
                "warning": None,
            })
    return rendered, warnings


def render_translated_pdf(
    input_path,
    output_path,
    units,
    translations,
    all_page_lines=None,
    image_regions=None,
    min_font_scale=0.60,
    font_scale_step=0.02,
    text_margin=0.0,
    full_page_images=None,
):
    doc = fitz.open(input_path)
    full_page_images = full_page_images or {}
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "strategy": "openrouter-multimodal-hybrid-preserve-image-geometry-local-text-fitting",
        "pages": [],
        "warnings": [],
    }

    by_page = {}
    for u in units:
        by_page.setdefault(u.page_index, []).append(u)

    image_regions = image_regions or []
    image_by_page = {}
    for region in image_regions:
        image_by_page.setdefault(region.page_index, []).append(region)

    page_indexes = sorted(set(by_page) | set(image_by_page) | set(full_page_images))
    for page_index in page_indexes:
        page_units = by_page.get(page_index, [])
        page = doc[page_index]
        all_lines = (all_page_lines or {}).get(page_index, [])
        grids = detect_table_grids(page)

        # Full-page Gemini image path. A scanned page is rebuilt as one complete
        # generated image; do not run the old per-image OCR/text overlay path.
        if page_index in full_page_images:
            fp = full_page_images[page_index]
            png = fp.get("png") if isinstance(fp, dict) else None
            if not png:
                raise RuntimeError(f"Full-page image result for page {page_index + 1} contains no PNG data.")

            # Remove the original page visuals while preserving the page rectangle.
            page.apply_redactions()
            rect = page.rect
            page.insert_image(rect, stream=png, keep_proportion=False, overlay=True)

            page_report = {
                "page": page_index + 1,
                "units": [],
                "image_text_regions": [],
                "full_page_image": {
                    "model": fp.get("model") if isinstance(fp, dict) else None,
                    "source_size": fp.get("source_size") if isinstance(fp, dict) else None,
                    "output_size": fp.get("output_size") if isinstance(fp, dict) else None,
                    "passes": fp.get("passes") if isinstance(fp, dict) else None,
                },
                "validation": {"pass": True, "skipped": True},
            }
            report["pages"].append(page_report)
            continue

        # GLOBAL TEXT CLEANUP:
        # Remove the complete original text layer once. This prevents the
        # "struck-through/ghost text" problem caused by overlapping redaction
        # rectangles from partial source-only cleanup.
        if all_lines:
            for line in all_lines:
                r = fitz.Rect(line.bbox)
                r.x0 -= max(0.3, text_margin)
                r.x1 += max(0.3, text_margin)
                r.y0 -= 0.4
                r.y1 += 0.4
                page.add_redact_annot(r, fill=None)
            page.apply_redactions(images=0, graphics=0, text=0)

        # Remove only decorative rules that would cross translated glyphs.
        # Table borders are protected.
        source_lines = [ln for u in page_units for ln in u.lines]
        decoration_rects = _decoration_lines_to_remove(page, source_lines, grids)
        for r in decoration_rects:
            page.add_redact_annot(r, fill=None)
        if decoration_rects:
            page.apply_redactions(images=0, graphics=0, text=0)

        # Mark regions covered by translated units. All remaining original
        # text is reinserted unchanged, so dates/phone numbers/Latin metadata
        # never disappear when the page is globally cleaned.
        covered = []
        for u in page_units:
            for l in u.lines:
                covered.append(fitz.Rect(l.bbox))

        def covered_by_translation(line):
            r = fitz.Rect(line.bbox)
            area = max(1, r.width * r.height)
            return any((r & c).get_area() / area > 0.55 for c in covered)

        # Reinsert untouched text that was not part of a translation unit.
        # This is what makes global cleanup safe.
        if all_lines:
            for line in all_lines:
                if covered_by_translation(line):
                    continue
                txt = line.text.strip()
                if not txt:
                    continue
                fontfile = choose_font(is_bold(max(line.spans, key=lambda s: len(s.text), default=None).flags if line.spans else 0))
                size = max(5.2, max((s.size for s in line.spans), default=9.5))
                # Preserve original line position using target font metrics.
                _insert_line(page, line, txt, size, fontfile, align=0)

        page_report = {
            "page": page_index + 1,
            "units": [],
            "components": {
                "images": len(page.get_images(full=True)),
                "drawings": len(page.get_drawings()),
                "table_grids": len(grids),
                "native_or_ocr_lines": len(all_lines),
                "image_text_regions": len(image_by_page.get(page_index, [])),
            },
        }

        for u in page_units:
            translation = translations[u.unit_id].strip()
            fontfile = choose_font(is_bold(u.flags))
            warning = None

            if u.kind == "table_cell":
                size, overflow = _insert_cell(
                    page,
                    u.bbox,
                    translation,
                    fontfile,
                    u.size,
                    min_font_scale,
                )
                if overflow:
                    warning = "Cell required minimum font-size fallback."
            else:
                # Translate the whole logical block into the original union
                # rectangle. The older line-by-line fitting used every source
                # line's Japanese width as an independent English column, which
                # produced uneven alignment and premature font shrinking.
                rect = fitz.Rect(u.bbox)
                max_lines = max(1, len(u.lines))
                align = 1 if u.kind == "title" else 0
                size = u.size
                minimum = max(5.2, u.size * min_font_scale)
                chosen_lines = None
                while size >= minimum - 1e-6:
                    lines = _wrap(translation, max(8.0, rect.width), fontfile, size)
                    asc, desc = _metrics(fontfile, size)
                    line_h = max(1.0, asc - desc) * 1.02
                    if lines and len(lines) <= max_lines and len(lines) * line_h <= rect.height + 0.5:
                        chosen_lines = lines
                        break
                    size -= 0.15
                if chosen_lines is None:
                    size = minimum
                    chosen_lines = _wrap(translation, max(8.0, rect.width), fontfile, size)
                    warning = (
                        "Translation exceeded the original text box at the minimum "
                        "font scale; it was constrained to the original region."
                    )
                    if len(chosen_lines) > max_lines:
                        chosen_lines = chosen_lines[:max_lines]
                asc, desc = _metrics(fontfile, size)
                line_h = max(1.0, asc - desc) * 1.02
                first_baseline = rect.y0 + asc
                for i, txt in enumerate(chosen_lines):
                    if not txt:
                        continue
                    width = measure(txt, fontfile, size)
                    if align == 1:
                        x = rect.x0 + max(0.0, (rect.width - width) / 2.0)
                    else:
                        x = rect.x0
                    _insert_text(page, (x, first_baseline + i * line_h), txt, size, fontfile)

            page_report["units"].append({
                "id": u.unit_id,
                "kind": u.kind,
                "source": u.text,
                "translation": translation,
                "bbox": list(u.bbox),
                "original_line_count": len(u.lines),
                "final_font_size": size,
                "warning": warning,
            })
            if warning:
                report["warnings"].append({
                    "page": page_index + 1,
                    "unit_id": u.unit_id,
                    "warning": warning,
                })

        image_rendered, image_warnings = render_image_text_regions(
            page, image_by_page.get(page_index, []), min_font_scale=min_font_scale
        )
        page_report["image_text_regions"] = image_rendered
        for warning in image_warnings:
            report["warnings"].append(warning)

        expected = []
        for item in page_report["units"]:
            if item.get("warning"):
                # Regions with warnings are still reported, but are not used as
                # hard validation gates because a minimum-scale fallback may be valid.
                continue
            r = fitz.Rect(item["bbox"])
            safe = fitz.Rect(r.x0 - 0.5, r.y0 - 0.5, r.x1 + 0.5, r.y1 + 0.5)
            expected.append({"id": item["id"], "text": item["translation"], "safe": list(safe)})
        qa = validate_page(page, expected, grids)
        page_report["validation"] = qa
        if not qa["pass"]:
            report["warnings"].append({
                "page": page_index + 1,
                "unit_id": "PAGE_QA",
                "warning": "Post-render collision/geometry validation found potential layout defects.",
                "qa": qa,
            })
        report["pages"].append(page_report)

    report["validation_pass"] = all(p.get("validation", {}).get("pass", False) for p in report["pages"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    report_path = str(Path(output_path).with_suffix(".report.json"))
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, report_path
