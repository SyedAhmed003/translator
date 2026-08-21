
import json
from pathlib import Path
import pymupdf as fitz

from .fonts import choose_font, is_bold
from .validator import validate_page
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


def render_image_text_regions(page, regions, min_font_scale=0.90):
    rendered = []
    warnings = []
    for region in regions:
        rect = fitz.Rect(region.page_bbox)
        if rect.is_empty or rect.width < 2 or rect.height < 2:
            continue

        # Keep the original embedded image object untouched. The white patch
        # only masks the old glyphs visually; image dimensions/placement are
        # never changed. This is appropriate for scanned documents and image
        # snippets with a paper/white background.
        page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)

        fontfile = choose_font(False)
        size, tight = _image_text_font_size(region, region.translation, fontfile, min_font_scale)
        fontname, embedded = _font_spec(fontfile)
        kwargs = dict(
            fontname=fontname,
            fontsize=size,
            align=0,
            color=(0, 0, 0),
            lineheight=0.95,
        )
        if embedded:
            kwargs["fontfile"] = embedded

        result = page.insert_textbox(
            rect,
            region.translation,
            **kwargs,
        )
        warning = None
        if result < 0:
            # Retry with a smaller size while keeping the exact region.
            retry_size = size
            while result < 0 and retry_size > 4.5:
                retry_size -= 0.2
                kwargs["fontsize"] = retry_size
                result = page.insert_textbox(rect, region.translation, **kwargs)
            size = retry_size
        if result < 0 or tight:
            warning = "Image text was fitted at the minimum local font size."
            warnings.append({
                "page": region.page_index + 1,
                "unit_id": region.region_id,
                "warning": warning,
            })

        rendered.append({
            "id": region.region_id,
            "source": region.source_text,
            "translation": region.translation,
            "bbox": list(rect),
            "font_size": size,
            "confidence": region.confidence,
            "warning": warning,
        })
    return rendered, warnings


def render_translated_pdf(
    input_path,
    output_path,
    units,
    translations,
    all_page_lines=None,
    image_regions=None,
    min_font_scale=0.90,
    font_scale_step=0.02,
    text_margin=0.0,
):
    doc = fitz.open(input_path)
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

    page_indexes = sorted(set(by_page) | set(image_by_page))
    for page_index in page_indexes:
        page_units = by_page.get(page_index, [])
        page = doc[page_index]
        all_lines = (all_page_lines or {}).get(page_index, [])
        grids = detect_table_grids(page)

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
                slots, size, overflow = _fit_unit_lines(
                    u, translation, fontfile, min_font_scale
                )
                if overflow:
                    # Never paint outside the original region. Instead use a
                    # conservative local fit and flag it for review.
                    warning = (
                        "English is longer than the original region at the minimum "
                        "font scale; content was kept inside the original region."
                    )
                    # Use a single compact textbox inside the original region.
                    # This avoids overlaps with the next paragraph.
                    rect = fitz.Rect(
                        u.bbox[0], u.bbox[1], u.bbox[2], u.bbox[3]
                    )
                    fontname, embedded = _font_spec(fontfile)
                    kwargs = dict(
                        fontname=fontname,
                        fontsize=size,
                        align=0,
                        color=(0, 0, 0),
                        lineheight=0.95,
                    )
                    if embedded:
                        kwargs["fontfile"] = embedded
                    page.insert_textbox(rect, translation, **kwargs)
                else:
                    align = 1 if u.kind == "title" else 0
                    for i, txt in enumerate(slots):
                        if txt:
                            _insert_line(page, u.lines[i], txt, size, fontfile, align)

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
