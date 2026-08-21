from __future__ import annotations

import pymupdf as fitz


def _intersection_area(a: fitz.Rect, b: fitz.Rect) -> float:
    r = a & b
    return max(0.0, r.get_area())


def validate_page(page: fitz.Page, expected_regions: list[dict], table_grids=None) -> dict:
    """Deterministic post-render QA. It never changes the PDF; it reports defects."""
    text = page.get_text("dict")
    rendered = []
    for block in text.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                s = span.get("text", "").strip()
                if not s:
                    continue
                rendered.append({"text": s, "bbox": fitz.Rect(span["bbox"])})

    collisions = []
    for i in range(len(rendered)):
        for j in range(i + 1, len(rendered)):
            a, b = rendered[i], rendered[j]
            area = _intersection_area(a["bbox"], b["bbox"])
            if area > 0.4:
                min_area = min(max(1, a["bbox"].get_area()), max(1, b["bbox"].get_area()))
                if area / min_area > 0.12:
                    collisions.append({"type": "text-text", "a": a["text"], "b": b["text"], "area": area})

    border_hits = []
    for d in page.get_drawings():
        r = d.get("rect")
        if not r or (r.width < 20 and r.height < 20):
            continue
        for item in rendered:
            b = item["bbox"]
            # Horizontal rules directly under/above a text line are normally
            # intentional decoration. A collision is only reported when the
            # rule crosses the vertical body of the glyph box. Vertical rules
            # are suspicious when they cross the horizontal body of the text.
            if r.height <= 2.0 and r.width >= 20:
                y = (r.y0 + r.y1) / 2
                crosses_body = b.y0 + 0.18 * b.height < y < b.y1 - 0.18 * b.height
                x_overlap = max(0.0, min(r.x1, b.x1) - max(r.x0, b.x0))
                if crosses_body and x_overlap > 0.35 * min(r.width, b.width):
                    border_hits.append({"type": "horizontal-line-crossing-text", "text": item["text"], "drawing": [r.x0, r.y0, r.x1, r.y1]})
            elif r.width <= 2.0 and r.height >= 10:
                x = (r.x0 + r.x1) / 2
                crosses_body = b.x0 + 0.12 * b.width < x < b.x1 - 0.12 * b.width
                y_overlap = max(0.0, min(r.y1, b.y1) - max(r.y0, b.y0))
                if crosses_body and y_overlap > 0.35 * min(r.height, b.height):
                    border_hits.append({"type": "vertical-line-crossing-text", "text": item["text"], "drawing": [r.x0, r.y0, r.x1, r.y1]})

    outside = []
    for region in expected_regions:
        if not region.get("safe"):
            continue
        safe = fitz.Rect(region["safe"])
        needle = region.get("text", "")
        for item in rendered:
            if needle and needle[:20] not in item["text"] and item["text"] not in needle:
                continue
            if not safe.contains(item["bbox"]):
                outside.append({"id": region.get("id"), "text": item["text"], "safe": list(safe), "bbox": list(item["bbox"])})

    return {
        "text_count": len(rendered),
        "text_text_collisions": collisions,
        "text_drawing_collisions": border_hits,
        "outside_safe_regions": outside,
        "pass": not collisions and not border_hits and not outside,
    }
