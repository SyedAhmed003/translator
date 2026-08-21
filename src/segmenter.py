from .extractor import contains_japanese
import re
from .models import TextLine, TextUnit


def _union(boxes):
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _style(line):
    if not line.spans:
        return "", 10.0, 0, 0
    s = max(line.spans, key=lambda x: len(x.text))
    return s.font, s.size, s.color, s.flags


def _width(line):
    return line.bbox[2] - line.bbox[0]


def _center(line):
    return (line.bbox[0] + line.bbox[2]) / 2


def _is_title_line(line):
    return _width(line) > 150 and 160 < _center(line) < 435


def _is_body_line(line):
    return _width(line) >= 300


def _should_group(prev, cur):
    _, ps, _, _ = _style(prev)
    _, cs, _, _ = _style(cur)
    if abs(ps - cs) > 0.8:
        return False

    gap = cur.bbox[1] - prev.bbox[3]

    # Main title is two centered lines with a larger gap.
    if prev.bbox[1] < 285 and cur.bbox[1] < 285 and _is_title_line(prev) and _is_title_line(cur) and gap <= 30:
        return True

    # Normal document line spacing is about 18 points.
    if gap < 3 or gap > 12:
        return False

    # Company/contact metadata must remain individual rows.
    if prev.bbox[1] < 215 and cur.bbox[1] < 215:
        return False

    # Paragraph lines: keep a complete paragraph together for translation,
    # but do not merge unrelated headings/labels.
    if _is_body_line(prev) and _is_body_line(cur):
        return abs(prev.bbox[0] - cur.bbox[0]) <= 25

    return False


def _kind(lines, page_width=595):
    widths = [_width(x) for x in lines]
    centers = [_center(x) for x in lines]
    avg_center = sum(centers) / len(centers)
    max_width = max(widths)
    text_len = len(" ".join(x.text for x in lines))

    # Geometry-based classification rather than fixed page coordinates.
    if len(lines) <= 2 and max_width > page_width * 0.25 and abs(avg_center - page_width / 2) < page_width * 0.22:
        return "title"
    if len(lines) == 1 and max_width < page_width * 0.45 and lines[0].bbox[0] < page_width * 0.25:
        return "heading"
    if len(lines) >= 3 and max_width >= page_width * 0.45:
        return "paragraph"
    if text_len < 90:
        return "label"
    return "paragraph"


def _is_source_text(text: str, source_language: str) -> bool:
    if not text.strip():
        return False
    lang = (source_language or "").lower()
    if "japanese" in lang:
        return contains_japanese(text)
    if "chinese" in lang:
        return bool(re.search(r"[\u3400-\u9fff]", text))
    if "korean" in lang:
        return bool(re.search(r"[\uac00-\ud7af]", text))
    if "arabic" in lang:
        return bool(re.search(r"[\u0600-\u06ff]", text))
    if "russian" in lang or "ukrainian" in lang:
        return bool(re.search(r"[\u0400-\u04ff]", text))
    if "thai" in lang:
        return bool(re.search(r"[\u0e00-\u0e7f]", text))
    if "hindi" in lang or "devanagari" in lang:
        return bool(re.search(r"[\u0900-\u097f]", text))
    if "tamil" in lang:
        return bool(re.search(r"[\u0b80-\u0bff]", text))
    if "telugu" in lang:
        return bool(re.search(r"[\u0c00-\u0c7f]", text))
    if "malayalam" in lang:
        return bool(re.search(r"[\u0d00-\u0d7f]", text))
    if "kannada" in lang:
        return bool(re.search(r"[\u0c80-\u0cff]", text))
    # Latin-script languages.
    return bool(re.search(r"[A-Za-zÀ-ÿ]", text))


def _cell_for_line(line, table_grids):
    if not table_grids:
        return None
    cx = _center(line)
    cy = (line.bbox[1] + line.bbox[3]) / 2
    for top, bottom, xs in table_grids:
        if top - 2 <= cy <= bottom + 2:
            for i in range(len(xs) - 1):
                if xs[i] - 1 <= cx <= xs[i + 1] + 1:
                    return (top, bottom, xs[i], xs[i + 1], i)
    return None


def segment_lines(lines: list[TextLine], page_index: int, source_language: str = "Japanese", table_grids=None, page_width=595) -> list[TextUnit]:
    # Only source-language-bearing lines are translation targets. Non-text
    # PDF objects (images/vector drawings) are never touched by this stage.
    lines = [x for x in lines if _is_source_text(x.text, source_language)]
    lines.sort(key=lambda x: (x.bbox[1], x.bbox[0]))

    units = []

    # Generic table handling: if the PDF contains a vector table grid, create
    # one translation unit per occupied cell/row. This avoids any dependency on
    # a particular table wording or page coordinate.
    table_lines = {}
    normal_lines = []
    for line in lines:
        cy = (line.bbox[1] + line.bbox[3]) / 2
        matching_grid = None
        if table_grids:
            for grid in table_grids:
                top, bottom, xs = grid
                if top - 2 <= cy <= bottom + 2:
                    matching_grid = grid
                    break
        if matching_grid:
            top, bottom, xs = matching_grid
            # A PDF text line may span several table cells because the PDF
            # extractor merged adjacent text runs. Re-split it by span X
            # positions before creating translation units.
            cell_spans = {}
            for span in line.spans:
                scx = (span.bbox[0] + span.bbox[2]) / 2
                col = None
                for ci in range(len(xs) - 1):
                    if xs[ci] - 1 <= scx <= xs[ci + 1] + 1:
                        col = ci
                        break
                if col is not None:
                    cell_spans.setdefault(col, []).append(span)
            # First split spans that cross a cell border. Some PDFs store a
            # sequence like "額 １株につき..." as one span even though the
            # vector border falls between the characters. A proportional split
            # by the span's X geometry keeps the text on the correct side.
            split_spans = []
            for span in line.spans:
                pieces = [(span, span.text, span.bbox[0], span.bbox[2])]
                for border in xs[1:-1]:
                    next_pieces = []
                    for sp, txt, sx0, sx1 in pieces:
                        if sx0 < border < sx1 and len(txt) > 1:
                            ratio = (border - sx0) / max(0.001, sx1 - sx0)
                            cut = max(1, min(len(txt) - 1, round(len(txt) * ratio)))
                            # Prefer a whitespace boundary when the source
                            # contains a clear label/value separator.
                            ws = [i for i, ch in enumerate(txt) if ch.isspace() and 0 < i < len(txt) - 1]
                            if ws:
                                cut = min(ws, key=lambda i: abs(i - cut))
                            left_txt, right_txt = txt[:cut], txt[cut:]
                            left_x1 = sx0 + (sx1 - sx0) * (len(left_txt) / len(txt))
                            right_x0 = left_x1
                            from .models import TextSpan
                            left = TextSpan(left_txt, (sx0, span.bbox[1], left_x1, span.bbox[3]), span.font, span.size, span.color, span.flags)
                            right = TextSpan(right_txt, (right_x0, span.bbox[1], sx1, span.bbox[3]), span.font, span.size, span.color, span.flags)
                            next_pieces.extend([(left, left_txt, sx0, left_x1), (right, right_txt, right_x0, sx1)])
                        else:
                            next_pieces.append((sp, txt, sx0, sx1))
                    pieces = next_pieces
                split_spans.extend([x[0] for x in pieces])

            for col in range(len(xs) - 1):
                x0, x1 = xs[col], xs[col + 1]
                spans = [sp for sp in split_spans if (sp.bbox[0] + sp.bbox[2]) / 2 >= x0 - 1 and (sp.bbox[0] + sp.bbox[2]) / 2 <= x1 + 1]
                if not spans:
                    continue
                row_key = round(line.bbox[1] / 2) * 2
                cell_lines_key = (top, bottom, x0, x1, col, row_key)
                cell_line = TextLine(
                    text="".join(sp.text for sp in sorted(spans, key=lambda z: z.bbox[0])).strip(),
                    bbox=(min(sp.bbox[0] for sp in spans), min(sp.bbox[1] for sp in spans),
                          max(sp.bbox[2] for sp in spans), max(sp.bbox[3] for sp in spans)),
                    spans=spans,
                )
                table_lines.setdefault(cell_lines_key, []).append(cell_line)
        else:
            normal_lines.append(line)

    for i, (key, group) in enumerate(sorted(table_lines.items(), key=lambda kv: (kv[0][5], kv[0][2])), 1):
        spans = [sp for line in group for sp in line.spans]
        dominant = max(spans, key=lambda s: len(s.text), default=None)
        top, bottom, x0, x1, col, row_key = key
        units.append(TextUnit(
            unit_id=f"p{page_index + 1}_t{i:03d}", page_index=page_index,
            text=" ".join(line.text for line in sorted(group, key=lambda x: x.bbox[0])).strip(),
            bbox=(x0, top, x1, bottom),
            lines=group, font=dominant.font if dominant else "",
            size=dominant.size if dominant else 10.0, color=dominant.color if dominant else 0,
            flags=dominant.flags if dominant else 0, kind="table_cell",
        ))

    groups = []
    current = []
    for line in normal_lines:
        if not current or _should_group(current[-1], line):
            current.append(line)
        else:
            groups.append(current)
            current = [line]
    if current:
        groups.append(current)

    start_index = len(units)
    for i, group in enumerate(groups, start_index + 1):
        spans = [s for line in group for s in line.spans]
        dominant = max(spans, key=lambda s: len(s.text), default=None)
        units.append(TextUnit(
            unit_id=f"p{page_index + 1}_u{i:03d}",
            page_index=page_index,
            text="\n".join(line.text for line in group).strip(),
            bbox=_union([line.bbox for line in group]),
            lines=group,
            font=dominant.font if dominant else "",
            size=dominant.size if dominant else 10.0,
            color=dominant.color if dominant else 0,
            flags=dominant.flags if dominant else 0,
            kind=_kind(group, page_width),
        ))
    return units
