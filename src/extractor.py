
import re
import pymupdf as fitz
from .models import TextLine, TextSpan

CJK_RE = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]')


def contains_japanese(text):
    return bool(CJK_RE.search(text))


def _span_from_dict(span):
    return TextSpan(
        text=span.get('text', ''),
        bbox=tuple(span.get('bbox', (0, 0, 0, 0))),
        font=span.get('font', ''),
        size=float(span.get('size', 10)),
        color=int(span.get('color', 0)),
        flags=int(span.get('flags', 0)),
    )


def _join_fragments(parts):
    out = ''
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not out:
            out = part
            continue
        if contains_japanese(out) or contains_japanese(part):
            out += part
        else:
            out += ' ' + part
    return out.strip()


def extract_lines_from_dict(data):
    raw = []
    for block in data.get('blocks', []):
        if block.get('type') != 0:
            continue
        for line in block.get('lines', []):
            spans = [_span_from_dict(s) for s in line.get('spans', [])]
            if not spans:
                continue
            text = _join_fragments([s.text for s in spans])
            if not text:
                continue
            raw.append(
                TextLine(
                    text=text,
                    bbox=tuple(line.get('bbox', (0, 0, 0, 0))),
                    spans=spans,
                )
            )

    raw.sort(key=lambda l: (round(l.bbox[1], 1), l.bbox[0]))
    merged = []

    for line in raw:
        cy = (line.bbox[1] + line.bbox[3]) / 2
        placed = False
        for existing in reversed(merged[-8:]):
            ey = (existing.bbox[1] + existing.bbox[3]) / 2
            if (
                abs(cy - ey) <= 1.2
                and abs(
                    (line.bbox[3] - line.bbox[1])
                    - (existing.bbox[3] - existing.bbox[1])
                ) <= 1.5
            ):
                existing.spans.extend(line.spans)
                existing.bbox = (
                    min(existing.bbox[0], line.bbox[0]),
                    min(existing.bbox[1], line.bbox[1]),
                    max(existing.bbox[2], line.bbox[2]),
                    max(existing.bbox[3], line.bbox[3]),
                )
                existing.spans.sort(key=lambda s: s.bbox[0])
                existing.text = _join_fragments([s.text for s in existing.spans])
                placed = True
                break
        if not placed:
            merged.append(line)

    merged.sort(key=lambda l: (l.bbox[1], l.bbox[0]))
    return merged


def extract_lines(page):
    return extract_lines_from_dict(page.get_text('dict'))
