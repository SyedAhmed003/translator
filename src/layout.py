import pymupdf as fitz


def expand_rect(rect, margin: float, page_rect: fitz.Rect) -> fitz.Rect:
    r = fitz.Rect(rect)
    r.x0 = max(page_rect.x0, r.x0 - margin)
    r.y0 = max(page_rect.y0, r.y0 - margin)
    r.x1 = min(page_rect.x1, r.x1 + margin)
    r.y1 = min(page_rect.y1, r.y1 + margin)
    return r


def try_fit_text(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    fontname: str,
    start_size: float,
    min_scale: float,
    step: float,
    align: int = 0,
):
    size = start_size
    min_size = max(4.0, start_size * min_scale)

    while size >= min_size - 1e-6:
        # Shape/fit test using a temporary textbox call is not reversible
        # in all PyMuPDF versions, so use insert_textbox on a copy-like
        # measurement approach through Shape.
        shape = page.new_shape()
        spare = shape.insert_textbox(
            rect,
            text,
            fontname=fontname,
            fontsize=size,
            align=align,
        )
        # Do not commit the test shape.
        # spare >= 0 means the text fits in the rectangle.
        if spare >= 0:
            return size, spare

        size -= step

    return None, None
