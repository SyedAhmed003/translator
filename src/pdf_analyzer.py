import pymupdf as fitz


def analyze_pdf(path: str) -> dict:
    doc = fitz.open(path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append(
            {
                "page": i + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "has_extractable_text": bool(text),
                "text_chars": len(text),
                "image_count": len(page.get_images(full=True)),
                "drawing_count": len(page.get_drawings()),
            }
        )

    result = {
        "file": str(path),
        "page_count": len(doc),
        "pages": pages,
        "recommended_pipeline": "native-text" if any(
            p["has_extractable_text"] for p in pages
        ) else "ocr-required",
    }

    doc.close()
    return result
