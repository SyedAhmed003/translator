from pathlib import Path
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]


def choose_font(bold: bool = False) -> str:
    candidates = []

    if bold:
        candidates.extend(
            [
                ROOT / "fonts" / "NotoSans-Bold.ttf",
                Path("C:/Windows/Fonts/arialbd.ttf"),
                Path("C:/Windows/Fonts/segoeuib.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                ROOT / "fonts" / "NotoSans-Regular.ttf",
                Path("C:/Windows/Fonts/arial.ttf"),
                Path("C:/Windows/Fonts/segoeui.ttf"),
            ]
        )

    for path in candidates:
        if path.exists():
            return str(path)

    # Built-in PDF font fallback. Good for Latin English text.
    return "helv"


def is_bold(flags: int) -> bool:
    # PDF font flags: bit 4 is commonly used for bold.
    return bool(flags & 16)
