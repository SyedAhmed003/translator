from src.models import TextLine, TextSpan
from src.segmenter import segment_lines


def line(text, y, x0=50, x1=500, size=10):
    span = TextSpan(
        text=text,
        bbox=(x0, y, x1, y + 10),
        font="TestFont",
        size=size,
        color=0,
        flags=0,
    )
    return TextLine(
        text=text,
        bbox=(x0, y, x1, y + 10),
        spans=[span],
    )


def test_segments_wrapped_japanese_paragraph():
    lines = [
        line("これは第一行です。", 100),
        line("これは第二行です。", 118),
        line("別の見出し", 160, x0=80, x1=180),
    ]

    units = segment_lines(lines, 0)

    assert len(units) >= 2
    assert "第一行" in units[0].text
