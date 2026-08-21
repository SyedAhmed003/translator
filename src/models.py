from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextSpan:
    text: str
    bbox: tuple[float, float, float, float]
    font: str
    size: float
    color: int
    flags: int


@dataclass
class TextLine:
    text: str
    bbox: tuple[float, float, float, float]
    spans: list[TextSpan] = field(default_factory=list)


@dataclass
class TextUnit:
    unit_id: str
    page_index: int
    text: str
    bbox: tuple[float, float, float, float]
    lines: list[TextLine]
    font: str
    size: float
    color: int
    flags: int
    align: str = "left"
    kind: str = "paragraph"
    translation: Optional[str] = None
    needs_review: bool = False
    warning: Optional[str] = None
