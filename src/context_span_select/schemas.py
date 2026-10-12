from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Span:
    """Char span in a document (inclusive start, exclusive end)."""

    start: int
    end: int
    text: str = ""
    chunk_id: int | None = None


@dataclass(slots=True)
class Example:
    """Normalized unit for selection experiments."""

    id: str
    question: str
    doc_id: str
    document: str
    answer: str | None = None
    gold_spans: list[Span] = field(default_factory=list)
    gold_chunk_ids: list[int] = field(default_factory=list)
    unanswerable: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PackResult:
    example_id: str
    chunk_ids: list[int]
    chars_used: int
    budget_chars: int
    scores: dict[int, float] = field(default_factory=dict)
    method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
