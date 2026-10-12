from __future__ import annotations

import re

import numpy as np

from context_span_select.select.base import Selector


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class OverlapSelector(Selector):
    """Cheap lexical overlap stub until BM25+embedding hybrid is wired."""

    name = "overlap_stub"

    def rank(self, question: str, chunks: list[str]) -> np.ndarray:
        q = _tokens(question)
        if not chunks:
            return np.asarray([], dtype=int)
        scores = np.asarray(
            [len(q & _tokens(c)) / max(1, len(q)) for c in chunks],
            dtype=np.float64,
        )
        return np.argsort(-scores, kind="stable")
