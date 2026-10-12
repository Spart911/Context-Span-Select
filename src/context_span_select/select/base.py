from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from context_span_select.packing import pack
from context_span_select.schemas import PackResult


class Selector(ABC):
    name: str = "base"

    @abstractmethod
    def rank(self, question: str, chunks: list[str]) -> np.ndarray:
        """Return chunk ids in descending relevance order."""

    def select(
        self,
        example_id: str,
        question: str,
        chunks: list[str],
        budget_chars: int,
        window: int = 0,
    ) -> PackResult:
        order = self.rank(question, chunks)
        lengths = np.asarray([len(c) for c in chunks], dtype=np.int64)
        picked, used = pack(order, lengths, window, budget_chars)
        return PackResult(
            example_id=example_id,
            chunk_ids=sorted(picked),
            chars_used=int(used),
            budget_chars=budget_chars,
            method=self.name,
        )
