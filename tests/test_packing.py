from __future__ import annotations

import numpy as np

from context_span_select.packing import chunk_spans, covered, pack
from context_span_select.select import build_selector


def test_chunk_spans_cover_text():
    text = "abcdefghijklmnopqrstuvwxyz"
    spans = chunk_spans(text, 10)
    assert spans[0] == (0, 10)
    assert "".join(text[a:b] for a, b in spans) == text


def test_pack_respects_budget():
    lengths = np.asarray([100, 100, 100, 100], dtype=int)
    picked, used = pack([0, 1, 2, 3], lengths, window=0, budget=250)
    assert used <= 250
    assert len(picked) == 2


def test_covered():
    assert covered({1, 2, 3}, [{1, 2}, {3}])
    assert not covered({1, 2}, [{1, 2}, {3}])


def test_overlap_selector_ranks_relevant_first():
    selector = build_selector("overlap_stub")
    chunks = [
        "unrelated weather report",
        "Bank of Japan adjusted parameters for JGB yields",
        "sports scores from yesterday",
    ]
    order = selector.rank("Which bank adjusted JGB yields?", chunks)
    assert int(order[0]) == 1
