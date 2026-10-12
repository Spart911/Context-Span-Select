from __future__ import annotations

from context_span_select.packing import covered


def supporting_fact_cover(picked: set[int], gold_groups: list[set[int]]) -> float:
    """1.0 if every gold group ⊆ picked, else 0.0 (pack-level cover)."""
    return 1.0 if covered(picked, gold_groups) else 0.0


def supporting_fact_recall(picked: set[int], gold_ids: set[int]) -> float:
    """Token/chunk recall of flat gold ids."""
    if not gold_ids:
        return 1.0
    return len(picked & gold_ids) / len(gold_ids)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
