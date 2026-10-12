from __future__ import annotations

import numpy as np


def chunk_spans(text: str, size: int) -> list[tuple[int, int]]:
    """Non-overlapping char chunks of roughly `size` characters."""
    if size <= 0:
        raise ValueError("size must be positive")
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + size)
        spans.append((i, j))
        i = j
    return spans


def pack(
    order: list[int] | np.ndarray,
    lengths: np.ndarray,
    window: int,
    budget: int,
) -> tuple[set[int], int]:
    """Greedy pack under char budget with optional ±window neighbors."""
    picked: set[int] = set()
    used = 0
    n = len(lengths)
    smallest = int(lengths.min()) if n else 0
    for seed in order:
        seed = int(seed)
        block = [seed] + [
            j
            for d in range(1, window + 1)
            for j in (seed - d, seed + d)
            if 0 <= j < n
        ]
        for j in block:
            if j in picked:
                continue
            size = int(lengths[j])
            if used + size > budget:
                if j == seed:
                    break
                continue
            picked.add(j)
            used += size
        if used + smallest > budget:
            break
    return picked, used


def covered(picked: set[int], groups: list[set[int]]) -> bool:
    """True if every gold group is fully contained in `picked`."""
    return all(group <= picked for group in groups)
