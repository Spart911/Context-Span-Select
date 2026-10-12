from context_span_select.select.base import Selector
from context_span_select.select.hybrid_stub import OverlapSelector

SELECTORS = {
    "overlap_stub": OverlapSelector,
    "hybrid_stub": OverlapSelector,
}


def build_selector(name: str) -> Selector:
    if name not in SELECTORS:
        raise KeyError(f"unknown selector {name!r}; known: {sorted(SELECTORS)}")
    return SELECTORS[name]()
