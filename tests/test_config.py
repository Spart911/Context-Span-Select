from __future__ import annotations

from context_span_select.config import default_config, load_config
from context_span_select.paths import CONFIGS


def test_default_config_loads():
    cfg = default_config()
    assert cfg["project"] == "context-span-select"
    assert 4000 in cfg["pack"]["budgets_chars"]


def test_extends_merges():
    cfg = load_config(CONFIGS / "benchmarks" / "hotpotqa_distractor.yaml")
    assert cfg["benchmark_id"] == "hotpotqa_distractor"
    assert cfg["project"] == "context-span-select"
    assert cfg["eval"]["levels"] == ["A", "C"]
