from __future__ import annotations

import argparse
import json
from pathlib import Path

from context_span_select import __version__
from context_span_select.config import default_config, load_config
from context_span_select.io import read_json
from context_span_select.packing import chunk_spans
from context_span_select.paths import MANIFESTS, ROOT
from context_span_select.select import build_selector


def cmd_info(_: argparse.Namespace) -> int:
    print(f"context-span-select {__version__}")
    print(f"root: {ROOT}")
    return 0


def cmd_benchmarks(_: argparse.Namespace) -> int:
    path = MANIFESTS / "benchmarks.yaml"
    text = path.read_text(encoding="utf-8")
    print(text)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else default_config()
    selector = build_selector(cfg["selector"]["name"])
    doc = (
        "The Bank of Japan adjusted operational parameters. "
        "This helped moderate JGB yield volatility in 2013. "
        "Other central banks took different steps."
    )
    question = "Which central bank moderated JGB yield volatility?"
    size = int(cfg["pack"]["chunk_chars"])
    spans = chunk_spans(doc, size)
    chunks = [doc[a:b] for a, b in spans]
    budget = int(cfg["pack"]["budgets_chars"][0])
    result = selector.select("demo", question, chunks, budget_chars=budget, window=0)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.path)
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="css", description="Context-Span-Select CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="show version and paths")
    p_info.set_defaults(func=cmd_info)

    p_bench = sub.add_parser("benchmarks", help="print benchmark registry")
    p_bench.set_defaults(func=cmd_benchmarks)

    p_demo = sub.add_parser("demo", help="run overlap stub on a tiny example")
    p_demo.add_argument("--config", type=Path, default=None)
    p_demo.set_defaults(func=cmd_demo)

    p_cfg = sub.add_parser("config", help="dump resolved YAML config as JSON")
    p_cfg.add_argument("path", type=Path)
    p_cfg.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
