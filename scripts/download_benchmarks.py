#!/usr/bin/env python3
"""List / download public benchmarks into data/raw/<id>/.

Requires optional extra: pip install -e ".[hf]"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "benchmarks.yaml"
RAW = ROOT / "data" / "raw"

DOWNLOADERS = {
    "hotpotqa_distractor": {
        "hub": "hotpot_qa",
        "config": "distractor",
        "split": "validation",
    },
    "squad_v2": {
        "hub": "rajpurkar/squad_v2",
        "config": None,
        "split": "validation",
    },
    "qasper": {
        "hub": "allenai/qasper",
        "config": None,
        "split": "validation",
    },
    "longbench_qa": {
        "hub": "THUDM/LongBench",
        "config": None,
        "split": "test",
    },
}


def load_registry() -> list[dict]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return data["benchmarks"]


def cmd_list() -> int:
    for row in load_registry():
        print(
            f"{row['id']:24s} tier={row['tier']} status={row['status']:8s} "
            f"levels={','.join(row.get('levels', []))}"
        )
    print("\nDownloadable now:", ", ".join(sorted(DOWNLOADERS)))
    return 0


def cmd_download(only: str | None, limit: int | None) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print('Install HF extra: pip install -e ".[hf]"', file=sys.stderr)
        return 2

    ids = [only] if only else sorted(DOWNLOADERS)
    for bid in ids:
        if bid not in DOWNLOADERS:
            print(f"skip {bid}: no downloader yet (see docs/BENCHMARKS.md)", file=sys.stderr)
            continue
        spec = DOWNLOADERS[bid]
        out_dir = RAW / bid
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {bid} → {out_dir}", flush=True)
        if spec["config"]:
            ds = load_dataset(spec["hub"], spec["config"], split=spec["split"])
        else:
            ds = load_dataset(spec["hub"], split=spec["split"])
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        path = out_dir / f"{spec['split']}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in ds:
                handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
        meta = {
            "benchmark_id": bid,
            "hub": spec["hub"],
            "config": spec["config"],
            "split": spec["split"],
            "n": len(ds),
            "path": str(path.relative_to(ROOT)),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {path} n={len(ds)}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list registry and exit")
    parser.add_argument("--only", type=str, default=None, help="download one benchmark id")
    parser.add_argument("--limit", type=int, default=None, help="cap rows for smoke downloads")
    args = parser.parse_args()
    if args.list or (args.only is None and args.limit is None):
        raise SystemExit(cmd_list())
    raise SystemExit(cmd_download(args.only, args.limit))


if __name__ == "__main__":
    main()
