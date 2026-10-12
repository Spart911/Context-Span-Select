#!/usr/bin/env python3
"""Run Kev attention-head evidence selection experiment with strict logging.

Must be launched from the kev uv environment (mlx + kev installed), e.g.:

  cd ~/.cache/kev
  PYTHONPATH="/path/to/Context-Span-Select/src" \\
    uv run --extra serve python /path/to/Context-Span-Select/scripts/run_kev_attn_experiment.py \\
      --config experiments/20261012_kev_attn_qrheads/config.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_span_select.experiment import ExperimentRun, ExperimentSpec
from context_span_select.kev import (
    ATTN_LAYERS,
    N_HEADS,
    N_TOTAL,
    branch_attention,
    build_labeled_state,
    chunk_scores_from_attn,
    head_qr_scores,
    install_attention_capture,
    layer_module_ids,
    load_kev4b_mlx8,
    state_token_chunk_ids,
)
from context_span_select.kev.attention import flat_to_layer_head
from context_span_select.packing import covered, pack
from context_span_select.paths import RESULTS


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_packs(cfg: dict) -> Path:
    packs = Path(cfg["data"]["packs"])
    if not packs.is_absolute():
        packs = ROOT / packs
    if not packs.exists():
        raise FileNotFoundError(f"packs not found: {packs}")
    return packs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/20261012_kev_attn_qrheads/config.yaml",
    )
    ap.add_argument("--smoke", type=int, default=0, help="override detect+eval to N each")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    desc = cfg["experiment"]
    if args.smoke:
        cfg.setdefault("run", {})["detect_n"] = args.smoke
        cfg.setdefault("run", {})["eval_n"] = args.smoke

    spec = ExperimentSpec(
        experiment_id=desc["id"],
        title=desc["title"],
        description=desc["description"],
        hypothesis=desc["hypothesis"],
        method=desc["method"],
        dataset=desc["dataset"],
        metrics=list(desc["metrics"]),
        budgets=list(cfg.get("budgets", [])),
        tags=list(desc.get("tags", [])),
        notes=desc.get("notes", ""),
    )
    run = ExperimentRun(RESULTS / "runs", spec, config=cfg)

    try:
        packs_path = resolve_packs(cfg)
        run.event("packs_resolved", {"path": str(packs_path)})
        packs = json.loads(packs_path.read_text(encoding="utf-8"))
        detect_n = int(cfg.get("run", {}).get("detect_n") or len(packs["detect"]))
        eval_n = int(cfg.get("run", {}).get("eval_n") or len(packs["eval"]))
        detect_rows_in = packs["detect"][:detect_n]
        eval_rows_in = packs["eval"][:eval_n]
        run.event(
            "packs_loaded",
            {
                "detect_available": len(packs["detect"]),
                "eval_available": len(packs["eval"]),
                "detect_used": len(detect_rows_in),
                "eval_used": len(eval_rows_in),
                "detect_cover": float(np.mean([r["pack_cover"] for r in detect_rows_in])),
            },
        )
        # keep a copy of packs slice used
        run.artifact(
            "packs_slice.json",
            {"detect": detect_rows_in, "eval": eval_rows_in, "source": str(packs_path)},
        )

        run.event("model_load_start", {"model": "kev-4b-mlx-8bit"})
        install_attention_capture()
        ck, tok, model = load_kev4b_mlx8()
        layer_ids = layer_module_ids(model.text)
        run.event(
            "model_ready",
            {"requested": ck.requested, "capture_layers": len(layer_ids), "attn_layers": list(ATTN_LAYERS)},
        )

        # ---- detection ----
        head_sums = np.zeros(N_TOTAL, dtype=np.float64)
        head_n = 0
        detect_rows = []
        topk_list = list(cfg.get("topk", [4, 8, 16]))
        budgets = list(cfg.get("budgets", [4000, 6800, 16000]))
        seed = int(cfg.get("seed", 20261012))

        for k, row in enumerate(detect_rows_in):
            state, ordered, ranges = build_labeled_state(row)
            gold = set(row["gold_in_pack"])
            tok_chunk = state_token_chunk_ids(tok, state, ranges)
            try:
                attn, p_true, wall, Ls = branch_attention(model, tok, state, layer_ids)
            except Exception as exc:
                run.event("detect_fail", {"id": row["id"], "error": str(exc)}, level="ERROR")
                continue
            qr = head_qr_scores(attn, tok_chunk, gold)
            if gold:
                head_sums += qr
                head_n += 1
            rec = {
                "id": row["id"],
                "qi": row["qi"],
                "n_pack": len(ordered),
                "chars": row["chars"],
                "n_gold_in_pack": len(gold),
                "pack_cover": row["pack_cover"],
                "p_true": p_true,
                "wall_ms": wall,
                "Ls": Ls,
                "qr_mean": float(qr.mean()),
                "qr_max": float(qr.max()),
            }
            detect_rows.append(rec)
            if (k + 1) % 5 == 0 or k == 0 or k + 1 == len(detect_rows_in):
                run.event(
                    "detect_progress",
                    {"i": k + 1, "n": len(detect_rows_in), **{kk: rec[kk] for kk in ("id", "n_gold_in_pack", "p_true", "wall_ms", "Ls")}},
                )

        if head_n == 0:
            raise RuntimeError("no detection examples with gold in pack")

        head_mean = head_sums / head_n
        ranking = np.argsort(-head_mean)
        top = {k: ranking[:k].tolist() for k in topk_list}
        rng2 = np.random.default_rng(seed + 1)
        random16 = rng2.choice(N_TOTAL, size=16, replace=False).tolist()
        all_heads = list(range(N_TOTAL))

        top_desc = []
        for rank, h in enumerate(ranking[:16]):
            layer, head = flat_to_layer_head(int(h))
            top_desc.append(
                {"rank": rank + 1, "flat": int(h), "layer": layer, "head": head, "qr": float(head_mean[h])}
            )
            run.logger.info("TOP#%02d L%dH%d qr=%.4f", rank + 1, layer, head, head_mean[h])
        run.artifact("top_heads.json", {"top16": top_desc, "by_k": {str(k): top[k] for k in topk_list}})
        run.artifact("head_mean.json", {"head_mean": head_mean.tolist(), "ranking": ranking.tolist()})
        run.artifact("detect_rows.json", detect_rows)
        run.event("detection_done", {"n_with_gold": head_n, "n_detect": len(detect_rows)})

        # ---- eval ----
        methods = {
            "hybrid": None,
            "all_heads": all_heads,
            "random16": random16,
            **{f"top{k}": top[k] for k in topk_list},
        }
        eval_summary = {m: {b: [] for b in budgets} for m in methods}
        eval_rows = []
        for k, row in enumerate(eval_rows_in):
            lengths = np.asarray(row["lengths"], dtype=np.int64)
            state, ordered, ranges = build_labeled_state(row)
            tok_chunk = state_token_chunk_ids(tok, state, ranges)
            needed = [set(g) for g in row["needed"]]
            try:
                attn, p_true, wall, Ls = branch_attention(model, tok, state, layer_ids)
            except Exception as exc:
                run.event("eval_fail", {"id": row["id"], "error": str(exc)}, level="ERROR")
                continue
            out = {"id": row["id"], "qi": row["qi"], "p_true": p_true, "wall_ms": wall, "Ls": Ls}
            for m, heads in methods.items():
                if m == "hybrid":
                    for b in budgets:
                        picked, used = pack(np.asarray(row["hy_r"], dtype=int), lengths, 1, b)
                        hit = bool(covered(set(map(int, picked)), needed))
                        eval_summary[m][b].append(hit)
                        out[f"{m}@{b}"] = hit
                        out[f"{m}@{b}_chars"] = float(used)
                    continue
                scores = chunk_scores_from_attn(attn, tok_chunk, np.asarray(heads, dtype=int))
                scores = {i: scores.get(i, 0.0) for i in ordered}
                order = sorted(scores.keys(), key=lambda i: (-scores[i], i))
                for b in budgets:
                    picked, used = pack(order, lengths, 0, b)
                    hit = bool(covered(set(map(int, picked)), needed))
                    eval_summary[m][b].append(hit)
                    out[f"{m}@{b}"] = hit
                    out[f"{m}@{b}_chars"] = float(used)
            eval_rows.append(out)
            if (k + 1) % 5 == 0 or k == 0 or k + 1 == len(eval_rows_in):
                run.event(
                    "eval_progress",
                    {"i": k + 1, "n": len(eval_rows_in), "id": row["id"], "p_true": p_true, "wall_ms": wall},
                )

        table = {
            m: {
                str(b): {
                    "cover": float(np.mean(eval_summary[m][b])) if eval_summary[m][b] else None,
                    "n": len(eval_summary[m][b]),
                }
                for b in budgets
            }
            for m in methods
        }
        metrics = {
            "n_detect": len(detect_rows),
            "n_detect_with_gold": head_n,
            "n_eval": len(eval_rows),
            "detect_pack_cover": float(np.mean([r["pack_cover"] for r in detect_rows])),
            "detect_wall_ms_mean": float(np.mean([r["wall_ms"] for r in detect_rows])),
            "eval_wall_ms_mean": float(np.mean([r["wall_ms"] for r in eval_rows])) if eval_rows else None,
            "cover": table,
            "top16": top_desc,
            "random16": random16,
            "model": ck.requested,
        }
        run.save_metrics(metrics)
        run.artifact("eval_rows.json", eval_rows)

        # human-readable table into log + results/tables
        lines = [f"{'method':12s}" + "".join(f"{b:>10d}" for b in budgets)]
        for m in methods:
            cells = []
            for b in budgets:
                v = table[m][str(b)]["cover"]
                cells.append(f"{v:10.3f}" if v is not None else f"{'—':>10s}")
            lines.append(f"{m:12s}" + "".join(cells))
        table_txt = "\n".join(lines) + "\n"
        run.logger.info("COVER TABLE\n%s", table_txt)
        tables_dir = RESULTS / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        table_path = tables_dir / f"{run.run_id}_cover.md"
        table_path.write_text(
            f"# Cover — {run.run_id}\n\n```\n{table_txt}```\n",
            encoding="utf-8",
        )
        run.event("table_written", {"path": str(table_path)})

        # mirror description into experiment folder if present
        exp_dir = ROOT / "experiments" / spec.experiment_id
        if exp_dir.exists():
            latest = exp_dir / "LATEST_RUN.txt"
            latest.write_text(str(run.dir) + "\n", encoding="utf-8")
            shutil.copy2(run.dir / "DESCRIPTION.md", exp_dir / "LAST_DESCRIPTION.md")

        run.finish(
            {
                "cover_top4_4k": table.get("top4", {}).get("4000", {}).get("cover"),
                "cover_hybrid_4k": table.get("hybrid", {}).get("4000", {}).get("cover"),
                "delta_top4_vs_hybrid_4k": (
                    None
                    if table.get("top4", {}).get("4000", {}).get("cover") is None
                    else table["top4"]["4000"]["cover"] - table["hybrid"]["4000"]["cover"]
                ),
                "run_dir": str(run.dir),
            }
        )
        return 0
    except Exception as exc:
        run.fail(exc)
        run.finish({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
