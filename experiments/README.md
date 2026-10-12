# Experiments

One directory per experiment:

```text
experiments/YYYYMMDD_short_slug/
  README.md          # hypothesis, setup, conclusion
  config.yaml        # copy or extend from configs/
  notes.md           # optional
```

Heavy outputs go to `results/runs/`, `results/tables/`, `results/figures/` — not into this folder.

## Template

```bash
slug=$(date +%Y%m%d)_hotpot_overlap_smoke
mkdir -p experiments/$slug
cp configs/benchmarks/hotpotqa_distractor.yaml experiments/$slug/config.yaml
```

Write a short `README.md` before the run: goal, selector, budgets, expected metric.
