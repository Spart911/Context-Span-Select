# 20261012_kev_attn_qrheads

## Название

Головы внимания Kev-4B как селектор evidence-кусков

## Гипотеза

Внутри decision-модели Kev-4B (LoRA на Qwen3.5-4B-Base + pointer head) есть разреженное
подмножество full-attention голов, которые при вопросе `noul` смотрят на gold-доказательства
сильнее остальных. Top-K таких голов должны лучше ужимать hybrid-пачку при бюджетах 4k и 6.8k
символов, чем сам hybrid, все головы и случайные 16 голов.

## Описание

Эксперимент переносит в Context-Span-Select механизм поиска «какие куски текста отвечают на
вопрос» через веса внимания Kev, а не через отдельный вызов noul на каждый кусок.

Пайплайн:

1. Вход: заранее собранные hybrid+w1@16k пачки (внутренний корпус FOMC/papers/QA с точными span).
2. State = вопрос + куски с метками `[Pxx]`.
3. Ветка Kev: gate-инструкция + options no/yes + `<decide>`.
4. Захват softmax-внимания строк инструкции/`<decide>` на токены state по 8 слоям полного
   внимания × 16 голов = 128 кандидатов.
5. Детекция: QRscore = доля внимания на gold-куски среди внимания на все куски пачки; top-K.
6. Оценка: переранжирование кусков пачки баллами top-K / all / random / hybrid при бюджетах
   4k / 6.8k / 16k; метрика pack cover (все gold-группы ⊆ выбранных кусков).

## Метод

`kev_attn_qrhead` — Kev-4B в MLX-8bit, захват внимания на Qwen3NextAttention,
ранжирование голов в стиле QR, жадная упаковка по усреднённым по головам скорам кусков.

## Датасет

`data/processed/internal_spans/attn_packs.json` — срезы detect/eval из внутреннего span-QA
корпуса (train split), hybrid-пачки посчитаны заранее.

## Метрики

- pack cover @ 4000 / 6800 / 16000 символов
- QRscore по головам (детекция)
- wall_ms на пример
- p(true) gate score (логируется, не оптимизируется)

## Контроли

- hybrid+w1 (базовый packer)
- все 128 голов
- 16 случайных голов
- top-4 / top-8 / top-16 обнаруженных голов

## Как запускать

```bash
cd ~/.cache/kev
PYTHONPATH="/path/to/Context-Span-Select/src" \
  uv run --extra serve python \
  /path/to/Context-Span-Select/scripts/run_kev_attn_experiment.py \
  --config /path/to/Context-Span-Select/experiments/20261012_kev_attn_qrheads/config.yaml
```

Smoke-прогон:

```bash
... --smoke 8
```

## Артефакты

Каждый прогон пишет в `results/runs/<run_id>/`:

- `DESCRIPTION.md`, `description.json`, `config.json`
- `run.log`, `events.jsonl`
- `metrics.json`, `summary.json`
- `artifacts/` (строки detect/eval, ранжирование голов, срез пачек)

Путь последнего прогона: `LATEST_RUN.txt` (после успешного запуска).

## Результаты (полный прогон: detect=128, eval=100)

Прогон: `20261012_kev_attn_qrheads_20261012T112207Z` (~55 мин).

| метод | @4k | @6.8k | @16k |
|---|---:|---:|---:|
| hybrid | 0.81 | 0.87 | 0.94 |
| all_heads | 0.69 | 0.83 | 0.94 |
| random16 | 0.67 | 0.81 | 0.94 |
| **top4** | **0.91** | **0.93** | 0.94 |
| top8 | 0.89 | 0.92 | 0.94 |
| top16 | 0.88 | 0.92 | 0.94 |

Топ голов: L19H2, L19H11, L19H10, L15H11. Δ top4−hybrid @4k = **+0.10**.

## Графики

- [`results/figures/20261012_kev_attn_cover_by_budget.png`](../../results/figures/20261012_kev_attn_cover_by_budget.png) — cover vs бюджет
- [`results/figures/20261012_kev_attn_delta_vs_hybrid.png`](../../results/figures/20261012_kev_attn_delta_vs_hybrid.png) — Δ vs hybrid
- [`results/figures/20261012_kev_attn_top16_qr.png`](../../results/figures/20261012_kev_attn_top16_qr.png) — top-16 QR-голов

## Вывод

Разреженные QR-ранжированные головы внутри Kev обходят hybrid-упаковку на жёстких бюджетах;
более плотные наборы голов деградируют к уровню all-heads.
