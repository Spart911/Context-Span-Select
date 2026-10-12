# Эксперименты

Одна директория на эксперимент:

```text
experiments/YYYYMMDD_short_slug/
  README.md            # гипотеза, сетап, вывод (обязателен до прогона)
  config.yaml          # рецепт; должен содержать experiment.description (≥80 символов)
  LAST_DESCRIPTION.md  # автокопия с последнего прогона
  LATEST_RUN.txt       # путь к results/runs/<run_id>/
```

## Строгий лог (обязательно)

Каждый прогон идёт через `ExperimentRun` (`context_span_select.experiment`) и пишет:

```text
results/runs/<run_id>/
  DESCRIPTION.md / description.json
  config.json
  run.log
  events.jsonl
  metrics.json
  summary.json
  artifacts/
```

Тихих прогонов без лога нет. В конфиге должны быть заполнены
`experiment.{id,title,description,hypothesis,method,dataset,metrics}`.

## Шаблон

```bash
slug=$(date +%Y%m%d)_kev_attn_qrheads
mkdir -p experiments/$slug
# скопируй существующий рецепт или configs/default.yaml, заполни поля description
```

Тяжёлые артефакты — только в `results/`, не внутри `experiments/`.
