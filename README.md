# Context-Span-Select

Адаптивный **выбор evidence / span**: сжимаем длинный документ (или набор retrieved-фрагментов) до упакованного бюджета span'ов, чтобы отвечающая LLM не видела весь текст целиком.

Основной цикл:

```text
документ / корпус
  → кандидаты (hybrid / BM25 / dense)
  → селектор span'ов (attention heads, decision model, …)
  → опциональный refuse-гейт
  → упакованные span'ы → reader LLM
```

## Структура

```text
app/                  # CLI и будущий локальный API
configs/              # YAML-конфиги экспериментов и бенчмарков
data/
  raw/                # скачанные оригиналы (в gitignore)
  processed/          # нормализованные пакеты для экспериментов
  manifests/          # реестр датасетов (в репозитории)
  cache/              # эмбеддинги / индексы (в gitignore)
docs/                 # дизайн-заметки, каталог бенчмарков
experiments/          # одна папка на прогон (рецепты в репозитории, без тяжёлых артефактов)
notebooks/            # исследовательский анализ
results/
  runs/               # JSON-метрики по прогонам
  tables/             # CSV / markdown-таблицы
  figures/            # графики
scripts/              # скачивание / подготовка / отчёты
src/context_span_select/   # код библиотеки
tests/
artifacts/models/     # локальные веса (в gitignore)
logs/
```

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

css --help
pytest
```

Скачать публичные бенчмарки (HF опционально):

```bash
pip install -e ".[hf]"
python scripts/download_benchmarks.py --list
python scripts/download_benchmarks.py --only hotpotqa_distractor --limit 200
```

## Бенчмарки

Публичные наборы, которые считаем стандартными для этой задачи, перечислены в [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Приоритет:

1. **HotpotQA** (recall supporting facts)
2. **Natural Questions** / **SQuAD 2.0** (unanswerable → refuse-гейт)
3. **LongBench** single/multi-doc QA (длинный контекст + сжатие)
4. **QASPER** (длинные научные документы + evidence)
5. **BenchPress** / **LOFT**-подобные suites как кросс-paper baselines

## Эксперименты

Каждый эксперимент — папка `experiments/<YYYYMMDD>_<slug>/` с `README.md` (гипотеза, метод, данные) и `config.yaml`.

**Обязательный лог прогона** (создаёт `ExperimentRun`):

```text
results/runs/<run_id>/
  DESCRIPTION.md      # человекочитаемое описание
  description.json    # то же в JSON
  config.json         # снимок конфига
  run.log             # текстовый лог
  events.jsonl        # структурированные события
  metrics.json        # финальные метрики
  summary.json        # статус + пути
  artifacts/          # head ranking, per-example scores, …
```

### Kev attention (QR-heads → top-K chunks)

Какие куски pack'а «отвечают» на вопрос: attention heads Kev-4B (MLX-8bit), отбор по QR gold-mass на detect, оценка pack-cover на eval.

```bash
# пакеты: data/processed/internal_spans/attn_packs.json
# окружение: ~/.cache/kev (mlx + kev), не .venv репозитория
ROOT="/abs/path/Context-Span-Select"
cd ~/.cache/kev
PYTHONPATH="$ROOT/src" uv run --extra serve \
  python "$ROOT/scripts/run_kev_attn_experiment.py" \
  --config "$ROOT/experiments/20261012_kev_attn_qrheads/config.yaml"
# smoke: добавь --smoke 8
```

Перед офлайн-загрузкой модели останови HTTP-сервер Kev (`serve_kev4b_mlx8`), иначе OOM на 16GB.

## Статус

Каркас + runner эксперимента Kev attention (QR-heads). Публичные бенчмарки — stubs; первый селектор крутится на internal packs.
