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

Каждый эксперимент кладём в `experiments/<YYYYMMDD>_<slug>/` с `README.md` и симлинком или копией конфига из `configs/`. Метрики пишем в `results/runs/<run_id>.json`, графики — в `results/figures/`.

## Статус

Пока только каркас: структура пакета, конфиги, каталог бенчмарков, заглушка CLI. Селекторы и eval-циклы — дальше.
