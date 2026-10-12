# Общепринятые бенчмарки для Context-Span-Select

Задача: **отбор span/пассажей под бюджет** + опциональный **refuse** + качество ответа reader LLM.  
Ниже — то, что стабильно используют статьи по context compression / RAG pruning / evidence selection (2024–2026).

## Уровни оценки

| Уровень | Метрика | Зачем |
|---|---|---|
| A. Evidence / pack | recall supporting facts, pack cover @ budget | селектор без reader |
| B. Gate / refuse | AUROC, F1 on unanswerable / NONE | не звать LLM впустую |
| C. End-to-end | EM / F1 / Rouge / accuracy reader | польза сжатия для ответа |
| D. Cost | tokens kept, latency, $/query | практичность |

## Tier 1 — брать в первую очередь

| ID | Бенчмарк | Что меряем | Почему стандарт |
|---|---|---|---|
| `hotpotqa_distractor` | [HotpotQA](https://hotpotqa.github.io/) distractor | supporting-fact recall @k / @budget; multi-hop QA F1 | Везде как эталон evidence selection (AT2, Provence, …) |
| `nq` | [Natural Questions](https://ai.google.com/research/NaturalQuestions) | long/short answer; null answers | Классика ODQA + сжатия (LongLLMLingua, QUITO, ECoRAG) |
| `squad_v2` | [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | EM/F1 + unanswerable | Калибровка refuse gate |
| `longbench_qa` | [LongBench](https://github.com/THUDM/LongBench) EN QA subsets (`qasper`, `multifieldqa_en`, `hotpotqa`, `2wikimqa`, `narrativeqa`) | reader score при сжатии | Де-факто suite для long-context compression (Sentinel, DAC, AttentionRAG) |
| `qasper` | [QASPER](https://allenai.org/data/qasper) | evidence paragraphs + answer | Длинные scientific docs, ближе к НИР |

## Tier 2 — сильные внешние сравнения

| ID | Бенчмарк | Что меряем | Зачем |
|---|---|---|---|
| `benchpress` | [BenchPress](https://github.com/lil-lab/benchpress) (10 QA, short + LongBench mid) | единый compression suite | Свежий стандартизованный baseline 2025 |
| `asqa` | [ASQA](https://github.com/google-research/google-research/tree/master/asqa) | long-form + groundedness | Ambiguous QA; часто рядом с Trust-Score |
| `beir` | [BEIR](https://github.com/beir-cellar/beir) | nDCG@10 | Честное сравнение hybrid / dense / BM25 |
| `rgb` | [RGB](https://github.com/chen700564/RGB) | noise / refusal robustness | Стресс RAG-пайплайна |
| `longmemeval` | [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | R@k evidence in long chat | QRHead / long-session retrieval |

## Tier 3 — позже

| ID | Бенчмарк | Комментарий |
|---|---|---|
| `loft_rag` | LOFT RAG 128k (HotpotQA, MuSiQue, NQ, …) | frontier long-context vs retrieval |
| `longbench_v2` | LongBench v2 | очень длинный контекст, MCQ |
| `musique` | MuSiQue | жёсткий multi-hop |
| `refusalbench` | RefusalBench / GaRAGe | selective refusal |
| `mirage` | MIRAGE | компактный RAG pool |
| `qasa` | QASA | scientific associative selection |

## Рекомендуемый порядок внедрения в репо

1. **HotpotQA distractor** → метрика A (supporting facts), без reader.  
2. **NQ + SQuAD 2.0** → метрика B (HAS/NONE / unanswerable).  
3. **LongBench EN QA** → метрика C при фиксированных бюджетах токенов.  
4. **QASPER** → длинные одиночные документы.  
5. **BenchPress** → одна таблица, сопоставимая с чужими компрессорами.

Внутренний FOMC / handmade корпус — stress-set, не замена публичным бенчмаркам.

## Реестр в репо

Машиночитаемый список: [`data/manifests/benchmarks.yaml`](../data/manifests/benchmarks.yaml).  
Скачивание: `python scripts/download_benchmarks.py`.
