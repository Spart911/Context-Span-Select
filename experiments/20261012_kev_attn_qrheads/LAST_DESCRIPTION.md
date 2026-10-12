# Головы внимания Kev-4B как селектор evidence-кусков

- **experiment_id:** `20261012_kev_attn_qrheads`
- **run_id:** `20261012_kev_attn_qrheads_20261012T112207Z`
- **method:** kev_attn_qrhead
- **dataset:** internal_spans_hybrid16k_packs
- **metrics:** pack_cover@4000, pack_cover@6800, pack_cover@16000, head_qr_score, wall_ms
- **budgets:** 4000, 6800, 16000
- **tags:** kev, attention, qrhead, evidence-selection

## Гипотеза

Разреженные full-attention головы внутри Kev-4B концентрируются на gold evidence span'ах;
ранжирование кусков пачки по top-K таких голов обходит hybrid-упаковку на жёстких бюджетах
символов (4k / 6.8k) без отдельного вызова noul на каждый кусок.

## Описание

Порт пилота Kev attention QR-head в Context-Span-Select. Для каждого вопроса загружаем
hybrid@16k пачку пассажей, делаем один forward gate Kev, захватываем внимание query→state
по восьми слоям полного внимания (128 голов), скорим головы по доле внимания на gold
(QR-стиль), затем переупаковываем пассажи под фиксированные бюджеты через top-K / all /
random головы против baseline hybrid. Для каждого шага обязателен строгий трейсинг
JSONL + logfile.

## Заметки

Нужны mlx-lm и пакет kev (`uv --extra serve` из `~/.cache/kev`). Файл пачек уже должен
лежать в `data/processed`.
