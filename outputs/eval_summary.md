# Observability metrics

Згенеровано: 2026-09-02

```
Total cases:            10

Success rate:           10/10 = 100%
Partial success:        0/10 = 0%
Failure rate:           0/10 = 0%

Groundedness good:      6/10 = 60%
Groundedness partial:   0/10 = 0%
Groundedness bad:       0/10 = 0%
Groundedness n/a:       4/10 = 40%

Answer quality good:    10/10 = 100%
Answer quality partial: 0/10 = 0%
Answer quality bad:     0/10 = 0%

Average latency:        29,968 ms
Median latency:         18,465 ms
Min latency:            9,658 ms
Max latency:            110,592 ms

Error types:
  none:                  10

Routes taken:
  RAG:                   4
  fallback:              2
  tool:                  2
  RAG + tool:            2

Total cost of eval run: $0.6764
```

---

## Що вимірюється автоматично, а що оцінено вручну

Розділення навмисне, бо змішувати їх — значить видавати власні евристики за об'єктивні метрики.

| Колонка | Джерело |
|---|---|
| `route_or_mode`, `tools_used` | зі сліду виконання |
| `retrieved_chunks` | з результату `search_labour_law` |
| `latency_ms` | вимірювання навколо виклику |
| `errors` | механічні правила: не той маршрут, відсутня цитата, відсутня відмова там, де вона потрібна |
| `task_success`, `groundedness`, `answer_quality` | **прочитано вручну** і збережено в `eval_set.LABELS` |

Мітки лежать у коді як дані, а не в голові автора: звіт перегенеровується без повторного платного прогону, а самі оцінки можна перевірити рядок за рядком.
