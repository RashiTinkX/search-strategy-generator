# Determinism results (7 models × 3 questions × 3 runs, cache OFF, strict ON)

Chance-level cross-model agreement at n=7 is 0.14; numbers are not comparable across different model-set sizes.

| strategy | within-model | cross-model | heading Jaccard | block-count agr. | PMID Jaccard | median hit spread |
|---|---|---|---|---|---|---|
| llm · prompt v1 (old) | 0.46 | 0.19 | 0.32 | 0.67 | 0.33 | 356,496 |
| llm · prompt v2 | 0.71 | 0.24 | 0.44 | 0.81 | 0.51 | 38,309 |
| hybrid · model grouping | 0.73 | 0.43 | 0.71 | 0.76 | 0.39 | 1,694 |
| hybrid · span grouping | 0.75 | 0.38 | 0.70 | 0.71 | 0.56 | 1,771 |
| hybrid · + slot merging | 0.71 | 0.38 | 0.71 | 0.67 | – | – |
| mesh_only · no LLM | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
