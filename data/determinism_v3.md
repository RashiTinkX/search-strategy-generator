# Determinism results (7 models × 3 questions × 3 runs, cache OFF, strict ON)

Chance-level cross-model agreement at n=7 is 0.14; numbers are not comparable across different model-set sizes.

| strategy | within-model | cross-model | heading Jaccard | block-count agr. | PMID Jaccard | median hit spread |
|---|---|---|---|---|---|---|
| llm · prompt v1 (old) | 0.57 | 0.14 | 0.25 | 0.62 | 0.27 | 51,605 |
| llm · prompt v2 | 0.71 | 0.24 | 0.47 | 0.57 | 0.18 | 16,765 |
| hybrid · model picks synonyms | 0.75 | 0.52 | 0.74 | 0.76 | – | – |
| hybrid · derived synonyms | 0.90 | 0.76 | 0.95 | 0.76 | 0.68 | 348 |
| mesh_only · no LLM | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |

## What the columns mean

| column | definition |
|---|---|
| **within-model** | One model, one question, run 3×: the share of runs that compiled the **byte-identical** query. 1.00 means rerunning your own search is safe. |
| **cross-model** | Different models, same question: the share that compiled the byte-identical query (modal share). **Chance level is 1/number-of-models**, so this number is meaningless without knowing how many models were in the run, and is not comparable across runs of different size. |
| **heading Jaccard** | Mean pairwise overlap of the chosen MeSH heading *sets* — \|A∩B\| / \|A∪B\|. Ignores wording, order and formatting, so it measures whether models agreed on the *substance*. 1.00 = same headings. |
| **block-count agr.** | Share of models that agree on how many ANDed facets the question has. Catches the "one model used 1 block, another used 7" failure directly. |
| **PMID Jaccard** | Mean pairwise overlap of the PMID sets PubMed **actually returns** for each pair of queries. 1.00 = the two searches retrieve the same papers. This is the metric a reviewer should care about: two differently-worded queries can retrieve one corpus, and two similar-looking queries can retrieve different ones. Measured on **complete** sets — a question where any model's result set exceeds the cap (25,000) is reported as not measured rather than compared on a truncated prefix. For `mesh_only` there is only one query, so its 1.00 is true by construction, not measured. |
| **median hit spread** | Across models, the largest hit count minus the smallest, median over the questions. An absolute-count companion to PMID Jaccard: a spread of 356,496 means some model's query was wildly broader than another's. |

## What the strategies mean

| strategy | meaning |
|---|---|
| **llm** | The model proposes the concept blocks and MeSH headings freely. `prompt v1` is the original prompt, `prompt v2` the rewritten one. |
| **hybrid** | A deterministic MeSH lookup over the question produces a numbered candidate slate; the model may only *select* ids from it, so it cannot invent vocabulary. |
| **span grouping** | Who decides which headings share an OR block. The *question* decides: candidates found from overlapping question words are alternatives for one facet; candidates from disjoint words stay ANDed. The alternative (`model grouping`) lets the model group them, which let one model OR a technique with a brain region. |
| **closure** | Who picks the synonyms *inside* a chosen facet. With closure, the facet's canonical vocabulary is derived from the slate (every exact match, plus reworded matches at least as well-supported); the model only decides whether the facet belongs. Without it, the model names the synonyms — and models disagree there far more than they disagree about facets. |
| **mesh_only** | No LLM at all — every maximal MeSH match in the question becomes a block. Model-independent by construction. |

Two policies were measured and rejected: **slot merging** (merging blocks the model
gave the same PICO label — models label identical content with different slots, and
it cost 0.05 cross-model), and **model grouping** (letting the model decide which
headings share an OR block — one model ORed a technique with a brain region).

A `–` means the row was computed offline from stored selections, so it never issued
PubMed queries and has no retrieval numbers.

