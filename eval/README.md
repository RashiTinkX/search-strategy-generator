# Evaluation harness

Four scripts. They load `../.env` themselves, so no `run.sh` needed.

Current outputs live in `../data/`: `determinism_v3.json`, `determinism_v3.png` /
`_dark.png`, `determinism_v3.md` (table view, with column definitions), the raw run
logs in `logs/`, and `replay_ablation.json`. The superseded July run is in
`../data/baseline_2026-07-15/` — see its README before quoting those numbers.

## `determinism_eval.py` — live measurement

Runs the real pipeline (`backend/pipeline.py`) N times per (strategy, model,
question) and scores four things:

| metric | question it answers |
|---|---|
| **within-model** | same model, rerun → same query? (one lab's reproducibility) |
| **cross-model** | different models → the *same* query? (portability between labs) |
| **heading Jaccard** | do models pick the same MeSH headings, ignoring formatting? |
| **retrieval** (`--retrieval`) | do the queries return the same PMIDs? (what a reviewer cares about) |

```bash
# the headline matrix: hybrid vs both prompt versions vs no-LLM, + live PubMed overlap
.venv/bin/python eval/determinism_eval.py --runs 3 --modes hybrid,llm,mesh_only \
    --prompt-versions v1,v2 --retrieval

# one model, quick
.venv/bin/python eval/determinism_eval.py --models google/gemini-3.5-flash --runs 3

# your own questions
.venv/bin/python eval/determinism_eval.py --questions my_questions.json
# [{"question": "...", "domains": ["neuroscience"]}, ...]
```

Notable flags: `--no-strict` keeps the model's free-text instead of index-derived
terms; `--cache` pins each build (see the caveat below); `--no-span-grouping` lets
the *model* decide which candidates are ORed instead of grouping them by the
question spans they came from; `--no-closure` lets the *model* pick the synonyms
inside a facet instead of deriving them from the slate; `--out` sets the JSON report
path (default `data/determinism_v3.json`).

Every hybrid run also reports a **free ablation row** — the same selections rebuilt
with the opposite closure policy (`hybrid (no closure)`). It costs no extra LLM calls
and separates "which vocabulary did the model name" from "which facets did it
choose", which is where models actually disagree.

**Caching caveat.** `--cache` stores one build per (mode, model, question) and
replays it, which makes within-model determinism **1.00 by construction**. The
first run of this experiment used it, so its reported 1.00 was an artifact. It is
off by default here; use it only to re-test the deterministic layer over fixed LLM
output.

**Cost.** One LLM call per run per (mode, model, question). The default matrix is
`3 runs × 3 questions × 7 models × (hybrid + llm/v1 + llm/v2)` = 189 calls;
`mesh_only` makes none.

**`--retrieval-from <report.json>`** recomputes only the retrieval block of an
existing report, from the queries the report already stores — no LLM calls, just
NCBI. Use it after changing how retrieval is measured (that is why it exists: the
first runs measured PMID overlap with a `pmids()` that silently stopped at NCBI's
9,999-record ceiling, so overlaps on large result sets were computed from an
arbitrary prefix).

## `replay_baseline.py` — offline ablation, no API needed

Replays the 33 archived proposals in `data/map_cache/` (the raw LLM output from the
original 11-model run, prompt v1) through each deterministic rule in turn, so the
contribution of the layer is measurable with the prompt and models held fixed:

```bash
.venv/bin/python eval/replay_baseline.py      # -> data/replay_ablation.json
```

Stages: legacy (fuzzy resolution, no pruning, model's order) → exact-only
resolution → + subsumption pruning → + canonical ordering. See
`docs/architecture.md#cross-model-determinism-three-strategies-2026-08-10` for the
result and what it implies.

Note: the cache filenames are hashes produced by the original harness (`test.py`,
since lost), so the question is recovered from each proposal's own vocabulary
(`QUESTION_KEYS`). Add an entry there if you replay a different question set.

## `plot_report.py` — figure + table view

```bash
.venv/bin/python eval/plot_report.py     # -> data/determinism_v3{,_dark}.png + .md
```

Reads the JSON report and renders one panel per metric (light and dark, each
stepped for its own surface). Re-run it after any `determinism_eval.py` run —
`determinism_eval.py` writes JSON only, so the figure is never regenerated
implicitly, which is exactly how the old July PNG came to look current for a month.

## `selftest.py` — the deterministic rules, no network

```bash
.venv/bin/python eval/selftest.py        # 31 checks
```

Pins what cross-model agreement depends on: exact-only resolution, subsumption
pruning, canonical ordering, span grouping, facet closure, "no model prose reaches a
strict query", the slate's precision cases, and the MeSH index's English-label fix. Run it before committing a change to
`canonical.py` or `candidates.py`.
