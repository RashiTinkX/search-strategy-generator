# Evaluation harness

Two scripts. Both load `../.env` themselves, so no `run.sh` needed.

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
terms; `--cache` pins each build (see the caveat below); `--out` sets the JSON
report path (default `data/determinism_v2.json`).

**Caching caveat.** `--cache` stores one build per (mode, model, question) and
replays it, which makes within-model determinism **1.00 by construction**. The
first run of this experiment used it, so its reported 1.00 was an artifact. It is
off by default here; use it only to re-test the deterministic layer over fixed LLM
output.

**Cost.** One LLM call per run per (mode, model, question). The default matrix is
`3 runs × 3 questions × 7 models × (hybrid + llm/v1 + llm/v2)` = 189 calls;
`mesh_only` makes none.

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
