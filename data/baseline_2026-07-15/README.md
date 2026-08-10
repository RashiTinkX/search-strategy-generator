# Baseline run — 2026-07-15 (superseded)

The first determinism experiment: 11 models × 3 queries × 10 runs, prompt v1, the
pre-canonicalisation pipeline, produced by the original `test.py` harness (that
file was later overwritten and is not recoverable; `eval/` replaces it).

Kept for the record, **not** current. Two reasons not to read these numbers as
they were reported:

- `query_hash` determinism 1.00 was measured **with `--cache`**, which pins one LLM
  output per (model, question) — reproducibility by construction, not measurement.
- cross-model agreement 0.09 was at n=11, where chance level *is* 0.09. Cross-model
  agreement is not comparable across different model-set sizes.

Current results: `../determinism_v2.json`, `../determinism_v2_span.json`,
`../determinism_v2.png` (+ `_dark`), `../determinism_v2.md`, and the offline
ablation over these proposals in `../replay_ablation.json`. Narrative in
`../../docs/architecture.md`.
