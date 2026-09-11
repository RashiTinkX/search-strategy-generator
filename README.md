# Deterministic Exhaustive PubMed Search

A web tool for **reproducible, exhaustive** literature searches over PubMed. You
describe a research question; a search structure is proposed; everything after
that is deterministic and auditable. The result is a single Boolean PubMed
query (with a content hash), every matching record, and a saved protocol you
can re-run to get the same set.

Two strategies are available side by side (pick per search): `llm` asks a
model to freely propose MeSH headings and free-text (fast, higher variance);
`closure` asks a model only to mark facet boundaries in the question — voted
across several samples by self-consistency — then resolves vocabulary
afterwards as a pure function of the local MeSH index, so the model never
touches MeSH at all. See **Strategy: `closure` mode** below for why that
ordering matters and how to measure it yourself rather than take it on faith.

## How it works

```
research question
      │  (OpenRouter — the ONLY non-deterministic step; proposes structure only)
      ▼
concept blocks: MeSH heading candidates + free-text synonyms
      │  resolve every proposed heading against the LOCAL MeSH index
      │  → hallucinated headings resolve to nothing and are dropped
      ▼
MeSH explosion (tree numbers) + entry-term expansion + domain jargon
      │  deterministic query builder → Boolean query + sha256 hash
      ▼
NCBI E-utilities  (esearch history → efetch every record, no silent cap)
      ▼
results table · CSV / JSONL export · protocol.json (reproducibility artifact)
```

**Why it's exhaustive.** PubMed's `[MeSH Terms]` already auto-explodes indexed
articles. The additional recall comes from injecting *free-text* synonyms —
every entry term of the exploded descriptor subtree, plus domain-specific jargon
(RNA-seq, connectome, optogenetics…) and LLM-suggested variants — as
`[Title/Abstract]` terms. Those catch records PubMed hasn't MeSH-indexed yet
(recent papers, ahead-of-print).

**Why it's deterministic.** The LLM only *proposes*. You review and edit. The
compiled query is a pure function of your selections, hashed for reproducibility.
Re-running the same query returns the same PMID set (barring NCBI index updates).

## Setup

```bash
cd deterministic_search
cp .env.example .env        # add your OPENROUTER_API_KEY and NCBI_API_KEY
./run.sh                    # builds venv + MeSH index on first run, then serves
```

Open http://127.0.0.1:8077. Keys can also be pasted in the UI (kept in-browser
for the session only) instead of using `.env`.

### Data / index

- `mesh2025.nt` — MeSH RDF N-Triples dump (from
  <https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/>). To update to 2026, download
  `mesh2026.nt` and rebuild.
- `data/mesh.sqlite` — built once by `backend/build_index.py`
  (30,956 descriptors · 64,883 tree links · 265,695 entry terms).

## Using it

1. **Research question** — type it, tick relevant domains (bioinformatics /
   neuroscience add current jargon), optionally add scope notes. Click *Map*.
2. **Concepts** — review each block. Confirm/adjust the resolved MeSH headings
   (fuzzy matches are flagged), toggle *Explode*, and click **+ entry terms** on
   a descriptor to fold its full synonym set into the block's free-text. Edit
   free-text freely (one term per line).
3. **Inclusion / exclusion** — dates, languages, species, publication types to
   include or exclude, terms/fragments to exclude.
4. **Compile & search** — *Compile* shows the exact query + hash; *Preview count*
   asks PubMed how many hits; *Run exhaustive search* fetches every record (up to
   *Max records* — raise it to pull everything).
5. **Results** — sortable table (title, authors, journal, year, DOI, PMID) with
   **CSV / JSONL / protocol** export. Saved under `data/searches/<time>_<hash>/`.

### Quality checks and sensitivity searches

Each concept can be marked **Required**, **Optional / sensitivity**, or **Context
only**. Required concepts form the core search. Optional concepts produce a
primary query plus narrower and broader sensitivity variants; contextual concepts
are retained in the exported protocol but do not silently restrict retrieval.

Paste PMIDs for studies already known to be relevant into **Known relevant PMIDs**
and click **Check known PMIDs** after compiling. The app reports known-item recall
and missed PMIDs without fetching the full result set. This is a validation aid,
not proof that a search is exhaustive: the known set must be independent of the
strategy being assessed.

Large PubMed searches are automatically partitioned by publication date before
fetching, avoiding PubMed's 9,999-record history-server ceiling. The saved
`protocol.json` records each partition and surfaces any missing records.

## Strategy: `closure` mode (self-consistency facet voting)

`mode: "llm"` (the default above) asks the model to freely propose MeSH
headings and free-text — a high-entropy generation task, so two runs (or two
models) can reasonably disagree on vocabulary.

`mode: "closure"` (`backend/facets.py` + `backend/closure.py`) inverts what
the model is asked to do. The model is never shown MeSH at all; it is asked
only to mark WHERE each PICO-style concept sits in the question text (a much
lower-entropy task), and that call is sampled **k times** and voted by
self-consistency — a span only survives if a majority of the k runs agree on
it (`facets._cluster_and_vote`). Vocabulary is then derived afterwards, as a
pure function of (question, span, local MeSH index): every exact MeSH match
inside a voted span is kept, subject to subsumption pruning
(`closure.prune_subsumed`) so a broader heading's auto-explosion isn't
duplicated by also keeping a narrower one. No model ever sees or chooses a
MeSH heading, so nothing downstream of voting can vary run-to-run.

Falls back to a zero-network heuristic segmenter (`facets.heuristic_facets`,
conjunction/punctuation chunking) if no LLM key is configured for the chosen
model — a model-independent path the same way `mesh_only` served that
purpose upstream, just reached differently. Call it via
`{"mode": "closure", "facet_runs": 3, "min_agreement": 0.5}` on `/api/map`.

Measure it yourself before trusting either mode's number:
`.venv/Scripts/python.exe eval/eval_closure.py --models anthropic/claude-haiku-4.5 --modes llm,closure --runs 3 --retrieval`
runs both `llm` and `closure` against the same live model(s) and scores four
metrics — within-model, cross-model, heading Jaccard, and PMID Jaccard —
against real PubMed data. No API key required if you point `--models` at a
local Ollama model instead.

## Measured results

`eval/eval_closure.py` measures this rather than asserting it, same as the
upstream project's own `eval/` does for its `hybrid` mode — and the results
are directly comparable, since both use the same four metric definitions
(within-model / cross-model / heading Jaccard / PMID Jaccard).

**Headline: Claude Haiku 4.5 + Claude Sonnet 5, 6 questions × 3 runs, live retrieval**
(`data/closure_determinism_frontier.json`):

| strategy | within-model | cross-model | heading Jaccard | PMID Jaccard |
|---|---|---|---|---|
| upstream `llm` v2 (reference) | 0.71 | 0.24 | 0.47 | 0.18 |
| upstream `hybrid` (reference) | 0.90 | 0.76 | 0.95 | 0.68 |
| this project's `llm` mode | 0.583 | 0.417 | 0.609 | 0.282 |
| **this project's `closure` mode** | **0.972** | **0.972** | **1.000** | **0.839** |

The cleanest part of that result doesn't even need the upstream comparison:
`llm` mode and `closure` mode were run against the *identical* two models on
the *identical* six questions in the *same* codebase — a fully controlled
A/B test with no cross-project confound. `closure` mode's cross-model score
is more than double `llm` mode's using the same models. The single most
telling data point: Claude Sonnet 5 — a strong, capable model — scored
`within-model = 0.33` on 5 of 6 questions in `llm` mode, meaning it gave a
*different* answer on 2 of every 3 reruns of the exact same question. That's
not a weak-model artifact; it's direct evidence that unconstrained vocabulary
generation is inherently unstable, which is the instability `closure` mode's
design (self-consistency-voted span marking, zero model-chosen vocabulary) is
built to remove.

**Caveats, read before citing this table:**

- One question (RNA-seq / hippocampal neurons / Alzheimer's) scored a PMID
  Jaccard of just 0.035 in `closure` mode despite a *perfect* heading Jaccard
  of 1.00 — identical MeSH headings still retrieved almost entirely different
  papers, most likely from a free-text-term or explosion-scope difference
  invisible at the heading level. Real data has real outliers; this table's
  means don't show that unless you look at `data/closure_determinism_frontier.json`.
- A separate run with two small, free, local models (Llama 3.2 1B + Qwen 2.5
  1.5B — `data/closure_determinism_crossmodel.json`) scored `closure` mode at
  cross-model 0.60 — barely above the 0.50 chance floor for 2 models, and
  *below* what `llm` mode scored on the one question it managed to complete
  before hitting a credit wall. Model capability is a real confound: this
  architecture's advantage was only clearly demonstrated with capable models,
  not weak ones. Reported here rather than omitted.
- Comparing against upstream's *published* 0.90/0.76/0.95/0.68 still carries
  a cross-project caveat (different codebase, different exact 7 models,
  different MeSH index build) — it's a fair, good-faith comparison now that
  model capability isn't a confound on this side, but not a controlled trial
  of both codebases side by side. The `llm`-vs-`closure` comparison above,
  within this one run, doesn't have that caveat.

## Layout

```
backend/
  build_index.py      MeSH .nt → SQLite (one-time)
  mesh_index.py       label lookup · tree explosion · entry-term expansion
  domain_vocab.py     domain jargon clusters (data/domain_terms.json)
  openrouter_client.py  "llm" mode: question → freely proposed concepts
  facets.py           "closure" mode: question → voted facet spans (self-consistency)
  closure.py          "closure" mode: facet span → deterministic MeSH closure
  facet_pipeline.py   wires facets.py + closure.py into the app's concept shape
  query_builder.py    deterministic Boolean compile + inclusion/exclusion + hash
  pubmed.py           E-utilities esearch/efetch + CSV/JSONL export
  app.py              FastAPI + static UI
frontend/             index.html · app.js · style.css  (no build step)
eval/
  eval_closure.py     live determinism measurement: llm vs closure, same model
data/
  domain_terms.json   editable domain vocabularies
  mesh.sqlite         generated index
  closure_determinism*.json  eval_closure.py output -- see Measured results above
  searches/           saved runs (results + protocol.json)
tests/
  test_quality_features.py   portfolio + date-scope regression checks
  test_closure_pipeline.py   facets/closure regression checks
```

Run `.venv/Scripts/python.exe -m unittest discover -s tests` before committing a
change to `closure.py` or `facets.py` — no network required.

## Extending domain coverage

MeSH lags fast-moving fields. Edit `data/domain_terms.json` — add synonym
clusters under an existing domain or a new one. It's hot-reloaded; new domains
appear as chips in the UI. Each cluster is
`{"concept": "...", "synonyms": ["...", "..."]}`.

## Notes & limits

- **Determinism caveat:** the *query* is deterministic; PubMed's underlying index
  changes over time, so counts can drift. The saved `protocol.json` pins the
  query + hash + count + date so a run is auditable.
- **Query length:** pulling entry terms for large subtrees (e.g. *Neurons* → 820
  terms) makes long queries. E-utilities requests are sent via POST to handle
  this, but extremely large blocks may need trimming.
- LLM screening of abstracts against inclusion/exclusion is intentionally *not*
  wired in (you chose MeSH-mapping only). The criteria here are deterministic
  PubMed filters. Abstract-level screening could be added as a later stage.
- **Fixed:** `pubmed.py`'s `fetch_query` used to silently `continue` past a
  date slice with more than 9,999 hits in a single day, discarding every
  record from it with no signal. It now still fetches up to the ceiling for
  that slice and reports the shortfall via `missing`/`any_slice_truncated`,
  matching how an over-`max_records` cap is already reported.
- **Scope is deliberately PubMed only.** A multi-database (Europe PMC)
  extension was prototyped and measured, then dropped to keep this tool
  focused; see git history if you want to revisit it.
- **Fixed: `closure` mode's exhaustiveness used to be silently bounded by
  facet segmentation.** `closure.py` resolves every exact MeSH match inside a
  facet span it is given, but a question phrase that never landed inside any
  voted facet span never reached the MeSH matcher — with no signal that it
  happened. `facet_pipeline.coverage_gaps()` now detects any contiguous,
  search-worthy stretch of the question outside every voted span, and
  `resolve_coverage_gaps()` runs the same deterministic closure over it and
  appends the result as an extra, clearly-labeled block (visible in
  `coverage_gaps` in the `/api/map` response and tagged in each such block's
  `rationale`) instead of dropping it. See `tests/test_closure_pipeline.py::CoverageGapTests`.
- **Still open:** no stemming/morphological normalization anywhere in
  `closure.py` — an adjectival form ("hippocampal") will not resolve to its
  noun-form MeSH heading ("Hippocampus") unless MeSH's own entry-term table
  happens to list it. Confirmed live; not yet fixed.
