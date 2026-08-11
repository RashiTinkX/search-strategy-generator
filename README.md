# Deterministic Exhaustive PubMed Search

A web tool for **reproducible, exhaustive** literature searches over PubMed. You
describe a research question; the search structure is proposed, then resolved
against a local MeSH index; everything after that is deterministic and auditable.
The result is a single Boolean PubMed query (with a content hash), every matching
record, and a saved protocol you can re-run to get the same set.

Scope is deliberately **PubMed only**. The concept blocks the pipeline produces
(MeSH headings + free-text per facet) are database-agnostic, so rendering them for
another database later is a rendering problem, not a redesign — but nothing here
queries Crossref, OpenAlex, Europe PMC or Unpaywall today.

## How it works

```
                          research question
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      │ hybrid (default)          │ llm                       │ mesh_only
      ▼                           ▼                           ▼
candidate_slate()            🤖 model proposes           matched_spans()
MeSH lookup over the           blocks + headings         every maximal MeSH
question → numbered slate      from scratch              match in the question
      ▼                           │                           │
🤖 model SELECTS ids only         │                           │
(cannot write vocabulary)         │                           │
      └───────────────────────────┼───────────────────────────┘
                                  │
      │  canonical.canonicalize_blocks() — exact-only resolution · subsumption
      │  pruning · canonical ordering; hybrid also gets span grouping + facet
      │  closure (candidates.py), which is where cross-model agreement comes from
      ▼
MeSH explosion (tree numbers) + entry-term expansion + domain jargon
      │  deterministic query builder → Boolean query + sha256 hash
      ▼
NCBI E-utilities — esearch history → efetch every record, split by publication
                   date past NCBI's 9,999-record ceiling
      ▼
results table · CSV / JSONL export · protocol.json (reproducibility artifact)
```

**Why it's exhaustive.** PubMed's `[MeSH Terms]` already auto-explodes indexed
articles. The additional recall comes from injecting *free-text* synonyms — entry
terms of the selected descriptors (bounded, see `STRICT_MAX_TERMS`) plus
domain-specific jargon (RNA-seq, connectome, optogenetics…) — as
`[Title/Abstract]` terms. Those catch records PubMed hasn't MeSH-indexed yet
(recent papers, ahead-of-print). Retrieval itself pages through *every* hit: past
9,999 records the query is split into publication-date slices, because NCBI refuses
`retstart > 9998`.

**Why it's deterministic.** The model never writes the query. In `hybrid` mode it
cannot even write vocabulary — it picks numbered candidates off a slate that is a
pure function of (question, MeSH index), and the deterministic layer decides how
they group and which synonyms come along. You review and edit before anything is
searched. See **[measured reproducibility](#measured-reproducibility)** for how far
this actually goes.

## Three search strategies

Pick one per search (the *Strategy* dropdown, or `mode` on `/api/map`):

| mode | who chooses what | use it when |
|---|---|---|
| **`hybrid`** (default) | a deterministic MeSH lookup proposes candidates; the model selects which *facets* belong. It cannot invent a heading, choose the synonyms inside a facet, or decide the grouping. | almost always — best balance of judgement and reproducibility |
| **`llm`** | the model proposes blocks and headings freely (prompt `v2`; `v1` kept for comparison) | you want vocabulary the question's own words don't reach, and will review it |
| **`mesh_only`** | no model at all — every maximal MeSH match in the question becomes a block | you need a model-independent protocol, or as the fallback when a model call fails |

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
  (30,956 descriptors · 64,883 tree links · 265,694 entry terms). Only English
  (or untagged) `rdfs:label`s are kept; the dump contains a handful of
  non-English labels that PubMed cannot match. The build writes to a temp file and
  swaps, so an interrupted rebuild cannot leave a truncated index behind.

## Using it

1. **Research question** — type it, tick relevant domains (bioinformatics /
   neuroscience add current jargon), pick a strategy, optionally add scope notes.
   Click *Map*.
2. **Concepts** — review each block. Every heading shown is an exact MeSH match;
   candidates that did not resolve, headings pruned as redundant (already covered
   by a broader heading's explosion), and question phrases MeSH does not contain
   are all reported above the blocks. Toggle *Explode*, click **+ entry terms** to
   fold a descriptor's synonyms into the block, edit free-text freely.
3. **Inclusion / exclusion** — dates, languages, species, publication types to
   include or exclude, terms/fragments to exclude.
4. **Compile & search** — *Compile* shows the exact query + hash; *Preview count*
   asks PubMed how many hits; *Run exhaustive search* fetches every record (up to
   *Max records* — raise it to pull everything).
5. **Results** — table (title, authors, journal, year, DOI, PMID) with **CSV /
   JSONL / protocol** export, saved under `data/searches/<time>_<hash>/`.

*Strict MeSH synonyms* (on by default) derives every free-text term from the MeSH
index instead of model prose. Leave it on unless you specifically want the model's
wording in the query — it is the difference between a query that is a function of
your selections and one that is a function of which model you called.

## Measured reproducibility

`eval/` measures this rather than asserting it. Headline, 7 models × 3 questions ×
3 runs, map cache off (`data/determinism_v3.md` has the full table with every
column defined, `data/determinism_v3.png` the figure):

| strategy | within-model | cross-model | heading Jaccard | PMID Jaccard |
|---|---|---|---|---|
| `llm` / prompt v1 | 0.57 | 0.14 | 0.25 | 0.27 |
| `llm` / prompt v2 | 0.71 | 0.24 | 0.47 | 0.18 |
| `hybrid` | **0.90** | **0.76** | **0.95** | **0.68** |
| `mesh_only` | 1.00 | 1.00 | 1.00 | 1.00* |

*within-model* = same model, rerun, same query. *cross-model* = different models,
same query — chance level is 1/7 = **0.14**, which is exactly where the original
prompt sits. *PMID Jaccard* = overlap of the papers PubMed actually returns.
\* `mesh_only` produces one query for everyone, so its 1.00 is true by
construction.

Read these with two caveats. Run-to-run variance is real at this sample size — an
identical-configuration run scored `hybrid` at 0.90 cross-model rather than 0.76,
so treat single-decimal differences as noise. And **cross-model agreement is not
comparable across runs with different numbers of models**, because the chance floor
moves. Details and the per-question breakdown: `docs/architecture.md`.

Practical consequence: reproducibility is per **(question, model, strategy)**, so
`protocol.json` records all three. If another lab must reproduce the search with a
different model, use `mesh_only`.

## Layout

```
backend/
  build_index.py      MeSH .nt → SQLite (one-time)
  mesh_index.py       label lookup · tree explosion · entry terms · parents
  candidates.py       question → MeSH candidate slate · span groups · facet closure
  canonical.py        exact-only resolution · subsumption pruning · canonical order
  prompts.py          the two model roles (propose / select) + prompt v1 for A/B
  pipeline.py         one entry point for the three strategies (app + eval share it)
  openrouter_client.py  provider routing (OpenRouter / Ollama / HF), lenient JSON
  domain_vocab.py     domain jargon clusters (data/domain_terms.json)
  query_builder.py    deterministic Boolean compile + inclusion/exclusion + hash
  pubmed.py           E-utilities, date-partitioned retrieval, CSV/JSONL export
  app.py              FastAPI + static UI
frontend/             index.html · app.js · style.css  (no build step)
eval/                 determinism_eval.py · replay_baseline.py · plot_report.py
                      selftest.py — see eval/README.md
data/
  domain_terms.json   editable domain vocabularies
  mesh.sqlite         generated index
  determinism_v3.*    current results (json · png · md table)
  baseline_2026-07-15/  the superseded first experiment, with a README
  searches/           saved runs (results + protocol.json)
```

Before committing a change to `canonical.py` or `candidates.py`, run
`.venv/bin/python eval/selftest.py` — 31 checks, no network, and it pins the rules
cross-model agreement depends on.

## Extending domain coverage

MeSH lags fast-moving fields. Edit `data/domain_terms.json` — add synonym clusters
under an existing domain or a new one. It's hot-reloaded; new domains appear as
chips in the UI. Each cluster is `{"concept": "...", "synonyms": ["...", "..."]}`.
The file shipped here is a small seed — extend it with your review's vocabulary,
since in strict mode it (plus the question's own unmatched phrases) is what carries
non-MeSH jargon into the query.

## Notes & limits

- **Determinism caveat:** the *query* is deterministic; PubMed's underlying index
  changes over time, so counts drift. `protocol.json` pins the query, hash, count,
  date, model, strategy and the date slices used.
- **Query length:** free-text expansion is bounded (`STRICT_MAX_TERMS`, default 60
  per concept) so a compiled query stays a few KB and fits PubMed's own search box
  (~8 KB URL limit). E-utilities requests go via POST regardless.
- **Dead terms:** PubMed reports some inverted MeSH entry forms
  (`"Determination, RNA Sequence"`) as *not found* — on one 81-term query, 23 of
  them. They cost query length, not correctness; pruning them is not done yet.
- **Facet omission is the residual disagreement.** Models legitimately differ on
  whether a facet is required (is `Hippocampus` essential to that question?). No
  canonicalisation can settle that, which is why the review step exists.
- LLM screening of abstracts against inclusion/exclusion is intentionally *not*
  wired in. The criteria here are deterministic PubMed filters. Abstract-level
  screening could be added as a later stage.
