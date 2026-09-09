# Deterministic Exhaustive PubMed Search

A web tool for **reproducible, exhaustive** literature searches over PubMed. You
describe a research question; an LLM (via OpenRouter) proposes a search
structure; everything after that is deterministic and auditable. The result is a
single Boolean PubMed query (with a content hash), every matching record, and a
saved protocol you can re-run to get the same set.

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

## Layout

```
backend/
  build_index.py      MeSH .nt → SQLite (one-time)
  mesh_index.py       label lookup · tree explosion · entry-term expansion
  domain_vocab.py     domain jargon clusters (data/domain_terms.json)
  openrouter_client.py  question → proposed concepts (structure only)
  query_builder.py    deterministic Boolean compile + inclusion/exclusion + hash
  pubmed.py           E-utilities esearch/efetch + CSV/JSONL export
  app.py              FastAPI + static UI
frontend/             index.html · app.js · style.css  (no build step)
data/
  domain_terms.json   editable domain vocabularies
  mesh.sqlite         generated index
  searches/           saved runs (results + protocol.json)
```

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
```
