# Deterministic PubMed Search — Architecture

How an exhaustive, **reproducible** literature search is built. The core idea:
the LLM is the *only* non-deterministic component and it merely **proposes**
structure — every proposal is resolved against a local, deterministic MeSH index
before it can affect the query. So the same question always compiles to the same
byte-identical PubMed query (and hash).

```mermaid
flowchart TD
    U(["🧑‍🔬 Researcher"]) -->|research question| FE["🖥️ Frontend<br/>vanilla JS + HTML"]

    %% ---------------- Step 1: MAP (the only non-deterministic hop) ----------------
    subgraph S1["① MAP  ·  intent understanding & keyword breakdown"]
        direction TB
        MAP["POST /api/map"]
        LLM{{"🤖 OpenRouter LLM<br/>temperature = 0<br/><i>proposes only</i>"}}
        RES["_resolve_mesh()<br/>match every proposed heading"]
        AUG["augment free-text with<br/>domain-vocab synonyms"]
        MAP --> LLM -->|"concepts + candidate<br/>MeSH names + synonyms (JSON)"| RES --> AUG
    end

    %% ---------------- Step 2: EXPAND (preview) ----------------
    subgraph S2["② EXPAND  ·  optional preview"]
        EXP["POST /api/expand"]
        EXPL["MeshIndex.expand()<br/>tree explosion + entry terms"]
        EXP --> EXPL
    end

    %% ---------------- Step 3: COMPILE (deterministic core) ----------------
    subgraph S3["③ COMPILE  ·  deterministic query builder"]
        COMP["POST /api/compile"]
        QB["query_builder.compile_search()<br/>PICO blocks: (MeSH OR free-text) AND … NOT …"]
        HASH["sha256 → query_hash<br/><i>byte-stable → reproducible</i>"]
        COMP --> QB --> HASH
    end

    %% ---------------- Step 4 & 5: COUNT / SEARCH ----------------
    subgraph S4["④ COUNT & ⑤ SEARCH  ·  PubMed retrieval"]
        CNT["POST /api/count<br/>esearch only → hits + translation"]
        SR["POST /api/search<br/>esearch + efetch every record"]
        PM["pubmed.PubMed<br/>NCBI E-utilities client"]
        CNT --> PM
        SR --> PM
    end

    %% ---------------- Step 6: outputs ----------------
    subgraph S6["⑥ EXPORT"]
        SAVE[("data/searches/&lt;stamp&gt;_&lt;hash&gt;/<br/>results.csv · results.jsonl · protocol.json")]
        DL["GET /api/download"]
    end

    %% ---------------- data stores ----------------
    NT[/"mesh2025.nt<br/>2 GB N-Triples"/]
    BUILD["build_index.py"]
    IDX[("mesh.sqlite<br/>descriptors · trees · entry_terms")]
    VOCAB[("domain_terms.json<br/>bioinformatics · neuroscience<br/><i>hot-reloaded</i>")]
    NCBI[["🌐 NCBI E-utilities<br/>esearch / efetch"]]

    NT --> BUILD --> IDX

    %% ---------------- flow ----------------
    FE --> MAP
    FE --> EXP
    FE --> COMP
    FE --> CNT
    FE --> SR
    FE --> DL

    IDX -. lookup .-> RES
    IDX -. explode .-> EXPL
    VOCAB -. synonyms .-> AUG

    AUG -->|"reviewed concepts"| FE
    EXPL -->|"preview terms"| FE
    HASH -->|"query + hash"| FE
    PM <-->|HTTP| NCBI
    PM -->|articles| SAVE
    SAVE --> DL --> U
    HASH -. same query string .-> CNT
    HASH -. same query string .-> SR

    %% ---------------- styling ----------------
    classDef nondet fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,color:#111;
    classDef det fill:#e0f2e9,stroke:#1e8449,stroke-width:1.5px,color:#111;
    classDef store fill:#eef2ff,stroke:#3b4cca,stroke-width:1.5px,color:#111;
    classDef ext fill:#fff6e0,stroke:#b9770e,stroke-width:1.5px,color:#111;
    classDef ui fill:#f3e8ff,stroke:#7d3c98,stroke-width:1.5px,color:#111;

    class LLM nondet;
    class RES,AUG,EXPL,QB,HASH,PM,COMP,MAP,EXP,CNT,SR,DL,BUILD det;
    class IDX,VOCAB,SAVE,NT store;
    class NCBI ext;
    class U,FE ui;
```

## The determinism boundary

```mermaid
flowchart LR
    Q(["same research<br/>question"]) --> L{{"🤖 LLM<br/>proposes structure<br/><b>non-deterministic</b>"}}
    L -->|"may reword /<br/>reorder keywords"| G["🧱 deterministic layer<br/>MeSH resolution · dedupe ·<br/>compile · sha256"]
    G --> H(["byte-identical<br/>query + hash ✅"])

    classDef nondet fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,color:#111;
    classDef det fill:#e0f2e9,stroke:#1e8449,stroke-width:2px,color:#111;
    class L nondet;
    class G,H det;
```

**How far this holds — and where it breaks.** The deterministic layer does real
work: a hallucinated MeSH heading resolves to *nothing* and drops out; terms are
lower-cased, de-duped, sorted; block/clause order is fixed. But we *measured* it
(`test.py`, 10 models × 10 runs) and the naive pipeline is **not** reproducible:
`query_hash` determinism was ≈ **0.10** for most models (all 10 runs → 10
different queries). At `temperature=0` the LLM still emits a **different set of
free-text synonyms — and even a different set of MeSH headings — each run**, and
those flow straight into the query. The deterministic layer can't absorb a
changing *input set*. See **[Reproducibility in practice](#reproducibility-in-practice-measured--fixed)**
for the numbers and the three mechanisms that actually get it to 1.00.

## How MeSH is used

MeSH (Medical Subject Headings) is NLM's controlled vocabulary. It does two jobs
here: it **filters LLM hallucinations** (a proposed heading that isn't real
resolves to nothing) and it **drives exhaustive recall** (tree explosion + entry
terms). All of it is local and deterministic.

### 1 · Building the index (one-time, offline)

`mesh2025.nt` (2 GB N-Triples) is distilled into a small SQLite DB. Only
subjects that carry a `preferredConcept` are kept as descriptors — that filters
out tree-number nodes that share the `D…` prefix but aren't real headings.

```mermaid
flowchart LR
    NT[/"mesh2025.nt<br/>2 GB N-Triples"/] --> B["build_index.py<br/>stream-parse triples"]
    B --> D[("descriptor<br/>dui · label · label_norm")]
    B --> T[("tree<br/>dui · tree number")]
    B --> E[("entry_term<br/>dui · term · term_norm")]
    D --- IDX[("mesh.sqlite")]
    T --- IDX
    E --- IDX

    classDef store fill:#eef2ff,stroke:#3b4cca,stroke-width:1.5px,color:#111;
    classDef det fill:#e0f2e9,stroke:#1e8449,stroke-width:1.5px,color:#111;
    class NT,D,T,E,IDX store;
    class B det;
```

| Table | Columns | Purpose |
|-------|---------|---------|
| `descriptor` | `dui`, `label`, `label_norm` | the official heading (D-unique-id) |
| `tree` | `dui`, `tree` | hierarchy positions (e.g. `G11.427.590`) → explosion |
| `entry_term` | `dui`, `term`, `term_norm` | synonyms / variants → free-text recall |

### 2 · Resolving an LLM-proposed heading (`_resolve_mesh` → `MeshIndex.search`)

Each candidate name from the LLM is normalized then matched in a fixed priority
order. A name that matches nothing simply **drops out** — that is the
hallucination filter.

```mermaid
flowchart TD
    C["LLM candidate<br/>e.g. 'Neuronal Plasticity'"] --> N["normalize<br/>lowercase · collapse spaces"]
    N --> E1{exact heading?}
    E1 -->|yes| SEL["✅ selected DUI + label<br/>matched = true"]
    E1 -->|no| E2{exact entry term?}
    E2 -->|yes| SEL
    E2 -->|no| E3{substring LIKE?<br/>ranked by label length, dui}
    E3 -->|hits| OPTS["top-8 options →<br/>user reviews &amp; picks in UI"]
    E3 -->|none| DROP["∅ dropped<br/>hallucination filtered out"]
    OPTS --> SEL

    classDef nondet fill:#ffe0e0,stroke:#c0392b,stroke-width:2px,color:#111;
    classDef det fill:#e0f2e9,stroke:#1e8449,stroke-width:1.5px,color:#111;
    class C nondet;
    class N,E1,E2,E3,SEL,OPTS,DROP det;
```

### 3 · Explosion + entry terms = exhaustive recall

The selected descriptor feeds two complementary recall paths that are ORed
inside its concept block:

```mermaid
flowchart TD
    SEL["selected DUI<br/>D009473 · Neuronal Plasticity"] --> TREE["tree number<br/>G11.561.638"]

    TREE --> EXP["MeshIndex.explode()<br/>tree = X OR tree LIKE 'X.%'<br/>→ all narrower descriptors"]

    SEL --> P1["Heading tagged [MeSH Terms]"]
    P1 --> A1["🔎 PubMed AUTO-EXPLODES:<br/>every INDEXED article under the<br/>heading and its descendants"]

    EXP --> ETS["MeshIndex.entry_terms()<br/>headings + synonyms of all descendants"]
    ETS --> DOM["+ domain_terms.json jargon<br/>+ British/US spellings, abbrevs"]
    DOM --> P2["each term tagged [Title/Abstract]"]
    P2 --> A2["🔎 catches NOT-yet-indexed &amp;<br/>preprint-style records"]

    A1 --> REC{{"exhaustive recall<br/>indexed ∪ not-yet-indexed"}}
    A2 --> REC

    classDef det fill:#e0f2e9,stroke:#1e8449,stroke-width:1.5px,color:#111;
    classDef store fill:#eef2ff,stroke:#3b4cca,stroke-width:1.5px,color:#111;
    classDef goal fill:#fff6e0,stroke:#b9770e,stroke-width:2px,color:#111;
    class SEL,TREE,EXP,ETS,P1,P2,A1,A2 det;
    class DOM store;
    class REC goal;
```

**The key insight:** `[MeSH Terms]` alone only finds articles NLM has already
manually indexed — which lags months behind publication and misses preprints. By
also injecting the exploded descendants' **entry terms** (plus domain jargon) as
`[Title/Abstract]`, the query reaches recent and not-yet-indexed records too.
That union is the "exhaustive" part, and because both halves come from the
deterministic index, the result is fully reproducible.

## Core algorithms

The system is a chain of small, deterministic algorithms wrapped around a single
LLM proposal step. Each one below is taken directly from the code.

### A · Index build — streaming N-Triples → SQLite  (`build_index.py`)

MeSH RDF is a 2 GB flat triple file; loading it into an RDF library is
unnecessary. Instead we stream it line-by-line and route each triple by the
**first letter of its subject** (`D`escriptor / `M` concept / `T` term),
reconstructing the `descriptor → concept → term → label` chain in memory, then
flushing to SQLite.

```text
for each line in mesh.nt:                       # ~ tens of millions, O(1) mem/line
    (subj, pred, obj, is_uri) = parse_ntriple(line)   # hand-rolled, no rdflib
    skip unless pred ∈ {label, treeNumber, concept, preferredConcept,
                        term, preferredTerm, prefLabel, altLabel}
    switch subj[0]:
        'D': record label / tree numbers / concept links for the descriptor
        'M': record term links for the concept
        'T': record synonym strings for the term

for each descriptor D with a (preferred) Concept:      # ← the key filter
    emit descriptor(D.dui, label, norm(label))
    emit tree(D.dui, t)            for t in D.tree_numbers
    synonyms = {D.label} ∪ {labels of every term of every concept of D}
    emit entry_term(D.dui, s, norm(s))  for s in synonyms
```

```mermaid
flowchart TD
    NT[/"mesh.nt line"/] --> P["parse_ntriple()"]
    P --> W{"predicate<br/>wanted?"}
    W -->|no| SKIP["skip"]
    W -->|yes| R{"subj[0] ?"}
    R -->|D| DD["descriptor:<br/>label · tree# · concept links"]
    R -->|M| MM["concept:<br/>term links"]
    R -->|T| TT["term:<br/>synonym strings"]
    DD & MM & TT --> AGG["in-memory maps"]
    AGG --> F{"descriptor has<br/>preferredConcept?"}
    F -->|no| DROP["drop pseudo-node<br/>e.g. D02.455"]
    F -->|yes| EMIT["emit descriptor +<br/>tree + entry_term rows"]
    EMIT --> DB[("mesh.sqlite")]

    classDef bad fill:#ffe0e0,stroke:#c0392b,color:#111;
    classDef store fill:#eef2ff,stroke:#3b4cca,color:#111;
    class DROP,SKIP bad;
    class DB store;
```

- **Why the "has a concept" filter?** Category-D tree nodes (e.g. `D02.455`) are
  also `D`-prefixed and carry an `rdfs:label`, but no `preferredConcept`.
  Requiring a concept drops those pseudo-descriptors — the gotcha that made
  early builds noisy.
- `norm(s)` = lowercase + whitespace-collapse, stored alongside the raw string
  so lookups are case/spacing-insensitive but display stays exact.

### B · Candidate resolution ladder  (`MeshIndex.search`)

Turns a free-text LLM suggestion into real descriptors via a fixed-priority
cascade; stops widening once it has enough. This is both the fuzzy matcher and
the hallucination filter.

```text
resolve(text, limit=8):
    seen = ordered-unique-by-DUI
    add  exact heading      WHERE label_norm = norm(text)          # tier 1
    add  exact entry term   WHERE term_norm  = norm(text)          # tier 2
    if |seen| < limit:
        add substring        WHERE label_norm LIKE '%norm(text)%'  # tier 3
             ORDER BY length(label), dui   LIMIT limit             # shortest first
    return seen[:limit]        #  ∅  ⇒ candidate was a hallucination, dropped
```

```mermaid
flowchart TD
    T["norm(text)"] --> T1{"exact heading?"}
    T1 -->|hit| S[("seen<br/>ordered-unique by DUI")]
    T1 -->|miss| T2{"exact entry term?"}
    T2 -->|hit| S
    T2 -->|miss| T3{"seen &lt; limit?"}
    T3 -->|yes| L["substring LIKE<br/>ORDER BY length(label), dui"]
    L --> S
    T3 -->|no| RET["return seen[:limit]"]
    S --> RET
    RET --> CHK{"empty?"}
    CHK -->|yes| H["∅ hallucination<br/>dropped"]
    CHK -->|no| OK["ranked options<br/>user picks"]

    classDef bad fill:#ffe0e0,stroke:#c0392b,color:#111;
    classDef good fill:#e0f2e9,stroke:#1e8449,color:#111;
    classDef store fill:#eef2ff,stroke:#3b4cca,color:#111;
    class H bad;
    class OK good;
    class S store;
```

- Ordering is total and deterministic (`length(label), dui`), so the same
  suggestion always yields the same ranked options — no tie-break ambiguity.

### C · Tree explosion — dotted-prefix set union  (`MeshIndex.explode`)

MeSH "explode" = a heading plus everything hierarchically beneath it. Tree
numbers encode the hierarchy as dotted paths (`G11.561.638`), so descendants are
exactly the rows whose tree number equals or is prefixed by one of the seed's:

```text
explode(dui):
    found = {dui}
    for tn in tree_numbers(dui):
        found ∪= { d : tree(d) = tn  OR  tree(d) LIKE tn || '.%' }
    return sorted(found)          # stable
```

```mermaid
flowchart LR
    SEED["seed DUI<br/>+ its tree numbers { tn }"] --> M["for each tn:<br/>tree = tn OR tree LIKE 'tn.%'"]
    M --> U(["∪ descendant DUIs<br/>(seed included)"])
    U --> SORT["sorted(found) — stable"]

    classDef good fill:#e0f2e9,stroke:#1e8449,color:#111;
    class SORT good;
```

- Purely relational; no recursion needed because the dotted path *is* the
  ancestry. `A01` → `A01`, `A01.111`, `A01.923.047`, …

### D · Deterministic query compilation + hash  (`query_builder`)

Concept blocks compile to a canonical Boolean string. Determinism comes from:
order-preserving case-insensitive de-dup, fixed clause order, and consistent
quoting.

```text
block(concept)  = "(" + OR_join(
                      [ q(h)+MeSH-tag       for h in dedupe(concept.mesh) ] +
                      [ q(t)+"[Title/Abstract]" for t in dedupe(concept.freetext) ]
                  ) + ")"

query = AND_join(non-empty blocks)                       # PICO: blocks ANDed
for clause in [date, languages, pub-types, species, custom_include]:
    query = "(query) AND clause"                         # inclusion filters
for clause in [exclude pub-types, exclude terms, custom_exclude]:
    query = "(query) NOT clause"                         # exclusions
hash  = sha256(query)[:16]                               # reproducibility key
```

```mermaid
flowchart TD
    C["concepts"] --> B["block =<br/>( MeSH[MeSH Terms] OR freetext[Title/Abstract] )<br/>after dedupe"]
    B --> AND["AND-join non-empty blocks"]
    AND --> INC["wrap ( … ) AND include<br/>date · lang · pubtype · species · custom"]
    INC --> EXC["wrap ( … ) NOT exclude<br/>pubtypes · terms · custom"]
    EXC --> Q(["query string"])
    Q --> HSH["sha256[:16]<br/>→ query_hash"]

    classDef good fill:#e0f2e9,stroke:#1e8449,color:#111;
    class Q,HSH good;
```

- `dedupe` keeps first occurrence, compares on `strip().lower()` → stable set,
  stable order. Same inputs ⇒ byte-identical `query` ⇒ identical `hash`. This is
  the invariant `test.py` measures as `query_hash` determinism.

### E · Exhaustive retrieval — history server + paging  (`pubmed.PubMed`)

To pull *every* match with no silent cap, we use the E-utilities history server:
one `esearch` stores the full result set server-side (`WebEnv` + `query_key`),
then `efetch` pages through it.

```text
search(query):  esearch(usehistory=y, retmax=0) → {count, WebEnv, query_key, translation}

fetch_all(WebEnv, query_key, count):
    target = min(count, max_records?)          # cap only if caller sets one
    for start in 0, 200, 400, … < target:      # batches of 200
        xml = efetch(WebEnv, query_key, retstart=start, retmax=200)
        articles += parse(xml)
    return sort(articles, key=int(pmid))        # stable order regardless of NCBI paging
```

```mermaid
flowchart TD
    Q["query"] --> ES["esearch<br/>usehistory=y · retmax=0"]
    ES --> H["count · WebEnv · query_key · translation"]
    H --> LOOP{"start &lt; target?"}
    LOOP -->|yes| EF["efetch<br/>retstart=start · retmax=200"]
    EF --> PA["parse XML → Article[]"]
    PA --> INC["start += 200"]
    INC --> LOOP
    LOOP -->|no| SORT["sort by int(pmid)"]
    SORT --> OUT[("results.csv · results.jsonl · protocol.json")]
    TH["self-throttle 10/3 req·s<br/>+ retry backoff on 429/5xx"] -. guards .-> EF

    classDef store fill:#eef2ff,stroke:#3b4cca,color:#111;
    classDef ext fill:#fff6e0,stroke:#b9770e,color:#111;
    class OUT store;
    class TH ext;
```

- **Self-throttle** to NCBI limits (10 req/s with key, else 3) and **retry with
  linear backoff** on 429/5xx. Result order is normalized by PMID so two runs
  produce identical files.

### F · Determinism scoring  (`test.py`)

For each `(model, query)` we run the pipeline `N` times and score agreement at
four levels. The score is the **modal frequency** — how often the single most
common output recurs:

```text
signature(run, level) = { map_raw:     json(map, order-preserved),
                          map_content: canonical(map),     # keys+lists sorted
                          query:       compiled query string,
                          query_hash:  sha256(query) }[level]

determinism(level) = max(count(sig)) / N        #  1.0 ⇔ every run identical
```

```mermaid
flowchart TD
    R["N runs of<br/>(model, query)"] --> SIG["signature per run per level<br/>map_raw · map_content · query · query_hash"]
    SIG --> CNT["Counter(signatures)"]
    CNT --> MODE["determinism =<br/>max(count) / N"]
    MODE --> D{"= 1.0 ?"}
    D -->|yes| DET["fully deterministic ✅"]
    D -->|no| VAR["distinct variants &gt; 1 ⚠️"]

    classDef good fill:#e0f2e9,stroke:#1e8449,color:#111;
    classDef bad fill:#ffe0e0,stroke:#c0392b,color:#111;
    class DET good;
    class VAR bad;
```

- `canonical()` sorts keys and lists so ordering-only churn (LLM reshuffling a
  synonym list) doesn't count as non-determinism — isolating *real* intent drift
  (`map_content`) from cosmetic drift (`map_raw`).

## Reproducibility in practice: measured & fixed

We built `test.py` to *measure* determinism instead of assuming it: for each
`(model, query)` it runs the whole pipeline N times and scores agreement (see
algorithm F). Running it exposed that the naive design was essentially
non-deterministic, and drove three fixes.

### What we found (10 runs/query, `temperature=0`)

| Configuration | Typical `query_hash` determinism | Why |
|---|---|---|
| **baseline** (LLM freetext → query) | ~**0.10** (8/10 models) | LLM emits a different synonym list every run; only gemini-3.5-flash was naturally stable |
| **`--strict`** (freetext from MeSH entry terms) | 0.10 → **0.4–0.94** | removes freetext noise; ceiling = how stable the model's *heading choice* is |
| **`--strict --cache`** (+ pin the map) | **1.00** across all models | heading set no longer varies → query is a pure function of pinned descriptors |

The residual gap after `--strict --cache` (opus 0.85, deepseek 0.74) turned out
to be a **bug**, not model variance — see mechanism 3.

### The three mechanisms

```mermaid
flowchart TD
    Q(["same question"]) --> LLM{{"🤖 LLM proposes<br/>concepts + MeSH headings"}}
    LLM --> P{"① pin?<br/>(--cache)"}
    P -->|"cache hit"| PIN[("pinned map<br/>data/map_cache/&lt;key&gt;.json")]
    P -->|"miss → call once"| PIN
    PIN --> RES["resolve headings → DUIs<br/>(deterministic index)"]
    RES --> S{"② strict?"}
    S -->|"yes"| ENT["free-text = exploded<br/>entry_terms of the DUIs<br/>(NOT raw LLM freetext)"]
    S -->|"no"| RAW["raw LLM freetext<br/>(run-varying)"]
    ENT --> TS["③ thread-safe index<br/>(per-thread sqlite conn)"]
    RAW --> TS
    TS --> HASH(["compile → sha256<br/>query_hash"])

    classDef good fill:#e0f2e9,stroke:#1e8449,color:#111;
    classDef bad fill:#ffe0e0,stroke:#c0392b,color:#111;
    classDef store fill:#eef2ff,stroke:#3b4cca,color:#111;
    class ENT,TS,HASH good;
    class RAW bad;
    class PIN store;
```

1. **Strict term derivation (`--strict`, and the app's "Strict MeSH synonyms"
   toggle).** Use the LLM *only* to pick MeSH headings; derive free-text from the
   deterministic index, so the query is a pure function of the chosen DUIs — no
   run-varying LLM prose enters it. **Descendant mapping is intelligent:** the
   `"Heading"[MeSH Terms]` tag already auto-explodes the whole descendant
   hierarchy for indexed articles *server-side, at zero query length*, so we do
   **not** enumerate every descendant's synonyms (that produced 3000-term,
   URL-breaking queries). Free-text is a **bounded** set — each heading's own
   lexical variants first, then descendant variants only while under a budget
   (`STRICT_MAX_TERMS`, default 60/concept) — enough to catch not-yet-indexed
   records while keeping the query compact (≈5 KB, usable in PubMed's own search
   box). Deterministic via sorted, stable truncation (`MeshIndex.strict_terms`).

2. **Map pinning (`--cache`).** A fresh LLM call is inherently non-deterministic
   in *which* headings it returns. Pinning runs the map **once** per
   `(model, question, domains)`, persists it, and reuses it — so the same
   question always yields the same headings. This mirrors the app's existing
   `protocol.json`: the reproducibility unit is the **saved mapping**, not a
   re-invoked LLM. Combined with `--strict` → `query_hash == 1.00`.

3. **Thread-safe MeSH index (bug fix).** `MeshIndex` shared one SQLite
   connection across threads; concurrent `entry_terms` lookups corrupted each
   other → occasionally a wrong term set (or `sqlite3.InterfaceError`). That was
   the last 0.85/0.74 wobble, and it affected the **production app** too (FastAPI
   serves concurrent requests). Fixed with per-thread connections
   (`threading.local`); 120 concurrent lookups now return one identical result.

### Model-agnostic: cloud + local open-source

The map step speaks the OpenAI chat-completions shape, so the client routes by
model-id prefix — no code change to swap providers:

```mermaid
flowchart LR
    M["model id"] --> R{"prefix?"}
    R -->|"ollama/…"| O["🖥️ local Ollama<br/>:11434 · no key"]
    R -->|"hf/…"| H["🤗 HF Inference router<br/>HF_TOKEN"]
    R -->|"anything else"| OR["☁️ OpenRouter<br/>OPENROUTER_API_KEY"]

    classDef ext fill:#fff6e0,stroke:#b9770e,color:#111;
    class O,H,OR ext;
```

Because `--strict` sources terms from MeSH, even a tiny local model works: a 1B
`ollama/llama3.2:1b` writes poor free-text (full sentences), but that is
discarded — it only needs to *name* MeSH-resolvable concepts, and the
deterministic layer supplies exhaustive, reproducible synonyms.

## Legend

| Color | Meaning |
|-------|---------|
| 🔴 red | non-deterministic (LLM proposal only) |
| 🟢 green | deterministic backend logic |
| 🔵 blue | data stores / artifacts |
| 🟠 orange | external service (NCBI) |
| 🟣 purple | user / UI |
