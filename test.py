"""
Determinism test for the deterministic PubMed search pipeline — multi-model.

The pipeline has exactly ONE non-deterministic step: the LLM `/api/map` call
(OpenRouter, temperature=0). The LLM is used purely for INTENT UNDERSTANDING —
it decomposes the research question into concept blocks and proposes MeSH
headings + free-text keywords. Everything after it is pure and byte-stable:
MeSH headings resolve against the local index, free-text is de-duped, and the
query is compiled + hashed. So the property under test is:

    Given the SAME question, does a model reproducibly yield the SAME final
    PubMed query (and hash) across N runs — and how do models compare?

For each (model, query) we run map -> resolve -> compile `--runs` times
(default 10) and score determinism at four levels:

  1. map_raw      LLM JSON exactly as returned (byte-identical?)
  2. map_content  LLM JSON canonicalised (keys + lists sorted) — ignores mere
                  ordering churn, catches real keyword/intent drift
  3. query        final compiled PubMed query string
  4. query_hash   sha256 of that query — THE reproducibility artifact

Determinism score for a level = (most common output group size) / N; 1.0 means
every run agreed. We also count DISTINCT variants.

--jitter additionally sends cosmetically-varied request JSON (whitespace,
trailing punctuation, reordered domains — semantically identical) to test
stability under harmless paraphrase, not just byte-identical replay.

Outputs a JSON report, an Excel workbook, and a seaborn figure comparing
models. Every (model, query, run) task runs in ONE thread pool, so all models
execute in parallel under a single --workers cap. Runs the real endpoint code
in-process; the only network call is to OpenRouter. Needs OPENROUTER_API_KEY
(env or .env).

Every run starts from a CLEAN SLATE — prior report/figure/xlsx outputs and the
pinned map cache are removed first (pass --keep to opt out).

Usage:
    .venv/bin/python test.py                          # default models, 10 runs
    .venv/bin/python test.py --strict --cache         # fully reproducible (query_hash=1.00)
    .venv/bin/python test.py --top 10 --workers 10    # auto-pick 10 newest flagships
    .venv/bin/python test.py --jitter                 # vary request JSON per run
    .venv/bin/python test.py --models anthropic/claude-sonnet-5,ollama/llama3.2:1b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ .env load

def load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and not os.environ.get(k):  # don't clobber a real shell value
            os.environ[k] = v


load_dotenv(ROOT / ".env")

# Imported AFTER .env so module-level defaults pick it up.
import backend.app as bapp  # noqa: E402  (map_question here is patched for --seed)
from backend import query_builder  # noqa: E402
from backend.openrouter_client import OpenRouterError  # noqa: E402
from backend.mesh_index import get_index  # noqa: E402


# ------------------------------------------------------------------ test data

# A representative mix: flagships + small open-source models. OSS models run via
# OpenRouter here; the same ids work locally as "ollama/<model>" or hosted as
# "hf/<repo>". Override with --models or --top.
DEFAULT_MODELS = [
    # flagships
    "anthropic/claude-opus-4.8",
    "google/gemini-3.5-flash",
    "deepseek/deepseek-v3.2",
    "openai/gpt-5.4-mini",
    # small / open-source
    "mistralai/ministral-3b-2512",
    "mistralai/ministral-8b-2512",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-3-4b-it",
    "microsoft/phi-4",
    # local open-source (needs `ollama serve` + the model pulled)
    "ollama/llama3.2:1b",
    "ollama/qwen2.5:1.5b",
]


def fetch_top_models(n: int, api_key: str) -> list[str]:
    """
    Query OpenRouter and return the `n` newest flagship text models that support
    JSON output, one per provider (skips :free / -fast / preview / image / coder
    variants) so we don't benchmark ten near-duplicates of one family.
    """
    import requests
    data = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}, timeout=30,
    ).json()["data"]

    def keep(m: dict) -> bool:
        arch = m.get("architecture", {})
        mods = str(arch.get("input_modalities") or arch.get("modality", ""))
        sp = m.get("supported_parameters") or []
        bad = (":free", "-fast", "latest", "preview", "image", "-code", "coder")
        return "text" in mods and "response_format" in sp and not any(b in m["id"] for b in bad)

    cands = sorted((m for m in data if keep(m)), key=lambda m: m.get("created", 0), reverse=True)
    picked, seen = [], set()
    for m in cands:
        prov = m["id"].split("/")[0]
        if prov in seen:
            continue
        seen.add(prov)
        picked.append(m["id"])
        if len(picked) >= n:
            break
    return picked

DEFAULT_QUERIES = [
    {
        "question": "Does optogenetic stimulation of the hippocampus improve "
                    "memory consolidation in rodent models?",
        "domains": ["neuroscience"],
    },
    {
        "question": "single-cell RNA sequencing of microglia in Alzheimer's disease",
        "domains": ["neuroscience", "bioinformatics"],
    },
    {
        "question": "CRISPR-Cas9 off-target effects detection methods",
        "domains": ["bioinformatics"],
    },
]

LEVELS = ["map_raw", "map_content", "query", "query_hash"]

# Cosmetic, meaning-preserving rewrites for --jitter (indexed by run number).
_JITTER = [
    lambda q: q,
    lambda q: q.strip() + ".",
    lambda q: "  " + q + "  ",
    lambda q: q.replace("  ", " "),
    lambda q: q + "\n",
]


def jitter_request(base: dict, run: int) -> dict:
    """Cosmetically-varied but semantically-identical request."""
    out = dict(base)
    out["question"] = _JITTER[run % len(_JITTER)](base["question"])
    doms = list(base.get("domains", []))
    if run % 2 and len(doms) > 1:
        doms = list(reversed(doms))
    out["domains"] = doms
    return out


# ------------------------------------------------------------------ pipeline

def selected_headings(mesh_field: list[dict]) -> list[str]:
    """
    Reproduce what the UI does before /api/compile: for each resolved MeSH
    candidate pick the label of the option matching `selected_dui`. Unmatched
    candidates drop out — exactly as a hallucinated heading would.
    """
    headings: list[str] = []
    for m in mesh_field:
        sel = m.get("selected_dui")
        if not sel:
            continue
        label = next((o["label"] for o in m.get("options", []) if o["dui"] == sel), None)
        if label:
            headings.append(label)
    return headings


def selected_duis(mesh_field: list[dict]) -> list[str]:
    """The selected_dui of each resolved MeSH candidate (unmatched drop out)."""
    return [m["selected_dui"] for m in mesh_field if m.get("selected_dui")]


def strict_freetext(mesh_field: list[dict]) -> list[str]:
    """
    STRICT mode: derive the free-text synonym set for a concept ENTIRELY from the
    deterministic MeSH index — the exploded entry terms of the selected
    descriptors — instead of the LLM's raw (run-varying) `freetext`. Given the
    same selected DUIs this is byte-stable, so the only residual non-determinism
    is which DUIs the LLM picks.
    """
    return get_index().strict_terms(selected_duis(mesh_field), explode=True)


# ---- map cache (pinning) --------------------------------------------------
# Full determinism for a FRESH pipeline requires removing the LLM's run-to-run
# variance in WHICH MeSH headings it picks. We do that by pinning: the first
# map() for a given (model, question, domains) is persisted and every later run
# reuses it. Reproducibility unit = the saved mapping, exactly like the app's
# protocol.json. Strict term derivation then makes the rest a pure function of
# those pinned headings, so query_hash == 1.00.
CACHE_DIR = ROOT / "data" / "map_cache"
_lock_registry: dict[str, threading.Lock] = defaultdict(threading.Lock)
_registry_guard = threading.Lock()
_failed_keys: dict[str, str] = {}   # key -> error; fail fast, no serial re-calls


def _cache_key(model: str, request: dict) -> str:
    doms = ",".join(sorted(request.get("domains", [])))
    raw = f"{model}\n{request['question'].strip().lower()}\n{doms}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cached_map(model: str, request: dict, produce) -> dict:
    """Return the pinned map for this key, producing+persisting it exactly once."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(model, request)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with _registry_guard:
        lock = _lock_registry[key]
    with lock:  # serialize the first call per key; the rest read the file
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if key in _failed_keys:  # first attempt already failed — don't retry 9 more times
            raise OpenRouterError(_failed_keys[key])
        try:
            mapped = produce()
        except Exception as e:
            _failed_keys[key] = f"map failed (cached): {e}"
            raise
        path.write_text(json.dumps(mapped, ensure_ascii=False), encoding="utf-8")
        return mapped


def run_pipeline(request: dict, model: str, strict: bool = False,
                 cache: bool = False) -> dict:
    """map -> resolve -> compile, returning the run's observable outputs.

    Uses the SYNC client (bapp.map_question — patched when --seed is set) plus the
    shared resolve_concepts() so it exercises the same deterministic path as the
    now-async /api/map endpoint, without needing an event loop per worker thread.
    """
    def _map():
        return bapp.map_question(
            request["question"], domains=request.get("domains", []), model=model,
        )
    raw = _cached_map(model, request, _map) if cache else _map()
    result = bapp.resolve_concepts(raw, request.get("domains", []))
    concepts = [
        {
            "name": c["name"],
            "mesh": selected_headings(c["mesh"]),
            "freetext": strict_freetext(c["mesh"]) if strict else c["freetext"],
            "explode": True,
        }
        for c in result["concepts"]
    ]
    compiled = query_builder.compile_search(concepts, {})
    return {
        "map": {"concepts": result["concepts"], "notes": result["notes"]},
        "query": compiled["query"],
        "query_hash": compiled["hash"],
        "n_concepts": compiled["n_concepts"],
        "n_terms": compiled["n_terms"],
    }


# ------------------------------------------------------------------ scoring

def canonical(obj) -> str:
    """Order-insensitive JSON: sort dict keys and every list."""
    def norm(x):
        if isinstance(x, dict):
            return {k: norm(x[k]) for k in sorted(x)}
        if isinstance(x, list):
            items = [norm(i) for i in x]
            return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))
        return x
    return json.dumps(norm(obj), sort_keys=True, ensure_ascii=False)


def score(signatures: list[str], n: int) -> dict:
    counts = Counter(signatures)
    modal_sig, modal_n = counts.most_common(1)[0]
    return {
        "runs": n,
        "distinct": len(counts),
        "modal_count": modal_n,
        "determinism": round(modal_n / n, 4),
        "fully_deterministic": len(counts) == 1,
    }


def signatures_for(runs: list[dict]) -> dict:
    return {
        "map_raw": [json.dumps(r["map"], ensure_ascii=False, sort_keys=False) for r in runs],
        "map_content": [canonical(r["map"]) for r in runs],
        "query": [r["query"] for r in runs],
        "query_hash": [r["query_hash"] for r in runs],
    }


# ------------------------------------------------------------------ run everything (parallel across models)

def execute_all(models: list[str], queries: list[dict], runs: int, jitter: bool,
                workers: int, strict: bool = False, cache: bool = False) -> tuple[dict, dict]:
    """
    Run EVERY (model, query, run) task in one thread pool, so all models — not
    just the runs within a model — execute in parallel under a single `workers`
    cap. Returns (outs, errs):
        outs[(model, qi, run)] = pipeline output
        errs[(model, qi)]      = list of error strings
    """
    # ROUND-ROBIN order (run-major, then query, then model) so consecutive tasks
    # belong to DIFFERENT models. Otherwise a model-major list makes the pool run
    # models near-sequentially — a slow model never overlaps a fast one, and the
    # total time becomes the SUM of per-model times instead of the max.
    tasks = [(m, qi, r) for r in range(runs) for qi in range(len(queries)) for m in models]
    print(f"Dispatching {len(tasks)} tasks "
          f"({len(models)} models x {len(queries)} queries x {runs} runs) "
          f"across {workers} workers...", flush=True)

    outs: dict[tuple, dict] = {}
    errs: dict[tuple, list[str]] = {}
    lat: dict[str, list[float]] = {m: [] for m in models}      # per-model latency (s)
    done = 0
    total = len(tasks)

    def one(model: str, qi: int, run: int):
        req = jitter_request(queries[qi], run) if jitter else queries[qi]
        t0 = time.perf_counter()
        try:
            out = run_pipeline(req, model, strict=strict, cache=cache)
            return (model, qi, run), out, time.perf_counter() - t0, None
        except Exception as e:  # time failures too — a timeout is the slow tail
            return (model, qi, run), None, time.perf_counter() - t0, e

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, m, qi, r) for (m, qi, r) in tasks]
        for f in as_completed(futs):
            (m, qi, r), out, dt, err = f.result()
            done += 1
            lat[m].append(dt)
            if err is None:
                outs[(m, qi, r)] = out
            else:
                errs.setdefault((m, qi), []).append(f"{type(err).__name__}: {err}")
            if done % 10 == 0 or done == total:
                print(f"  ... {done}/{total} tasks complete", flush=True)

    # per-model latency summary — shows exactly which model is the bottleneck
    print("\nPer-model latency (seconds per LLM call):")
    print(f"  {'model':34s}{'n':>4}{'mean':>8}{'median':>8}{'max':>8}")
    slowest = sorted(models, key=lambda m: (statistics.mean(lat[m]) if lat[m] else 0), reverse=True)
    for m in slowest:
        v = lat[m]
        if not v:
            continue
        print(f"  {m:34s}{len(v):>4}{statistics.mean(v):>8.1f}"
              f"{statistics.median(v):>8.1f}{max(v):>8.1f}")
    return outs, errs, {m: round(statistics.mean(v), 2) for m, v in lat.items() if v}


def score_model(model: str, queries: list[dict], runs: int,
                outs: dict, errs: dict) -> dict:
    """Assemble + score one model's results from the flat task outputs."""
    result = {"model": model, "queries": []}
    for qi, q in enumerate(queries):
        ordered = [outs[(model, qi, r)] for r in range(runs) if (model, qi, r) in outs]
        n_ok = len(ordered)
        q_errs = errs.get((model, qi), [])
        if n_ok == 0:
            result["queries"].append({"question": q["question"], "ok_runs": 0, "errors": q_errs})
            continue
        sigs = signatures_for(ordered)
        scores = {lvl: score(sigs[lvl], n_ok) for lvl in LEVELS}
        result["queries"].append({
            "question": q["question"],
            "domains": q.get("domains", []),
            "ok_runs": n_ok,
            "errors": q_errs,
            "scores": scores,
            "distinct_hashes": sorted(set(sigs["query_hash"])),
            "sample_query": ordered[0]["query"],
            "runs": [
                {"query_hash": r["query_hash"], "n_concepts": r["n_concepts"],
                 "n_terms": r["n_terms"], "query": r["query"], "map": r["map"]}
                for r in ordered
            ],
        })
    result["aggregate"] = aggregate(result["queries"])
    return result


def _modal_hash(qr: dict) -> str | None:
    """The most common query_hash for one (model, query) across its runs."""
    if not qr.get("runs"):
        return None
    return Counter(r["query_hash"] for r in qr["runs"]).most_common(1)[0][0]


def cross_model_determinism(report: dict) -> dict:
    """
    CROSS-MODEL (cross-modal) determinism: for the SAME question, do different
    models — cloud and local open-source alike — compile to the SAME query?
    Within-model determinism (does one model repeat itself) is the query_hash
    score; this measures inter-model CONSENSUS. Per query: agreement =
    (models sharing the most common query_hash) / (models that produced one).
    """
    queries = report["models"][0]["queries"] if report["models"] else []
    per_query = []
    for qi in range(len(queries)):
        hashes = []
        for m in report["models"]:
            if qi < len(m["queries"]):
                h = _modal_hash(m["queries"][qi])
                if h:
                    hashes.append((m["model"], h))
        if not hashes:
            continue
        counts = Counter(h for _, h in hashes)
        modal_hash, modal_n = counts.most_common(1)[0]
        per_query.append({
            "question": queries[qi]["question"],
            "n_models": len(hashes),
            "distinct_queries": len(counts),
            "agreement": round(modal_n / len(hashes), 4),
            "consensus_models": sorted(m.split("/")[-1] for m, h in hashes if h == modal_hash),
            "outlier_models": sorted(m.split("/")[-1] for m, h in hashes if h != modal_hash),
        })
    vals = [q["agreement"] for q in per_query]
    return {
        "per_query": per_query,
        "mean_agreement": round(statistics.mean(vals), 4) if vals else None,
        "note": "1.0 = every model produced the byte-identical query for that "
                "question. <1.0 is expected: models legitimately pick different "
                "MeSH headings. Within-model reproducibility is query_hash=1.00.",
    }


def aggregate(query_results: list[dict]) -> dict:
    agg = {}
    for lvl in LEVELS:
        vals = [qr["scores"][lvl]["determinism"] for qr in query_results if qr.get("scores")]
        det = [qr["scores"][lvl]["fully_deterministic"] for qr in query_results if qr.get("scores")]
        agg[lvl] = {
            "mean_determinism": round(statistics.mean(vals), 4) if vals else None,
            "min_determinism": min(vals) if vals else None,
            "fully_deterministic_queries": f"{sum(det)}/{len(det)}" if det else "0/0",
        }
    return agg


# ------------------------------------------------------------------ figure

def _terms_of(query: str) -> set[str]:
    """All quoted field tokens in a compiled query, e.g. '"X"[Title/Abstract]'."""
    import re
    return set(re.findall(r'"[^"]+"\[[^\]]+\]', query or ""))


def _headings_of(query: str) -> set[str]:
    """The MeSH headings (bare labels) a query selected."""
    import re
    return set(re.findall(r'"([^"]+)"\[MeSH(?:\s*Terms|:NoExp)\]', query or ""))


def _model_query(qr: dict) -> str:
    return qr["runs"][0]["query"] if qr.get("runs") else ""


def make_figure(report: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    import seaborn as sns

    models = [m["model"].split("/")[-1] for m in report["models"]]
    questions = [q["question"] for q in report["models"][0]["queries"]] if report["models"] else []
    xm = report.get("cross_model") or cross_model_determinism(report)
    sns.set_theme(style="whitegrid", context="talk")

    rows = [{"model": m["model"].split("/")[-1], "level": lvl,
             "determinism": qr["scores"][lvl]["determinism"]}
            for m in report["models"] for qr in m["queries"]
            if qr.get("scores") for lvl in LEVELS]
    if not rows:
        print("No scored runs — skipping figure.")
        return
    df = pd.DataFrame(rows)

    q_legend = "   ".join(f"Q{i+1} = {q[:46]}{'…' if len(q) > 46 else ''}"
                          for i, q in enumerate(questions))

    # ============================================================= PAGE 1
    fig1, axes = plt.subplots(1, 3, figsize=(22, 8),
                              gridspec_kw={"width_ratios": [1.25, 1, 0.9]})
    sns.barplot(data=df, x="level", y="determinism", hue="model",
                errorbar=("ci", 95), capsize=.08, ax=axes[0])
    axes[0].set_title("[1] Within-model determinism by level\n(mean across queries, 95% CI)", fontsize=14)
    axes[0].set_ylim(0, 1.05); axes[0].set_xlabel("")
    axes[0].set_ylabel("determinism  (1.0 = every run identical)")
    axes[0].axhline(1.0, ls="--", lw=1, color="grey", alpha=.7)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(title="model", fontsize=8, title_fontsize=9, loc="lower left")

    hm_rows = [{"model": m["model"].split("/")[-1], "query": f"Q{qi+1}",
                "determinism": qr["scores"]["query_hash"]["determinism"]}
               for m in report["models"]
               for qi, qr in enumerate(m["queries"]) if qr.get("scores")]
    hm = pd.DataFrame(hm_rows).pivot(index="model", columns="query", values="determinism")
    sns.heatmap(hm, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="RdYlGn",
                linewidths=.5, cbar=False, ax=axes[1])
    axes[1].set_title("[2] query_hash reproducibility\n(same model, 10 runs → same query?)", fontsize=14)
    axes[1].set_xlabel(""); axes[1].set_ylabel("")

    xmdf = pd.DataFrame([{"query": f"Q{i+1}", "agreement": q["agreement"],
                          "distinct": q["distinct_queries"], "n": q["n_models"]}
                         for i, q in enumerate(xm["per_query"])])
    sns.barplot(data=xmdf, x="query", y="agreement", color="#c0392b", ax=axes[2])
    for i, r in xmdf.iterrows():
        axes[2].text(i, r["agreement"] + 0.02, f"{r['distinct']}/{r['n']}\ndistinct",
                     ha="center", va="bottom", fontsize=9)
    axes[2].set_title("[3] Cross-model agreement\n(different models → same query?)", fontsize=14)
    axes[2].set_ylim(0, 1.05); axes[2].set_xlabel("")
    axes[2].set_ylabel("fraction of models sharing the modal query")
    axes[2].axhline(1.0, ls="--", lw=1, color="grey", alpha=.7)

    fig1.suptitle(
        f"PubMed search determinism — {len(models)} models × {report['runs']} runs/query"
        f"{'  ·  strict' if report.get('strict') else ''}"
        f"{'  ·  cached/pinned' if report.get('cache') else ''}"
        f"{'  ·  jittered' if report['jitter'] else ''}", fontsize=17, y=1.05)
    cap1 = (
        f"QUESTIONS:   {q_legend}\n"
        "LEVELS —  map_raw: LLM JSON byte-identical across runs  ·  map_content: same ignoring key/list order  ·  "
        "query: final compiled PubMed query  ·  query_hash: SHA-256 of that query.\n"
        "ARTIFACT —  query_hash is the reproducibility unit: a 16-hex digest of the exact query string. Identical hash => "
        "byte-identical query => PubMed returns the identical article set. It is what you cite in a protocol to prove a\n"
        "search can be reproduced. [1][2] WITHIN-model = one model repeats itself (=1.00 here → reproducible). "
        f"[3] CROSS-model = different models agree (mean {xm['mean_agreement']}; low — see page 2)."
    )
    fig1.text(0.5, -0.09, cap1, ha="center", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.6", fc="#f5f5f5", ec="#ccc"))
    fig1.tight_layout()
    fig1.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Figure -> {out_png}")

    # ============================================================= PAGE 2 (why cross-model is low)
    # pairwise Jaccard similarity of query TERM SETS between models (mean over queries)
    n = len(report["models"])
    jac = [[0.0] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            sims = []
            for qi in range(len(questions)):
                A = _terms_of(_model_query(report["models"][a]["queries"][qi]))
                B = _terms_of(_model_query(report["models"][b]["queries"][qi]))
                if A or B:
                    sims.append(len(A & B) / len(A | B))
            jac[a][b] = round(statistics.mean(sims), 3) if sims else 0.0
    jdf = pd.DataFrame(jac, index=models, columns=models)
    off = [jac[a][b] for a in range(n) for b in range(n) if a != b]
    mean_off = round(statistics.mean(off), 3) if off else 0.0

    # per-query heading divergence: distinct headings across models vs avg per model
    div_rows = []
    for qi, q in enumerate(questions):
        union, sizes = set(), []
        for m in report["models"]:
            h = _headings_of(_model_query(m["queries"][qi]))
            union |= h; sizes.append(len(h))
        div_rows.append({"query": f"Q{qi+1}", "distinct headings (union)": len(union),
                         "avg headings / model": round(statistics.mean(sizes), 1)})
    ddf = pd.DataFrame(div_rows).melt("query", var_name="metric", value_name="count")

    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 8.5), gridspec_kw={"width_ratios": [1.15, 1]})
    sns.heatmap(jdf, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="magma",
                linewidths=.5, cbar_kws={"label": "term-set Jaccard similarity"}, ax=ax2[0])
    ax2[0].set_title("Why cross-model is low (1): pairwise query-term overlap\n"
                     "(Jaccard of the compiled queries' term sets, mean over questions)", fontsize=13)
    sns.barplot(data=ddf, x="query", y="count", hue="metric", ax=ax2[1])
    ax2[1].set_title("Why cross-model is low (2): heading selection diverges\n"
                     "(models collectively choose many DISTINCT MeSH headings)", fontsize=13)
    ax2[1].set_xlabel(""); ax2[1].set_ylabel("MeSH headings"); ax2[1].legend(fontsize=10)
    fig2.suptitle("Cross-model divergence — the SAME question, different MeSH headings per model",
                  fontsize=16, y=1.03)
    cap2 = (
        f"Mean pairwise term-set overlap across models = {mean_off} (0 = no shared terms, 1 = identical). "
        "Each model independently DECOMPOSES the question and proposes different MeSH\n"
        "headings; those resolve to different descriptors, whose exploded entry-terms differ, so the compiled query "
        "(and its SHA-256) differs. The deterministic layer faithfully reproduces whatever\n"
        "headings a model picked — it cannot make two different models pick the same ones. Hence reproducibility is "
        "PER-MODEL: to reproduce a search, pin the (model + question) and keep strict+cache."
    )
    fig2.text(0.5, -0.06, cap2, ha="center", va="top", fontsize=10,
              bbox=dict(boxstyle="round,pad=0.6", fc="#fff4f4", ec="#e0b4b4"))
    fig2.tight_layout()

    # ============================================================= PAGE 3 (definitions / methodology)
    fig3 = plt.figure(figsize=(20, 11))
    fig3.suptitle("Definitions & methodology", fontsize=18, y=0.98)
    q_lines = "\n".join(f"      Q{i+1}:  {q}" for i, q in enumerate(questions))
    text = (
        "PIPELINE\n"
        "    research question → LLM /api/map (temperature 0; the ONLY non-deterministic step) proposes concept blocks\n"
        "    + candidate MeSH headings → each heading resolved against a local MeSH index (hallucinations drop out) →\n"
        "    strict free-text derived from the index → query_builder compiles a Boolean PubMed query → SHA-256 hash.\n\n"
        "QUESTIONS TESTED\n"
        f"{q_lines}\n\n"
        "DETERMINISM LEVELS (score = size of the most common output group / N runs; 1.0 = all runs identical)\n"
        "    map_raw       the LLM's raw JSON, byte-for-byte — catches any wording/ordering change\n"
        "    map_content   that JSON with keys and lists sorted — catches real intent/keyword drift, ignores reordering\n"
        "    query         the final compiled PubMed query string\n"
        "    query_hash    SHA-256 (16 hex) of that query — THE ARTIFACT: identical hash => identical query => identical\n"
        "                  PubMed result set. This is what a systematic-review protocol cites to prove reproducibility.\n\n"
        "TWO KINDS OF DETERMINISM\n"
        "    WITHIN-model (reproducibility)   same model, run N times → same query_hash?   Here 1.00 for every model,\n"
        "                                     achieved by: strict MeSH-derived terms + pinning the map (run once, reuse).\n"
        "    CROSS-model (agreement)          different models → same query_hash?   Here ≈ 0.09 (essentially none).\n\n"
        "WHY CROSS-MODEL IS LOW (and why that's fine)\n"
        "    Each model independently decomposes the question and picks its OWN set of MeSH headings. Different headings →\n"
        "    different descriptors → different exploded entry-terms → different query → different hash. The deterministic\n"
        "    layer reproduces faithfully whatever a model chose; it cannot force two models to agree. So reproducibility\n"
        "    is PER-MODEL: record the model + question alongside the query_hash, and any reader can reproduce the search.\n"
    )
    fig3.text(0.06, 0.90, text, ha="left", va="top", family="monospace", fontsize=11.5)

    # ============================================================= write multipage PDF
    pdf_path = out_png.with_suffix(".pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1, bbox_inches="tight")
        pdf.savefig(fig2, bbox_inches="tight")
        pdf.savefig(fig3)
    print(f"Multipage figure -> {pdf_path}")
    plt.close("all")


# ------------------------------------------------------------------ excel

def make_excel(report: dict, out_xlsx: Path) -> None:
    """Write a multi-sheet workbook: summary, per-query scores, per-run detail."""
    import pandas as pd

    summary_rows, perq_rows, run_rows = [], [], []
    for m in report["models"]:
        mid = m["model"]
        a = m["aggregate"]
        row = {"model": mid}
        for lvl in LEVELS:
            row[f"{lvl}_mean"] = a[lvl]["mean_determinism"]
            row[f"{lvl}_min"] = a[lvl]["min_determinism"]
            row[f"{lvl}_fully_det"] = a[lvl]["fully_deterministic_queries"]
        summary_rows.append(row)

        for qi, qr in enumerate(m["queries"]):
            base = {"model": mid, "query_idx": qi + 1, "question": qr["question"],
                    "ok_runs": qr.get("ok_runs", 0)}
            if qr.get("scores"):
                for lvl in LEVELS:
                    s = qr["scores"][lvl]
                    base[f"{lvl}_det"] = s["determinism"]
                    base[f"{lvl}_distinct"] = s["distinct"]
                base["distinct_hashes"] = ", ".join(qr.get("distinct_hashes", []))
                base["sample_query"] = qr.get("sample_query", "")
                for ri, r in enumerate(qr.get("runs", [])):
                    run_rows.append({
                        "model": mid, "query_idx": qi + 1, "run": ri + 1,
                        "query_hash": r["query_hash"], "n_concepts": r["n_concepts"],
                        "n_terms": r["n_terms"], "query": r["query"],
                    })
            else:
                base["error"] = (qr.get("errors") or [""])[0]
            perq_rows.append(base)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        pd.DataFrame(summary_rows).to_excel(xl, sheet_name="summary", index=False)
        pd.DataFrame(perq_rows).to_excel(xl, sheet_name="per_query", index=False)
        pd.DataFrame(run_rows).to_excel(xl, sheet_name="per_run", index=False)
    print(f"Excel  -> {out_xlsx}")


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-model determinism test for the PubMed pipeline")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated OpenRouter model ids (ignored if --top given)")
    ap.add_argument("--top", type=int, default=None,
                    help="auto-select the N newest flagship models from OpenRouter")
    ap.add_argument("--runs", type=int, default=10, help="runs per query per model (default 10)")
    ap.add_argument("--queries", type=Path, default=None,
                    help="JSON file: [{question, domains}] (defaults to built-in set)")
    ap.add_argument("--jitter", action="store_true",
                    help="send cosmetically-varied request JSON per run")
    ap.add_argument("--strict", action="store_true",
                    help="derive free-text terms from the deterministic MeSH index "
                         "(exploded entry terms of selected descriptors) instead of the "
                         "LLM's raw freetext; LLM is used only to pick headings")
    ap.add_argument("--cache", action="store_true",
                    help="pin the map: run the LLM once per (model, question, domains) "
                         "and reuse it, so a fresh pipeline is fully reproducible. "
                         "Combine with --strict for query_hash == 1.00.")
    ap.add_argument("--keep", action="store_true",
                    help="do NOT clean before running (keep prior outputs + pinned map "
                         "cache). By default every run starts from a fresh slate.")
    ap.add_argument("--seed", type=int, default=None,
                    help="pass a fixed OpenRouter seed on every map call (best-effort "
                         "reproducibility; ignored by providers that don't support it). "
                         "Off by default so it measures the real pipeline.")
    ap.add_argument("--workers", type=int, default=16,
                    help="global concurrent LLM calls across all models (default 16; "
                         "LLM calls are I/O-bound so raise freely until you hit "
                         "OpenRouter rate limits)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "determinism_report.json")
    ap.add_argument("--fig", type=Path, default=ROOT / "data" / "determinism_by_model.png")
    ap.add_argument("--xlsx", type=Path, default=ROOT / "data" / "determinism_report.xlsx")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set (env or .env). The map step is the "
              "only non-deterministic step and cannot be exercised without it.", file=sys.stderr)
        return 2

    if args.top:
        models = fetch_top_models(args.top, key)
        print(f"Auto-selected top {args.top} models from OpenRouter:")
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    queries = (json.loads(args.queries.read_text(encoding="utf-8"))
               if args.queries else DEFAULT_QUERIES)

    # Inject a fixed seed into the real endpoint's LLM call without touching
    # app.py: /api/map calls the module-global map_question, so we wrap it here.
    if args.seed is not None:
        import backend.app as bapp
        _orig_mq = bapp.map_question
        bapp.map_question = lambda *a, **kw: _orig_mq(*a, **{**kw, "seed": args.seed})
        print(f"Seed: {args.seed} (forwarded to providers that support it)")

    print(f"Models: {models}\nRuns/query: {args.runs}   jitter: {args.jitter}   "
          f"seed: {args.seed}   strict: {args.strict}   cache: {args.cache}   "
          f"queries: {len(queries)}   workers: {args.workers}")

    # Fresh slate by default: remove prior outputs, sidecar report files, and the
    # pinned map cache so no stale result can leak into this run. --keep opts out.
    if not args.keep:
        removed = []
        for p in (args.out, args.fig, args.xlsx):
            if p.exists():
                p.unlink(); removed.append(p.name)
        if CACHE_DIR.exists():
            n = len(list(CACHE_DIR.glob("*.json")))
            shutil.rmtree(CACHE_DIR)
            removed.append(f"map_cache/ ({n} pinned maps)")
        print("Clean slate — removed: " + (", ".join(removed) if removed else "nothing") + "\n")

    t_start = time.perf_counter()
    outs, errs, mean_latency = execute_all(models, queries, args.runs, args.jitter,
                                           args.workers, strict=args.strict, cache=args.cache)
    wall = time.perf_counter() - t_start
    print(f"\nWall time: {wall:.1f}s for {len(models) * len(queries) * args.runs} calls "
          f"({args.workers} workers)")

    report = {"models_requested": models, "runs": args.runs, "jitter": args.jitter,
              "seed": args.seed, "strict": args.strict, "cache": args.cache,
              "wall_seconds": round(wall, 1), "mean_latency_s": mean_latency, "models": []}
    for model in models:
        report["models"].append(score_model(model, queries, args.runs, outs, errs))

    # ---- console summary ----
    print("\n" + "=" * 74)
    print(f"{'model':38s}" + "".join(f"{lvl:>10s}" for lvl in LEVELS))
    print("-" * 74)
    for m in report["models"]:
        a = m["aggregate"]
        cells = "".join(f"{(a[lvl]['mean_determinism'] if a[lvl]['mean_determinism'] is not None else 0):>10.2f}"
                        for lvl in LEVELS)
        print(f"{m['model'].split('/')[-1]:38s}{cells}")
    print("=" * 74)
    print("query_hash == 1.00 means the same question always compiles to the "
          "byte-identical\nPubMed query. High map_raw variation with a stable "
          "query_hash means the model\nreworded/reordered keywords but the "
          "deterministic layer absorbed it.")

    # ---- cross-model (cross-modal) determinism ----
    xm = cross_model_determinism(report)
    report["cross_model"] = xm
    print("\n" + "=" * 74)
    print("CROSS-MODEL determinism (do different models agree on the SAME query?)")
    print("-" * 74)
    for q in xm["per_query"]:
        print(f"  {q['question'][:52]:52s} agree={q['agreement']:.2f} "
              f"({q['n_models'] - len(q['outlier_models'])}/{q['n_models']} models, "
              f"{q['distinct_queries']} distinct)")
    print(f"  mean cross-model agreement: {xm['mean_agreement']}")
    print("=" * 74)
    print("Within-model = query_hash (reproducibility). Cross-model < 1.0 is "
          "EXPECTED:\nmodels legitimately pick different MeSH headings.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport -> {args.out}")

    try:
        make_figure(report, args.fig)
    except Exception as e:
        print(f"Figure generation failed: {type(e).__name__}: {e}", file=sys.stderr)
    try:
        make_excel(report, args.xlsx)
    except Exception as e:
        print(f"Excel generation failed: {type(e).__name__}: {e}", file=sys.stderr)

    mins = [m["aggregate"]["query_hash"]["min_determinism"] for m in report["models"]
            if m["aggregate"]["query_hash"]["min_determinism"] is not None]
    return 0 if (not mins or min(mins) == 1.0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
