"""
FastAPI backend for the deterministic PubMed literature-search tool.

Pipeline:
  1. /api/map      research question -> LLM-proposed concepts, MeSH resolved
                   against the local index (deterministic) + domain synonyms
  2. /api/expand   preview MeSH explosion + entry terms for one descriptor
  3. /api/compile  concepts + inclusion/exclusion filters -> reproducible query
  4. /api/count    esearch only: how many hits + PubMed's query translation
  5. /api/search   exhaustive efetch of every record; saves CSV/JSONL/protocol
  6. /api/download serve a saved export file

Run:  uvicorn backend.app:app --reload  (from project root)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import canonical, pipeline, query_builder
from .domain_vocab import get_vocab
from .mesh_index import get_index
from .openrouter_client import DEFAULT_PROMPT_VERSION, OpenRouterError
from .pubmed import PubMed, to_csv, to_jsonl

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SEARCH_DIR = ROOT / "data" / "searches"
SEARCH_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Deterministic PubMed Search")


# ---------------------------------------------------------------- models

class MapReq(BaseModel):
    question: str
    domains: list[str] = []
    model: str | None = None
    extra_context: str = ""
    api_key: str | None = None
    mode: str = "hybrid"                 # llm | hybrid | closure | mesh_only
    prompt_version: str | None = None    # llm mode only: v1 | v2
    facet_runs: int = 3                  # closure mode only: k for self-consistency voting


class ExpandReq(BaseModel):
    dui: str
    explode: bool = True


class CompileReq(BaseModel):
    concepts: list[dict]
    filters: dict = {}
    strict: bool = False
    canonical: bool = True               # prune subsumed headings, sort blocks


class CountReq(BaseModel):
    query: str
    api_key: str | None = None
    email: str | None = None


class SearchReq(BaseModel):
    query: str
    max_records: int | None = 20000
    api_key: str | None = None
    email: str | None = None
    protocol: dict = {}


# ---------------------------------------------------------------- helpers

def _prepare_concepts(concepts: list[dict], *, strict: bool, canonical_form: bool) -> list[dict]:
    """
    Deterministic preparation of (possibly human-edited) concepts for compiling.

    canonical_form: resolve headings exactly, drop headings that another heading
    in the same block already explodes over, sort everything (canonical.py).
    strict: derive each free-text list from the MeSH index instead of the model's
    prose, so the query is a pure function of the selected headings.
    """
    ix = get_index()
    blocks = list(concepts)
    if canonical_form:
        blocks = canonical.canonicalize_blocks(blocks, ix, prune=True)["blocks"]
    return pipeline.finalize(blocks, strict=strict, ix=ix)


def _pubmed(api_key: str | None, email: str | None) -> PubMed:
    return PubMed(
        api_key=api_key or os.environ.get("NCBI_API_KEY", ""),
        email=email or os.environ.get("NCBI_EMAIL", ""),
    )


# ---------------------------------------------------------------- routes

@app.get("/api/config")
def config():
    return {
        "has_openrouter_key": bool(os.environ.get("OPENROUTER_API_KEY")),
        "has_ncbi_key": bool(os.environ.get("NCBI_API_KEY")),
        "ncbi_email": os.environ.get("NCBI_EMAIL", ""),
        "default_model": os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
        "domains": get_vocab().domains(),
        "modes": list(pipeline.MODES),
        "default_mode": "hybrid",
        "prompt_version": DEFAULT_PROMPT_VERSION,
    }


@app.get("/api/vocab")
def vocab(domains: str = ""):
    d = [x for x in domains.split(",") if x] or None
    return {"clusters": [c.to_dict() for c in get_vocab().clusters(d)]}


def _ui_concepts(build: dict) -> list[dict]:
    """
    Canonical blocks -> the review shape the frontend renders.

    Every heading here is already resolved exactly (canonical.py), so each one
    carries its single descriptor and is pre-checked. Headings the model proposed
    that did NOT resolve are reported separately under "dropped" instead of being
    fuzzy-matched into something else.
    """
    ix = get_index()
    out = []
    for b, final in zip(build["blocks"], build["concepts"]):
        mesh = []
        for dui in b.get("duis", []):
            d = ix.get(dui)
            if d is None:
                continue
            mesh.append({"query": canonical.preferred_label(ix, dui), "matched": True,
                         "options": [d.to_dict()], "selected_dui": dui})
        out.append({
            "name": b.get("name", ""),
            "slot": b.get("slot", "other"),
            "rationale": b.get("rationale", ""),
            "mesh": mesh,
            # show the model's own terms for review; strict mode replaces them
            # with index-derived synonyms at compile time
            "freetext": b.get("freetext", []) + b.get("vocab_freetext", []),
            "strict_terms": len(final.get("freetext", [])),
        })
    return out


@app.post("/api/map")
async def api_map(req: MapReq):
    if not req.question.strip():
        raise HTTPException(400, "question is required")
    try:
        build = await pipeline.build_async(
            req.question, domains=req.domains, mode=req.mode, model=req.model,
            api_key=req.api_key, extra_context=req.extra_context,
            prompt_version=req.prompt_version, facet_runs=req.facet_runs,
        )
    except OpenRouterError as e:
        raise HTTPException(502, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "concepts": _ui_concepts(build),
        "notes": build["notes"],
        "model": build["model"],
        "mode": build["mode"],
        "prompt_version": build["prompt_version"],
        "dropped": build["dropped"],
        "invalid_ids": build["invalid_ids"],
        "slate": build["slate"].get("candidates", []),
        "unmatched": build["slate"].get("unmatched", []),
    }


@app.post("/api/expand")
def api_expand(req: ExpandReq):
    ix = get_index()
    if not ix.get(req.dui):
        raise HTTPException(404, f"Unknown descriptor {req.dui}")
    return ix.expand(req.dui, explode=req.explode)


@app.post("/api/compile")
def api_compile(req: CompileReq):
    concepts = _prepare_concepts(req.concepts, strict=req.strict, canonical_form=req.canonical)
    try:
        res = query_builder.compile_search(concepts, req.filters)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {**res, "concepts": concepts}


@app.post("/api/count")
def api_count(req: CountReq):
    pm = _pubmed(req.api_key, req.email)
    res = pm.search(req.query)
    return {
        "count": res["count"],
        "translation": res["translation"],
        "warnings": res["warnings"],
        "errors": res["errors"],
    }


@app.post("/api/search")
def api_search(req: SearchReq):
    pm = _pubmed(req.api_key, req.email)
    res = pm.search(req.query)
    count = res["count"]
    if count == 0:
        return {"count": 0, "fetched": 0, "articles": [], "translation": res["translation"]}
    # fetch_query, not fetch_all: past 9,999 records NCBI's history server refuses
    # retstart, so the query has to be split by publication date.
    fetched = pm.fetch_query(req.query, max_records=req.max_records)
    articles = fetched["articles"]
    rows = [a.to_row() for a in articles]

    qhash = query_builder.query_hash(req.query)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = SEARCH_DIR / f"{stamp}_{qhash}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.csv").write_text(to_csv(articles), encoding="utf-8")
    (folder / "results.jsonl").write_text(to_jsonl(articles), encoding="utf-8")
    protocol = {
        **req.protocol,
        "query": req.query,
        "query_hash": qhash,
        "pubmed_translation": res["translation"],
        "total_count": count,
        "fetched": len(articles),
        "capped": fetched["capped"],
        # audit trail for a >9,999-hit search: which date slices were walked, and
        # whether any records fell outside them
        "date_slices": [{k: s[k] for k in ("from", "to", "count", "truncated")}
                        for s in fetched["slices"]] if len(fetched["slices"]) > 1 else [],
        "records_unaccounted": fetched["missing"],
        "timestamp": stamp,
    }
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    return {
        "count": count,
        "fetched": len(articles),
        "capped": protocol["capped"],
        "translation": res["translation"],
        "hash": qhash,
        "folder": folder.name,
        "slices": len(fetched["slices"]),
        "records_unaccounted": fetched["missing"],
        "articles": rows,
    }


@app.get("/api/download")
def api_download(folder: str, fmt: str = "csv"):
    fname = {"csv": "results.csv", "jsonl": "results.jsonl", "protocol": "protocol.json"}.get(fmt)
    if not fname:
        raise HTTPException(400, "fmt must be csv | jsonl | protocol")
    path = SEARCH_DIR / folder / fname
    if not path.exists() or SEARCH_DIR not in path.resolve().parents:
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=f"{folder}_{fname}", media_type="application/octet-stream")


@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND / "index.html").read_text(encoding="utf-8")


# static assets (js/css) — mounted last so /api/* wins
app.mount("/", StaticFiles(directory=str(FRONTEND)), name="static")
