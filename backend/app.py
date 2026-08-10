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

from . import query_builder
from .domain_vocab import get_vocab
from .mesh_index import get_index
from .openrouter_client import OpenRouterError, map_question, map_question_async
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


class ExpandReq(BaseModel):
    dui: str
    explode: bool = True


class CompileReq(BaseModel):
    concepts: list[dict]
    filters: dict = {}
    strict: bool = False


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

def _resolve_mesh(candidate: str) -> dict:
    ix = get_index()
    exact = ix.exact(candidate)
    if exact:
        return {
            "query": candidate,
            "matched": True,
            "options": [d.to_dict() for d in exact],
            "selected_dui": exact[0].dui,
        }
    options = ix.search(candidate, limit=8)
    return {
        "query": candidate,
        "matched": False,
        "options": [d.to_dict() for d in options],
        "selected_dui": options[0].dui if options else "",
    }


def _strict_concepts(concepts: list[dict]) -> list[dict]:
    """
    Deterministic term derivation for compile: for each concept that has MeSH
    headings, replace its free-text with the exploded entry terms of those
    headings (a pure function of the local index — no LLM prose). Concepts with
    no MeSH heading keep their free-text, so current-jargon terms that aren't in
    MeSH still survive.
    """
    ix = get_index()
    out = []
    for c in concepts:
        headings = list(c.get("mesh", []))
        if headings:
            explode = bool(c.get("explode", True))
            duis = [d.dui for h in headings for d in ix.exact(h)]
            c = {**c, "freetext": ix.strict_terms(duis, explode=explode)}
        out.append(c)
    return out


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
    }


@app.get("/api/vocab")
def vocab(domains: str = ""):
    d = [x for x in domains.split(",") if x] or None
    return {"clusters": [c.to_dict() for c in get_vocab().clusters(d)]}


def resolve_concepts(result: dict, domains: list[str]) -> dict:
    """
    Deterministic post-LLM step: resolve each proposed MeSH heading against the
    local index and augment free-text with matching domain-vocab synonyms.
    Shared by the async endpoint and the (sync) determinism test.
    """
    vocab_obj = get_vocab()
    concepts_out = []
    for c in result["concepts"]:
        mesh = [_resolve_mesh(m) for m in c["mesh_candidates"]]
        # augment free-text with matching domain-vocab synonyms
        domain_syn: list[str] = []
        haystack = " ".join([c["name"]] + c["freetext"]).lower()
        for cl in vocab_obj.clusters(domains or None):
            if cl.concept.lower() in haystack or any(
                s.lower() in haystack for s in cl.synonyms
            ):
                domain_syn.extend(cl.synonyms)
        # dedupe freetext (LLM + domain), preserve order
        seen, freetext = set(), []
        for t in c["freetext"] + domain_syn:
            k = t.lower()
            if k not in seen:
                seen.add(k)
                freetext.append(t)
        concepts_out.append({
            "name": c["name"],
            "rationale": c["rationale"],
            "mesh": mesh,
            "freetext": freetext,
        })
    return {"concepts": concepts_out, "notes": result["notes"], "model": result["model"]}


@app.post("/api/map")
async def api_map(req: MapReq):
    if not req.question.strip():
        raise HTTPException(400, "question is required")
    try:
        result = await map_question_async(
            req.question, domains=req.domains, model=req.model,
            api_key=req.api_key, extra_context=req.extra_context,
        )
    except OpenRouterError as e:
        raise HTTPException(502, str(e))
    return resolve_concepts(result, req.domains)


@app.post("/api/expand")
def api_expand(req: ExpandReq):
    ix = get_index()
    if not ix.get(req.dui):
        raise HTTPException(404, f"Unknown descriptor {req.dui}")
    return ix.expand(req.dui, explode=req.explode)


@app.post("/api/compile")
def api_compile(req: CompileReq):
    concepts = _strict_concepts(req.concepts) if req.strict else req.concepts
    try:
        return query_builder.compile_search(concepts, req.filters)
    except ValueError as e:
        raise HTTPException(400, str(e))


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
    articles = pm.fetch_all(
        res["webenv"], res["query_key"], count, max_records=req.max_records,
    )
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
        "capped": req.max_records is not None and count > req.max_records,
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
