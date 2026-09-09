"""
Facet segmentation: research question -> ordered PICO-style facets, each a
(name, role, token span) over the question text. NOT a port of anything in
the upstream sensein/search-strategy-generator project -- that tool asks a
model to select vocabulary IDs from a candidate slate (a high-entropy
decision: "which of these 40 headings matter"). This module asks a model for
the lowest-entropy thing it can be asked for instead: WHERE in the sentence
each concept lives. Vocabulary selection happens afterwards in closure.py, as
a pure function of the span and the local MeSH index -- the model never sees
MeSH at all, so it cannot hallucinate a heading or pick between synonyms.

Reproducibility strategy: self-consistency voting (Wang et al. 2022,
"Self-Consistency Improves Chain of Thought Reasoning", applied here to span
segmentation rather than reasoning chains). Instead of trusting one sample,
draw k independent completions of the same low-entropy task and keep only
spans a majority of the k runs agree on. Because segmentation is already a
lower-entropy task than vocabulary generation, and disagreement gets voted
away instead of silently compiled into the query, agreement should be higher
than single-shot free-text proposal by construction -- that is a testable
claim, not an assumption; see eval/determinism_closure.py.

`heuristic_facets` is a zero-network, 100%-deterministic fallback (used when
no LLM key is configured, or wired in as one of the k "voters" itself) so
this pipeline always has a model-independent path, the same property
mesh_only gives the upstream tool -- built independently here via simple
conjunction/punctuation chunking rather than MeSH span-matching.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import requests

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-+]*")

ROLE_HINTS: tuple[tuple[str, str], ...] = (
    ("population", "population"), ("participant", "population"),
    ("patient", "population"), ("subject", "population"),
    ("cohort", "population"), ("species", "population"),
    ("animal", "population"), ("children", "population"),
    ("adult", "population"), ("women", "population"), ("men", "population"),
    ("intervention", "intervention"), ("exposure", "intervention"),
    ("treatment", "intervention"), ("therapy", "intervention"),
    ("drug", "intervention"), ("stimulation", "intervention"),
    ("training", "intervention"),
    ("versus", "comparator"), (" vs ", "comparator"), ("compared", "comparator"),
    ("placebo", "comparator"), ("control", "comparator"),
    ("outcome", "outcome"), ("effect", "outcome"), ("risk", "outcome"),
    ("mortality", "outcome"), ("survival", "outcome"), ("recovery", "outcome"),
    ("method", "method"), ("technique", "method"), ("assay", "method"),
    ("sequencing", "method"), ("imaging", "method"), ("analysis", "method"),
)

_SPLIT_WORDS = frozenset("""
and with in for versus vs among during following after before compared
""".split())

STOPWORDS = frozenset("""
a an the of to on at by is are was were be been being this that these those
their its his her our your my as it
""".split())


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def _guess_role(phrase: str) -> str:
    low = f" {phrase.lower()} "
    for hint, role in ROLE_HINTS:
        if hint in low:
            return role
    return "other"


def heuristic_facets(question: str) -> list[dict]:
    """Deterministic, network-free segmentation: chunk on commas/semicolons
    and a small set of coordinating words, dropping stray stopword-only
    chunks. Always available, always reproducible by construction."""
    toks = _tokens(question)
    chunks: list[list[int]] = []
    current: list[int] = []
    for i, t in enumerate(toks):
        low = t.lower()
        if low in _SPLIT_WORDS and current:
            chunks.append(current)
            current = []
            continue
        current.append(i)
    if current:
        chunks.append(current)

    facets = []
    for idxs in chunks:
        content = [i for i in idxs if toks[i].lower() not in STOPWORDS]
        if not content:
            continue
        start, end = content[0], content[-1]
        phrase = " ".join(toks[start:end + 1])
        facets.append({"name": phrase, "role": _guess_role(phrase),
                        "start": start, "end": end})
    return facets


# --------------------------------------------------------------- LLM voting

FACET_SYSTEM_PROMPT = """You segment a biomedical research question into PICO-style \
facets (Population, Intervention/Exposure, Comparator, Outcome, Method, Context).

Rules:
- Every "text" you return MUST be an EXACT, VERBATIM, contiguous substring of the \
question (same characters, same order, same case). Do not paraphrase, translate, \
reorder, or add words that are not in the question.
- Prefer 2-5 facets covering the question's distinct concepts. Do not split a single \
noun phrase (e.g. "type 2 diabetes") across two facets.
- Do not propose vocabulary, synonyms, or MeSH headings. Only mark spans and roles.

Return STRICT JSON only:
{"facets": [{"text": "<verbatim substring>", "role": "population|intervention|comparator|outcome|method|context"}]}"""

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "120"))

# Same provider-routing idea as openrouter_client.py (prefix a model id with
# "ollama/" or "hf/" to run it locally / via HF Inference instead of
# OpenRouter) -- infrastructure plumbing, reimplemented here rather than
# imported so facets.py has no dependency on openrouter_client.py's prompt
# or JSON-repair logic, which belongs to the legacy "llm" mode only.
_PROVIDERS = {
    "ollama": {"url": os.environ.get("OLLAMA_URL", "http://localhost:11434/v1") + "/chat/completions",
              "key_env": "OLLAMA_API_KEY", "requires_key": False},
    "hf": {"url": os.environ.get("HF_BASE_URL", "https://router.huggingface.co/v1") + "/chat/completions",
          "key_env": "HF_TOKEN", "requires_key": True},
}


def _resolve_provider(model: str) -> tuple[dict, str]:
    if model and "/" in model:
        prefix, rest = model.split("/", 1)
        if prefix in _PROVIDERS:
            return _PROVIDERS[prefix], rest
    return {"url": OPENROUTER_URL, "key_env": "OPENROUTER_API_KEY", "requires_key": True}, model


class FacetLLMError(RuntimeError):
    pass


def _extract_json(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1] if content.count("```") >= 2 else content[3:]
        if content.lstrip().startswith("json"):
            content = content.lstrip()[4:]
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        raise FacetLLMError(f"No JSON object in facet response: {content[:200]}")
    return json.loads(content[start:end + 1])


VALID_PICO_ROLES = frozenset(
    "population intervention comparator outcome method context other".split())


def _normalize_pico_role(role: str) -> str:
    """Small/instruction-weak models sometimes echo the prompt's pipe-joined
    enum verbatim instead of picking one value, or add stray punctuation.
    If exactly one valid role token is present, use it; if none (garbage) or
    more than one (an echoed enum -- not an actual choice) are present, map
    to 'other' rather than arbitrarily picking the first one or propagating
    garbage into the UI."""
    found = [p for p in re.split(r"[|,/\s]+", role.lower().strip()) if p in VALID_PICO_ROLES]
    return found[0] if len(found) == 1 else "other"


def _locate_span(question: str, text: str) -> tuple[int, int] | None:
    """Map a (possibly whitespace-mangled) model-returned substring back onto
    the question's own token indices, verbatim-only (no fuzzy matching -- an
    unmappable span is discarded rather than guessed at)."""
    toks = _tokens(question)
    want = _tokens(text)
    if not want:
        return None
    low = [t.lower() for t in toks]
    wlow = [t.lower() for t in want]
    n = len(wlow)
    for i in range(0, len(low) - n + 1):
        if low[i:i + n] == wlow:
            return i, i + n - 1
    return None


def _one_llm_pass(question: str, *, model: str | None, api_key: str | None,
                   seed: int) -> list[dict]:
    provider, upstream_model = _resolve_provider(model or DEFAULT_MODEL)
    key = api_key or os.environ.get(provider["key_env"], "")
    if provider["requires_key"] and not key:
        raise FacetLLMError(f"{provider['key_env']} not set")
    payload = {
        "model": upstream_model,
        "temperature": 0.4,          # slight temp: k=1 samples must differ or voting is moot
        "seed": seed,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": FACET_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if provider["url"] == OPENROUTER_URL:
        headers["HTTP-Referer"] = "http://localhost"
        headers["X-Title"] = "Deterministic Search - facets"
    r = requests.post(provider["url"], headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        raise FacetLLMError(f"LLM {r.status_code}: {r.text[:300]}")
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    out = []
    for f in parsed.get("facets", []):
        if not isinstance(f, dict):
            continue
        text = str(f.get("text", "")).strip()
        role = str(f.get("role", "other")).strip().lower()
        if not text:
            continue
        span = _locate_span(question, text)
        if span is None:
            continue  # unmappable / non-verbatim span: drop rather than guess
        out.append({"name": text, "role": _normalize_pico_role(role),
                     "start": span[0], "end": span[1]})
    return out


async def _one_llm_pass_async(question: str, *, model: str | None, api_key: str | None,
                               seed: int) -> list[dict]:
    import httpx
    provider, upstream_model = _resolve_provider(model or DEFAULT_MODEL)
    key = api_key or os.environ.get(provider["key_env"], "")
    if provider["requires_key"] and not key:
        raise FacetLLMError(f"{provider['key_env']} not set")
    payload = {
        "model": upstream_model,
        "temperature": 0.4,
        "seed": seed,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": FACET_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if provider["url"] == OPENROUTER_URL:
        headers["HTTP-Referer"] = "http://localhost"
        headers["X-Title"] = "Deterministic Search - facets"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(provider["url"], headers=headers, json=payload)
    if r.status_code != 200:
        raise FacetLLMError(f"LLM {r.status_code}: {r.text[:300]}")
    content = r.json()["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    out = []
    for f in parsed.get("facets", []):
        if not isinstance(f, dict):
            continue
        text = str(f.get("text", "")).strip()
        role = str(f.get("role", "other")).strip().lower()
        if not text:
            continue
        span = _locate_span(question, text)
        if span is None:
            continue
        out.append({"name": text, "role": _normalize_pico_role(role),
                     "start": span[0], "end": span[1]})
    return out


def _cluster_and_vote(runs: list[list[dict]], min_runs_agree: int) -> list[dict]:
    """Union-find spans that overlap across runs; keep a cluster only if
    >= min_runs_agree DISTINCT runs contributed to it. The kept span/role are
    the most-common (start,end)/role within the cluster (mode), so the result
    is a pure function of the (unordered) set of run outputs -- reruns of the
    LLM in a different order cannot change the vote."""
    items = [(ri, f) for ri, run in enumerate(runs) for f in run]
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            _, fi = items[i]
            _, fj = items[j]
            if fi["start"] <= fj["end"] and fj["start"] <= fi["end"]:
                union(i, j)

    clusters: dict[int, list[tuple[int, dict]]] = {}
    for idx, (ri, f) in enumerate(items):
        clusters.setdefault(find(idx), []).append((ri, f))

    out = []
    for members in clusters.values():
        distinct_runs = {ri for ri, _ in members}
        if len(distinct_runs) < min_runs_agree:
            continue
        span_votes = Counter((f["start"], f["end"]) for _, f in members)
        (start, end), _ = max(span_votes.items(),
                               key=lambda kv: (kv[1], -(kv[0][1] - kv[0][0])))
        role_votes = Counter(f["role"] for _, f in members if
                              (f["start"], f["end"]) == (start, end))
        role = role_votes.most_common(1)[0][0] if role_votes else "other"
        name_votes = Counter(f["name"] for _, f in members if
                              (f["start"], f["end"]) == (start, end))
        name = name_votes.most_common(1)[0][0]
        out.append({"name": name, "role": role, "start": start, "end": end,
                     "agreement": len(distinct_runs), "of": len(runs)})
    out.sort(key=lambda f: f["start"])
    return out


def segment(question: str, *, model: str | None = None, api_key: str | None = None,
            k: int = 3, use_llm: bool = True, min_agreement: float = 0.5) -> dict:
    """Synchronous k-sample self-consistency segmentation (used by the eval
    harness and any sync caller). Falls back to heuristic_facets if no key is
    configured (for a key-requiring provider) or every LLM pass fails."""
    provider, _ = _resolve_provider(model or DEFAULT_MODEL)
    have_key = bool(api_key or os.environ.get(provider["key_env"], ""))
    if not use_llm or (provider["requires_key"] and not have_key):
        return {"facets": heuristic_facets(question), "mode": "heuristic", "runs": 0}
    runs: list[list[dict]] = []
    errors: list[str] = []
    for i in range(max(1, k)):
        try:
            runs.append(_one_llm_pass(question, model=model, api_key=api_key, seed=1000 + i))
        except FacetLLMError as e:
            errors.append(str(e))
    if not runs:
        return {"facets": heuristic_facets(question), "mode": "heuristic_fallback",
                 "runs": 0, "errors": errors}
    need = max(1, int((len(runs) * min_agreement) + 0.999999))
    voted = _cluster_and_vote(runs, need)
    if not voted:
        return {"facets": heuristic_facets(question), "mode": "heuristic_fallback",
                 "runs": len(runs), "errors": errors}
    return {"facets": voted, "mode": "llm_voted", "runs": len(runs), "errors": errors}


async def segment_async(question: str, *, model: str | None = None,
                         api_key: str | None = None, k: int = 3, use_llm: bool = True,
                         min_agreement: float = 0.5) -> dict:
    """Async k-sample self-consistency segmentation (used by FastAPI). Runs
    the k model calls concurrently so latency stays close to one call."""
    import asyncio
    provider, _ = _resolve_provider(model or DEFAULT_MODEL)
    have_key = bool(api_key or os.environ.get(provider["key_env"], ""))
    if not use_llm or (provider["requires_key"] and not have_key):
        return {"facets": heuristic_facets(question), "mode": "heuristic", "runs": 0}
    tasks = [_one_llm_pass_async(question, model=model, api_key=api_key, seed=1000 + i)
             for i in range(max(1, k))]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    runs: list[list[dict]] = []
    errors: list[str] = []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
        else:
            runs.append(r)
    if not runs:
        return {"facets": heuristic_facets(question), "mode": "heuristic_fallback",
                 "runs": 0, "errors": errors}
    need = int((len(runs) * min_agreement) + 0.999999)
    voted = _cluster_and_vote(runs, need)
    if not voted:
        return {"facets": heuristic_facets(question), "mode": "heuristic_fallback",
                 "runs": len(runs), "errors": errors}
    return {"facets": voted, "mode": "llm_voted", "runs": len(runs), "errors": errors}
