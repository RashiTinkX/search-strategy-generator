"""
OpenRouter client: research question -> candidate search concepts.

The LLM is used ONLY to *propose* structure — concept blocks, candidate MeSH
heading names, and free-text synonyms. It never decides the final query. Every
proposed MeSH heading is resolved against the local MeSH index (mesh_index.py),
so a hallucinated heading simply resolves to nothing and is dropped. That keeps
the pipeline deterministic and auditable while still using the LLM's domain
knowledge to broaden recall.

Config (env):
    OPENROUTER_API_KEY   required
    OPENROUTER_MODEL     default: anthropic/claude-3.5-sonnet (override freely)
"""
from __future__ import annotations

import json
import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

# Provider routing by model-id prefix. All three speak the OpenAI chat-completions
# shape, so only the base URL and auth differ. Small open-source models can run
# locally via Ollama ("ollama/llama3.2:1b") or hosted via HF Inference
# ("hf/meta-llama/Llama-3.1-8B-Instruct"); anything else goes to OpenRouter.
PROVIDERS = {
    "ollama": {
        "url": os.environ.get("OLLAMA_URL", "http://localhost:11434/v1") + "/chat/completions",
        "key_env": "OLLAMA_API_KEY",   # usually unset; Ollama needs no auth
        "requires_key": False,
    },
    "hf": {
        "url": os.environ.get("HF_BASE_URL", "https://router.huggingface.co/v1") + "/chat/completions",
        "key_env": "HF_TOKEN",
        "requires_key": True,
    },
}
_OPENROUTER = {"url": OPENROUTER_URL, "key_env": "OPENROUTER_API_KEY", "requires_key": True}


def _resolve_provider(model: str) -> tuple[dict, str]:
    """Return (provider_cfg, upstream_model_id) from a possibly-prefixed model id."""
    if model and "/" in model:
        prefix, rest = model.split("/", 1)
        if prefix in PROVIDERS:
            return PROVIDERS[prefix], rest
    return _OPENROUTER, model

from .prompts import PROPOSE_PROMPTS, PROPOSE_V1, SELECT_HYBRID, format_slate

# Kept for back-compat: the original single prompt. New callers pass
# prompt_version="v2" (see prompts.py for why the rules changed).
SYSTEM_PROMPT = PROPOSE_V1
DEFAULT_PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v2")


class OpenRouterError(RuntimeError):
    pass


# Generous output budget: the system prompt asks for MANY synonyms, so a low
# provider default truncates the JSON mid-string (the UI 500). Ask for plenty.
MAX_TOKENS = int(os.environ.get("OPENROUTER_MAX_TOKENS", "8000"))
TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "180"))


def _build_request(question, domains, model, extra_context, seed,
                   system: str | None = None, slate_text: str = ""):
    """Shared request assembly for the sync and async paths."""
    model = model or DEFAULT_MODEL
    provider, upstream_model = _resolve_provider(model)
    key = provider.get("_api_key") or os.environ.get(provider["key_env"], "")
    if provider["requires_key"] and not key:
        raise OpenRouterError(
            f"{provider['key_env']} is not set. Export it or paste it in the UI settings."
        )

    user_msg = f"Research question / topic:\n{question}\n"
    if domains:
        user_msg += f"\nRelevant domains (favor their current jargon): {', '.join(domains)}\n"
    if extra_context:
        user_msg += f"\nAdditional scope notes:\n{extra_context}\n"
    if slate_text:
        user_msg += f"\n{slate_text}\n"

    payload = {
        "model": upstream_model,
        "temperature": 0,
        # top_p=1 with temperature 0 is greedy decoding on every provider that
        # honours it; sending both removes one source of sampling drift.
        "top_p": 1,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "response_format": {"type": "json_object"},
    }
    if seed is not None:  # best-effort; ignored by providers that don't support it
        payload["seed"] = seed
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if provider is _OPENROUTER:
        headers["HTTP-Referer"] = "http://localhost"
        headers["X-Title"] = "Deterministic PubMed Search"
    return provider, payload, headers


def _content(status: int, text: str, url: str) -> str:
    """Extract the assistant message content from a chat-completions response."""
    if status != 200:
        raise OpenRouterError(f"LLM {status} from {url}: {text[:500]}")
    try:
        data = json.loads(text)
        return data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise OpenRouterError(f"Unexpected LLM response from {url}: {text[:500]}") from e


def _strs(seq) -> list[str]:
    return [str(x).strip() for x in (seq or []) if str(x).strip()]


def _parse_response(status: int, text: str, url: str, model: str) -> dict:
    """Turn a raw HTTP response into the normalized concept dict (or raise)."""
    parsed = _loads_lenient(_content(status, text, url))
    clean = []
    for c in parsed.get("concepts", []):
        if not isinstance(c, dict):
            continue
        # v2 asks for "mesh"; v1 asked for "mesh_candidates". Accept both so the
        # evaluation can A/B the prompts through one code path.
        headings = _strs(c.get("mesh") or c.get("mesh_candidates"))
        clean.append({
            "name": str(c.get("name", "")).strip(),
            "slot": str(c.get("slot", "")).strip(),
            "mesh_candidates": headings,
            "freetext": _strs(c.get("freetext")),
            "rationale": str(c.get("rationale", "")).strip(),
        })
    return {"concepts": clean, "notes": str(parsed.get("notes", "")).strip(), "model": model}


def _parse_selection(status: int, text: str, url: str, model: str) -> dict:
    """
    Normalize a hybrid selection response: blocks of candidate ids.

    Ids that are not integers are dropped here; ids that are not in the slate are
    dropped by the caller (pipeline.py), which owns the slate.
    """
    parsed = _loads_lenient(_content(status, text, url))
    blocks = []
    for b in parsed.get("blocks", parsed.get("concepts", [])):
        if not isinstance(b, dict):
            continue
        ids: list[int] = []
        for x in b.get("ids", b.get("candidates", [])) or []:
            try:
                ids.append(int(str(x).strip()))
            except (TypeError, ValueError):
                continue
        blocks.append({
            "name": str(b.get("name", "")).strip(),
            "slot": str(b.get("slot", "")).strip(),
            "ids": sorted(set(ids)),
            "freetext": _strs(b.get("freetext")),
        })
    return {"blocks": blocks, "notes": str(parsed.get("notes", "")).strip(), "model": model}


def _system_for(prompt_version: str | None) -> str:
    v = (prompt_version or DEFAULT_PROMPT_VERSION).lower()
    if v not in PROPOSE_PROMPTS:
        raise OpenRouterError(f"Unknown prompt version {v!r} (have: {sorted(PROPOSE_PROMPTS)})")
    return PROPOSE_PROMPTS[v]


def _post_sync(provider, payload, headers):
    try:
        return requests.post(provider["url"], headers=headers, json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise OpenRouterError(f"LLM request failed ({provider['url']}): {e}") from e


async def _post_async(provider, payload, headers):
    import httpx
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            return await client.post(provider["url"], headers=headers, json=payload)
    except httpx.HTTPError as e:
        raise OpenRouterError(f"LLM request failed ({provider['url']}): {e}") from e


def map_question(question: str, domains: list[str] | None = None,
                 model: str | None = None, api_key: str | None = None,
                 extra_context: str = "", seed: int | None = None,
                 prompt_version: str | None = None) -> dict:
    """Synchronous map (used by the determinism evaluation and any sync caller)."""
    provider, payload, headers = _build_request(
        question, domains, model, extra_context, seed, system=_system_for(prompt_version))
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = _post_sync(provider, payload, headers)
    return _parse_response(r.status_code, r.text, provider["url"], model or DEFAULT_MODEL)


async def map_question_async(question: str, domains: list[str] | None = None,
                             model: str | None = None, api_key: str | None = None,
                             extra_context: str = "", seed: int | None = None,
                             prompt_version: str | None = None) -> dict:
    """Async map (used by the FastAPI endpoint so the event loop isn't blocked)."""
    provider, payload, headers = _build_request(
        question, domains, model, extra_context, seed, system=_system_for(prompt_version))
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = await _post_async(provider, payload, headers)
    return _parse_response(r.status_code, r.text, provider["url"], model or DEFAULT_MODEL)


def select_candidates(question: str, slate: dict, domains: list[str] | None = None,
                      model: str | None = None, api_key: str | None = None,
                      extra_context: str = "", seed: int | None = None) -> dict:
    """Hybrid mode (sync): pick blocks out of a deterministic candidate slate."""
    provider, payload, headers = _build_request(
        question, domains, model, extra_context, seed,
        system=SELECT_HYBRID, slate_text=format_slate(slate))
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = _post_sync(provider, payload, headers)
    return _parse_selection(r.status_code, r.text, provider["url"], model or DEFAULT_MODEL)


async def select_candidates_async(question: str, slate: dict, domains: list[str] | None = None,
                                  model: str | None = None, api_key: str | None = None,
                                  extra_context: str = "", seed: int | None = None) -> dict:
    """Hybrid mode (async) — used by the FastAPI endpoint."""
    provider, payload, headers = _build_request(
        question, domains, model, extra_context, seed,
        system=SELECT_HYBRID, slate_text=format_slate(slate))
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = await _post_async(provider, payload, headers)
    return _parse_selection(r.status_code, r.text, provider["url"], model or DEFAULT_MODEL)


def _loads_lenient(content: str) -> dict:
    """
    Parse model JSON tolerantly. Handles: clean JSON, markdown-fenced JSON, and
    TRUNCATED JSON (small models hit the token cap mid-string) by balancing any
    open string/brackets so we salvage the concepts instead of 500-ing.
    """
    content = (content or "").strip()
    if not content:
        raise OpenRouterError("LLM returned empty content")
    # strip a leading/trailing markdown code fence if present
    if content.startswith("```"):
        content = content.split("```", 2)[1] if content.count("```") >= 2 else content[3:]
        if content.lstrip().startswith("json"):
            content = content.lstrip()[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    if start == -1:
        raise OpenRouterError(f"No JSON object in LLM output: {content[:200]}")
    body = content[start:]
    try:
        return json.loads(body[: body.rfind("}") + 1])
    except json.JSONDecodeError:
        pass
    salvaged = _repair_truncated(body)
    if salvaged is not None:
        return salvaged
    raise OpenRouterError(f"Could not parse LLM JSON (truncated?): {content[:200]}")


def _balance(frag: str) -> str:
    """Close an unterminated string and any still-open [ / { in `frag`."""
    stack, in_str, esc = [], False, False
    for ch in frag:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    res = frag + ('"' if in_str else "")
    res = res.rstrip().rstrip(",")
    while stack:
        res += "}" if stack.pop() == "{" else "]"
    return res


def _repair_truncated(s: str):
    """
    Salvage truncated JSON (the model hit its output-token cap mid-structure).
    Tries the whole string, then progressively shorter prefixes cut at STRUCTURAL
    boundaries (right after a `}`/`]` or a closed string) so we never leave a
    dangling key/colon; each candidate is bracket-balanced and parsed. Returns the
    first dict that parses, else None.
    """
    cuts, in_str, esc = [len(s)], False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                cuts.append(i + 1)          # end of a complete string value
            continue
        if ch == '"':
            in_str = True
        elif ch in "}]":
            cuts.append(i + 1)               # end of a complete element/container
    for cut in sorted(set(cuts), reverse=True):
        try:
            obj = json.loads(_balance(s[:cut]))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
