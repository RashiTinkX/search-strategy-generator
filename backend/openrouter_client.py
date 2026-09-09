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

SYSTEM_PROMPT = """You are a biomedical search strategist helping build an EXHAUSTIVE, \
reproducible PubMed search for a systematic literature review.

Decompose the user's research question into orthogonal CONCEPT BLOCKS (PICO-style: \
e.g. Population, Intervention/Exposure, Outcome, Method). Blocks are ANDed together; \
terms within a block are ORed. Aim for high recall — err toward MORE synonyms.

For each concept block provide:
- "name": short label for the block
- "mesh_candidates": likely MeSH Descriptor HEADINGS (official controlled-vocabulary \
terms, e.g. "Neuronal Plasticity", "Magnetic Resonance Imaging"). Give the exact \
canonical heading you believe exists; do not invent qualifiers.
- "freetext": free-text title/abstract synonyms, including abbreviations, spelling \
variants (British/US), plurals, and CURRENT method/tool jargon that MeSH may lack \
(e.g. "RNA-seq", "scRNA-seq", "connectome", "optogenetics"). These catch articles \
not yet MeSH-indexed.

Return STRICT JSON only:
{
  "concepts": [
    {"name": "...", "mesh_candidates": ["..."], "freetext": ["..."], "rationale": "..."}
  ],
  "notes": "any caveats about scope or ambiguity"
}"""


class OpenRouterError(RuntimeError):
    pass


# Generous output budget: the system prompt asks for MANY synonyms, so a low
# provider default truncates the JSON mid-string (the UI 500). Ask for plenty.
MAX_TOKENS = int(os.environ.get("OPENROUTER_MAX_TOKENS", "8000"))
TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "180"))


def _build_request(question, domains, model, extra_context, seed):
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

    payload = {
        "model": upstream_model,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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


def _parse_response(status: int, text: str, url: str, model: str) -> dict:
    """Turn a raw HTTP response into the normalized concept dict (or raise)."""
    if status != 200:
        raise OpenRouterError(f"LLM {status} from {url}: {text[:500]}")
    try:
        data = json.loads(text)
        content = data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        raise OpenRouterError(f"Unexpected LLM response from {url}: {text[:500]}") from e

    parsed = _loads_lenient(content)
    clean = []
    for c in parsed.get("concepts", []):
        if not isinstance(c, dict):
            continue
        clean.append({
            "name": str(c.get("name", "")).strip(),
            "mesh_candidates": [str(x).strip() for x in c.get("mesh_candidates", []) if str(x).strip()],
            "freetext": [str(x).strip() for x in c.get("freetext", []) if str(x).strip()],
            "rationale": str(c.get("rationale", "")).strip(),
        })
    return {"concepts": clean, "notes": str(parsed.get("notes", "")).strip(), "model": model}


def map_question(question: str, domains: list[str] | None = None,
                 model: str | None = None, api_key: str | None = None,
                 extra_context: str = "", seed: int | None = None) -> dict:
    """Synchronous map (used by the determinism test and any sync caller)."""
    provider, payload, headers = _build_request(question, domains, model, extra_context, seed)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.post(provider["url"], headers=headers, json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise OpenRouterError(f"LLM request failed ({provider['url']}): {e}") from e
    return _parse_response(r.status_code, r.text, provider["url"], model or DEFAULT_MODEL)


async def map_question_async(question: str, domains: list[str] | None = None,
                             model: str | None = None, api_key: str | None = None,
                             extra_context: str = "", seed: int | None = None) -> dict:
    """Async map (used by the FastAPI endpoint so the event loop isn't blocked)."""
    import httpx
    provider, payload, headers = _build_request(question, domains, model, extra_context, seed)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(provider["url"], headers=headers, json=payload)
    except httpx.HTTPError as e:
        raise OpenRouterError(f"LLM request failed ({provider['url']}): {e}") from e
    return _parse_response(r.status_code, r.text, provider["url"], model or DEFAULT_MODEL)


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
