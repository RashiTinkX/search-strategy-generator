"""
Deterministic MeSH candidate generation from the question text.

This is the "retrieve" half of the hybrid pipeline. Given the raw question we
find, with no LLM involved, every MeSH descriptor the question *lexically*
mentions: each maximal word span that exactly matches a MeSH heading or entry
term (synonym), plus — optionally — the immediate broader descriptor of each
match so the model has a way to widen a facet.

Why it matters for reproducibility: in hybrid mode the LLM never writes a
heading, it only picks numbers off this slate. A heading that is not in the
slate cannot enter the query, so hallucination is structurally impossible and
the model's entire output space is a subset of a set that is itself a pure
function of (question, MeSH index).

The slate is also directly usable without any LLM (`mesh_only` mode): every
maximal match becomes its own ANDed block.
"""
from __future__ import annotations

import itertools
import re

from .mesh_index import MeshIndex

# Words that are never a search facet on their own. Several are real MeSH
# headings ("Models, Theoretical" has the entry term "model"), so without this
# list a question like "...detection methods" would AND a useless block and
# throttle recall to near zero.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can cannot could did do does doing down during
each few for from further had has have having he her here hers him his how i if in
into is it its itself just me more most my no nor not of off on once only or other
our out over own same she should so some such than that the their them then there
these they this those through to too under until up very was we were what when where
which while who whom why will with would you your
""".split())

# Generic research/discourse nouns: they DO resolve to MeSH descriptors but are
# almost never the facet the author means.
GENERIC_TERMS = frozenset("""
analysis approach approaches assessment characterization comparison control data
detection development effect effects evaluation evidence experiment experiments
factor factors finding findings function group groups identification impact
improvement influence level levels measure measurement measurements mechanism
mechanisms method methods model models outcome outcomes pattern patterns
performance population populations problem process processes protocol protocols
research response responses result results review reviews role sample samples
strategy strategies study studies system systems technique techniques test tests
treatment type types use validation variation
""".split())

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-+]*")
MAX_NGRAM = 6


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def _variants(phrase: str) -> list[str]:
    """
    Surface variants to try against the index, in a fixed order.

    MeSH stores headings in singular, inverted form ("Memory, Short-Term"), but
    entry terms carry the natural-order and plural forms, so a light
    singular/plural pass is enough and stays deterministic.
    """
    out = [phrase]
    low = phrase.lower()
    if "-" in phrase:            # "Alzheimer's-disease" style joins, "RNA-Seq"
        out.append(phrase.replace("-", " "))
    if low.endswith("ies") and len(low) > 4:
        out.append(phrase[:-3] + "y")
    if low.endswith("ses") and len(low) > 4:
        out.append(phrase[:-2])
    if low.endswith("s") and not low.endswith("ss"):
        out.append(phrase[:-1])
    else:
        out.append(phrase + "s")
    seen, uniq = set(), []
    for v in out:
        if v.lower() not in seen:
            seen.add(v.lower())
            uniq.append(v)
    return uniq


def _lookup(ix: MeshIndex, phrase: str):
    """Exact heading or exact entry-term hit for a phrase (or one of its
    singular/plural variants). Lowest DUI wins so ties are stable."""
    for v in _variants(phrase):
        hits = ix.exact(v) or ix.by_entry_term(v)
        if hits:
            return sorted(hits, key=lambda d: d.dui)[0], v
    # Technical compounds ("CRISPR-Cas9", "AAV-PHP.eB") are one token that MeSH
    # never carries verbatim, but a hyphen part usually is a heading ("CRISPR").
    # Restricted to compounds with a digit or an internal capital so ordinary
    # hyphenations ("single-cell" -> "single") are left alone.
    if "-" in phrase and " " not in phrase and (
        any(ch.isdigit() for ch in phrase) or any(ch.isupper() for ch in phrase[1:])
    ):
        for part in sorted(phrase.split("-"), key=lambda p: (-len(p), p)):
            if len(part) < 4 or part.lower() in STOPWORDS or part.lower() in GENERIC_TERMS:
                continue
            hits = ix.exact(part) or ix.by_entry_term(part)
            if hits:
                return sorted(hits, key=lambda d: d.dui)[0], part
    return None, ""


_SUBTOK_RE = re.compile(r"[a-z0-9]+")
_PREFIX = 6          # how much of a question word must agree with a term word
_ABBREV_MIN = 3      # a shorter term word is treated as an abbreviation


def _word_matches(word: str, term_toks: list[str]) -> bool:
    """
    Does a question word correspond to some word of a MeSH term?

    Prefix agreement of `_PREFIX` characters ("sequencing" ~ "sequence", but NOT
    "optogenetic" ~ "optically"), plus the abbreviation case where the term uses
    a short form of the question's word ("seq" in "Single-Cell RNA-Seq").
    """
    for sub in _SUBTOK_RE.findall(word.lower()):
        head = sub[:_PREFIX]
        if not any(t.startswith(head) or (len(t) >= _ABBREV_MIN and sub.startswith(t))
                   for t in term_toks):
            return False
    return True


def _term_covers(words: list[str], term: str) -> bool:
    """Every word of the question window is represented in the MeSH term."""
    term_toks = _SUBTOK_RE.findall(term.lower())
    return all(_word_matches(w, term_toks) for w in words)


def phrase_candidates(ix: MeshIndex, question: str, spans: list[dict], *,
                      max_windows: int = 8, per_window: int = 2) -> list[dict]:
    """
    Multi-word descriptors the question expresses in different words.

    "single-cell RNA sequencing" is exactly MeSH's "Single-Cell RNA-Seq"
    (D000092386) but shares no exact string with it, so span matching only finds
    the weaker "RNA Sequencing". This pass takes short windows of adjacent
    content words that contain something the span pass missed, and asks the index
    for entry terms containing a 3-character prefix of EVERY word in the window.
    Deterministic: fixed window order, fixed ranking (fewest words, then
    shortest, then alphabetical), fixed caps.
    """
    toks = _tokens(question)
    covered: set[int] = set()
    for s in spans:
        covered.update(range(s["start"], s["start"] + s["size"]))
    content = [(i, t) for i, t in enumerate(toks)
               if t.lower() not in STOPWORDS and t.lower() not in GENERIC_TERMS and len(t) >= 4]

    windows: list[list[tuple[int, str]]] = []
    for size in (3, 2):
        for j in range(0, len(content) - size + 1):
            win = content[j:j + size]
            if all(i in covered for i, _ in win):
                continue                      # nothing new to gain
            if win[-1][0] - win[0][0] > size:  # words must be near-adjacent
                continue
            windows.append(win)
    # Single words that are themselves compounds ("CRISPR-Cas9"): the useful
    # heading ("CRISPR-Cas Systems") shares their parts but not their string.
    # (also when the word already matched: "CRISPR-Cas9" matches the *sequence*
    # descriptor exactly, while the technique descriptor is the better facet)
    for i, t in content:
        if len(_SUBTOK_RE.findall(t.lower())) < 2:
            continue
        windows.append([(i, t)])
    windows = windows[:max_windows]

    seen: set[str] = {s["dui"] for s in spans}
    out: list[dict] = []
    for win in windows:
        words = [w for _, w in win]
        pats: list[str] = []
        for w in words:
            for sub in _SUBTOK_RE.findall(w.lower()):
                if len(sub) >= _ABBREV_MIN and f"%{sub[:3]}%" not in pats:
                    pats.append(f"%{sub[:3]}%")
        if not pats or len(pats) > 5:
            continue
        sql = ("SELECT dui, term FROM entry_term WHERE "
               + " AND ".join(["term_norm LIKE ?"] * len(pats)))
        rows = ix.like_terms(sql, pats)
        best: dict[str, str] = {}
        for dui, term in rows:
            if not _term_covers(words, term):
                continue                       # LIKE was only a cheap prefilter
            cur = best.get(dui)
            if cur is None or (len(term.split()), len(term), term.lower()) < (
                    len(cur.split()), len(cur), cur.lower()):
                best[dui] = term
        ranked = sorted(best.items(), key=lambda kv: (len(kv[1].split()), len(kv[1]),
                                                     kv[1].lower(), kv[0]))
        added = 0
        for dui, term in ranked:
            if dui in seen or added >= per_window:
                continue
            d = ix.get(dui)
            if d is None:
                continue
            seen.add(dui)
            out.append({"dui": dui, "label": d.label, "span": " ".join(words),
                        "relation": "phrase", "via": term,
                        "start": win[0][0], "size": win[-1][0] - win[0][0] + 1})
            added += 1
    return out


def matched_spans(question: str, ix: MeshIndex) -> list[dict]:
    """
    Every MAXIMAL word span of the question that resolves to a descriptor.

    Longest-match-first, left to right: once "memory consolidation" matches, the
    contained spans "memory" and "consolidation" are not considered again. Order
    of the returned list follows the question, which keeps the result readable;
    the numbered slate is sorted separately.
    """
    toks = _tokens(question)
    n = len(toks)
    taken = [False] * n
    spans: list[dict] = []
    for size in range(min(MAX_NGRAM, n), 0, -1):
        for i in range(0, n - size + 1):
            if any(taken[i:i + size]):
                continue
            words = toks[i:i + size]
            phrase = " ".join(words)
            low = phrase.lower()
            if size == 1:
                if len(low) < 4 or low in STOPWORDS or low in GENERIC_TERMS:
                    continue
            elif all(w.lower() in STOPWORDS for w in words):
                continue
            if words[0].lower() in STOPWORDS or words[-1].lower() in STOPWORDS:
                continue
            desc, variant = _lookup(ix, phrase)
            if desc is None:
                continue
            for k in range(i, i + size):
                taken[k] = True
            spans.append({"span": phrase, "matched_as": variant, "start": i,
                          "size": size, "dui": desc.dui, "label": desc.label})
    spans.sort(key=lambda s: s["start"])
    return spans


def unmatched_phrases(question: str, spans: list[dict]) -> list[str]:
    """Content words the index could not resolve — candidate free-text jargon."""
    toks = _tokens(question)
    covered = set()
    for s in spans:
        covered.update(range(s["start"], s["start"] + s["size"]))
    out: list[str] = []
    for i, t in enumerate(toks):
        low = t.lower()
        if i in covered or low in STOPWORDS or low in GENERIC_TERMS or len(low) < 4:
            continue
        out.append(t)
    return out


def candidate_slate(question: str, ix: MeshIndex, *, include_broader: bool = True,
                    include_phrase: bool = True, max_candidates: int = 60) -> dict:
    """
    Build the numbered candidate slate for hybrid selection.

    Returns
      {"candidates": [{"id", "label", "dui", "span", "relation", "trees"}...],
       "spans": [...], "unmatched": [...]}

    `id` numbering is a pure function of the candidate set (sorted by label then
    DUI), so the same question always yields the same numbering — a prerequisite
    for the model's answer being comparable across runs and models.
    """
    spans = matched_spans(question, ix)
    pool: dict[str, dict] = {}
    for s in spans:
        pool.setdefault(s["dui"], {"dui": s["dui"], "label": s["label"],
                                   "span": s["span"], "relation": "exact",
                                   "start": s["start"], "size": s["size"]})
    if include_phrase:
        for c in phrase_candidates(ix, question, spans):
            pool.setdefault(c["dui"], c)
    if include_broader:
        for s in spans:
            for p in ix.parents(s["dui"]):
                if p in pool:
                    continue
                d = ix.get(p)
                if d is None:
                    continue
                # a broader term inherits its child's span, so it ends up in the
                # same OR group as the phrase it generalises
                pool[p] = {"dui": p, "label": d.label, "span": s["span"],
                           "relation": "broader", "start": s["start"], "size": s["size"]}

    cands = sorted(pool.values(), key=lambda c: (c["label"].lower(), c["dui"]))
    # Broader terms must never crowd out the ones the question actually names.
    if len(cands) > max_candidates:
        rank = {"exact": 0, "phrase": 1, "broader": 2}
        keep = sorted(cands, key=lambda c: (rank.get(c["relation"], 3),
                                            c["label"].lower(), c["dui"]))[:max_candidates]
        cands = sorted(keep, key=lambda c: (c["label"].lower(), c["dui"]))
    for i, c in enumerate(cands, start=1):
        c["id"] = i
        c["trees"] = ix.trees(c["dui"])
    return {"candidates": cands, "spans": spans,
            "unmatched": unmatched_phrases(question, spans)}


def span_groups(slate: dict) -> dict[int, int]:
    """
    Candidate id -> OR-group id, from the question's own structure.

    Two candidates belong in the same OR block when the question words they came
    from overlap: "RNA sequencing" and the "single-cell sequencing" window share
    the word "sequencing", so `Sequence Analysis, RNA` and `Single-Cell Gene
    Expression Analysis` are alternatives for one facet — while "hippocampus" and
    "optogenetic" share nothing and must stay ANDed.

    This takes the grouping decision away from the model, which is the decision it
    is least consistent about: two models selected the *identical* four headings
    for one question and still produced different queries because one ORed
    `Optogenetics` with `Hippocampus`. Grouping is a property of the question, so
    it belongs in the deterministic layer.

    Union-find over word ranges; group ids are the smallest member id, so the
    numbering is a pure function of the slate.
    """
    cands = slate.get("candidates", [])
    parent: dict[int, int] = {c["id"]: c["id"] for c in cands}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in itertools.combinations(cands, 2):
        a_start, a_end = a.get("start", -1), a.get("start", -1) + a.get("size", 0)
        b_start, b_end = b.get("start", -1), b.get("start", -1) + b.get("size", 0)
        if a_start < 0 or b_start < 0:
            continue
        if a_start < b_end and b_start < a_end:      # word ranges intersect
            union(a["id"], b["id"])
    return {c["id"]: find(c["id"]) for c in cands}


def group_closure(slate: dict, selected_ids) -> dict[int, list[dict]]:
    """
    The canonical vocabulary of every facet the selection touched.

    Measured problem: models agree on the FACETS and disagree on which
    near-synonym to name inside one. For "CRISPR-Cas9" the slate offers
    `Clustered Regularly Interspaced Short Palindromic Repeats` (the question's
    literal words), `CRISPR-Cas Systems` and `RNA, Guide, CRISPR-Cas Systems` —
    all in one OR block, so the choice barely changes what PubMed returns but
    changes every byte of the query. Seven models produced five different answers.

    So the model's decision is reduced to the one it is consistent about — is this
    facet required at all — and the vocabulary inside a chosen facet is derived:

      * every `exact` candidate (the question's own words) is included;
      * a `phrase` candidate (same idea, different MeSH wording) is included only
        if its matched window is at least as long as the longest exact window in
        the group — a reworded match must be at least as well-supported as the
        literal one. This is what keeps `Single-Cell Gene Expression Analysis`
        (window "single-cell sequencing") while dropping `Comet Assay` and
        `Amyloid beta-Peptides`, which matched a single word;
      * `broader` candidates only survive in a group that has neither.

    Consequence to be aware of: selecting any member of a facet selects that
    facet's whole canonical vocabulary. The model can drop a facet, not a synonym.
    """
    groups = span_groups(slate)
    by_id = {c["id"]: c for c in slate.get("candidates", [])}
    members: dict[int, list[dict]] = {}
    for c in slate.get("candidates", []):
        members.setdefault(groups[c["id"]], []).append(c)

    out: dict[int, list[dict]] = {}
    for gid in sorted({groups[i] for i in selected_ids if i in by_id}):
        ms = members[gid]
        exact = [m for m in ms if m["relation"] == "exact"]
        phrase = [m for m in ms if m["relation"] == "phrase"]
        if exact:
            floor = max(m.get("size", 0) for m in exact)
            keep = exact + [m for m in phrase if m.get("size", 0) >= floor]
        elif phrase:
            keep = phrase
        else:
            keep = ms
        out[gid] = sorted(keep, key=lambda m: (m["label"].lower(), m["dui"]))
    return out


def mesh_only_blocks(question: str, ix: MeshIndex) -> list[dict]:
    """
    LLM-free baseline: one ANDed block per maximal matched span.

    Fully deterministic and model-independent by construction (cross-model
    agreement is 1.0 trivially), at the cost of the judgement an LLM adds — it
    cannot drop an irrelevant match or add a facet the question only implies.
    Also the fallback when the LLM leg of the hybrid pipeline fails.
    """
    spans = matched_spans(question, ix)
    blocks: list[dict] = []
    for s in spans:
        blocks.append({
            "name": s["span"],
            "slot": "other",
            "mesh": [s["label"]],
            "freetext": [],
            "explode": True,
        })
    return blocks
