"""
Europe PMC retrieval — the multi-database half of "exhaustive".

The upstream sensein/search-strategy-generator tool is explicit that it is
PubMed-only ("nothing here queries Crossref, OpenAlex, Europe PMC or
Unpaywall today" — its own README). A search restricted to one database is a
known, well-documented threat to systematic-review completeness (Cochrane and
PRISMA both require multiple sources); this module is the fix, not a
reimplementation of anything upstream has.

Europe PMC's REST API is free, keyless, and its query syntax has close
analogues to PubMed's field tags, so the same Concept/Filters objects
query_builder.py already builds can be translated deterministically (see
`translate_query`) rather than re-derived. Retrieval uses cursor-based
pagination (`cursorMark=*`, following `nextCursorMark`), which — unlike
PubMed's history server — has no hard 9,999-record ceiling, so no
date-bisection is needed here.

Scope is honest, not silent: `translate_query` maps MeSH headings, free-text
terms, publication-date ranges and language. Publication-type and species
filters are NOT yet translated (Europe PMC's PUB_TYPE vocabulary does not
line up 1:1 with PubMed's) — callers should treat an EuropePMC-sourced result
set as unfiltered on those two axes until this is extended.
"""
from __future__ import annotations

import time

import requests

from .pubmed import Article
from .query_builder import Concept, Filters, _dedupe, _q

REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PAGE_SIZE = 1000


def translate_query(concepts: list[Concept], filters: Filters) -> str:
    """Deterministic PubMed-Concept/Filters -> Europe PMC query string."""
    blocks = []
    for c in concepts:
        parts = []
        for h in _dedupe(c.mesh):
            parts.append(f'MESH:{_q(h)}')
        for t in _dedupe(c.freetext):
            parts.append(f'(TITLE:{_q(t)} OR ABSTRACT:{_q(t)})')
        if parts:
            blocks.append("(" + " OR ".join(parts) + ")")
    if not blocks:
        raise ValueError("No searchable concept blocks for Europe PMC translation.")
    query = " AND ".join(blocks)

    if filters.date_from or filters.date_to:
        lo = (filters.date_from or "1000").replace("/", "-")
        hi = (filters.date_to or "3000").replace("/", "-")
        query = f'({query}) AND FIRST_PDATE:[{lo} TO {hi}]'
    langs = _dedupe(filters.languages)
    if langs:
        # Europe PMC uses ISO 639-1/2 lowercase codes; PubMed's [la] values
        # ("English") don't match — pass through as-is and let the caller
        # supply EuropePMC-style codes if they want this filter honoured.
        query = f'({query}) AND (' + " OR ".join(f'LANG:{_q(l)}' for l in langs) + ")"
    if filters.custom_include:
        query = f"({query}) AND ({filters.custom_include})"
    ex_terms = _dedupe(filters.exclude_terms)
    if ex_terms:
        query = f'({query}) NOT (' + " OR ".join(
            f'(TITLE:{_q(t)} OR ABSTRACT:{_q(t)})' for t in ex_terms) + ")"
    if filters.custom_exclude:
        query = f"({query}) NOT ({filters.custom_exclude})"
    return query


class EuropePMC:
    def __init__(self, email: str = "", tool: str = "deterministic-lit-search"):
        self.email = email
        self.tool = tool
        self._session = requests.Session()
        self._min_interval = 0.34  # EBI has no published hard cap; be polite (~3 req/s)
        self._last = 0.0

    def _throttle(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self._min_interval:
            time.sleep(self._min_interval - dt)
        self._last = time.monotonic()

    def _get(self, params: dict) -> dict:
        for attempt in range(5):
            self._throttle()
            r = self._session.get(f"{REST}/search", params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Europe PMC request failed after retries")

    def count(self, query: str) -> int:
        data = self._get({"query": query, "format": "json", "pageSize": 1})
        return int(data.get("hitCount", 0))

    @staticmethod
    def _to_article(rec: dict) -> Article:
        art = Article()
        art.pmid = str(rec.get("pmid", "") or "")
        art.doi = str(rec.get("doi", "") or "")
        art.title = str(rec.get("title", "") or "")
        art.abstract = str(rec.get("abstractText", "") or "")
        art.journal = str(rec.get("journalTitle", "") or "")
        art.year = str(rec.get("pubYear", "") or "")
        authors = rec.get("authorString", "")
        art.authors = [a.strip() for a in authors.split(",") if a.strip()] if authors else []
        art.pub_types = list(rec.get("pubTypeList", {}).get("pubType", []) or [])
        art.mesh_terms = [m.get("descriptorName", "") for m in
                          (rec.get("meshHeadingList", {}) or {}).get("meshHeading", [])
                          if m.get("descriptorName")]
        return art

    def fetch_all(self, query: str, max_records: int | None = None,
                  page_size: int = PAGE_SIZE, progress=None) -> dict:
        """Cursor-paginate every result. No 9,999 ceiling here (unlike PubMed's
        history server), so no date bisection is required."""
        total = self.count(query)
        articles: list[Article] = []
        cursor = "*"
        while True:
            if max_records is not None and len(articles) >= max_records:
                break
            params = {"query": query, "format": "json", "resultType": "core",
                      "pageSize": min(page_size, 1000), "cursorMark": cursor}
            data = self._get(params)
            results = data.get("resultList", {}).get("result", [])
            if not results:
                break
            for rec in results:
                articles.append(self._to_article(rec))
                if max_records is not None and len(articles) >= max_records:
                    break
            if progress:
                progress(len(articles), total)
            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return {"articles": articles, "count": total, "fetched": len(articles),
                "capped": max_records is not None and total > max_records}
