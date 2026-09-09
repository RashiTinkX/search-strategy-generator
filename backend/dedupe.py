"""
Cross-database deduplication + PRISMA-style identification counts.

Merges PubMed and Europe PMC result sets into one deduplicated list, and
reports the counts a PRISMA flow diagram needs (records identified per
source, duplicates removed, records remaining) — turning "we searched two
databases" into an auditable number, the same way protocol.json turns "we
searched" into a hashed, re-runnable query.

Matching is deterministic and tiered, cheapest/most-certain first:
  1. DOI (normalized: lowercased, no "https://doi.org/" prefix, trailing
     slash stripped) — the strongest signal when present on both sides.
  2. PMID — Europe PMC carries the PubMed PMID for indexed records, so a
     record fetched from both sources on the PubMed track usually matches
     here even without a DOI.
  3. Normalized title + publication year — the fallback for preprints or
     records missing both of the above; title is lowercased, punctuation and
     whitespace collapsed, so trivial formatting differences don't create
     false negatives to break the match, and year is required alongside it
     to avoid over-merging distinct records that share a common title.

Every merge tier is logged per pair so `protocol.json` can show exactly which
rule matched two records, rather than asserting deduplication happened.

CAVEAT (found while testing this module against live PubMed + Europe PMC):
comparing two SOURCES CAPPED at the same `max_records` is not meaningful —
PubMed's default esearch order and Europe PMC's default relevance order are
different orderings over different-sized universes, so two same-sized capped
samples can show near-zero overlap even when the full result sets overlap
substantially. A live check on a 2-day, MeSH-scoped window found 10/20
Europe PMC records were PubMed duplicates when BOTH sides were fetched
exhaustively (`max_records=None`), vs 0/first-200-of-1536 when both were
capped. Only compare exhaustive fetches, or treat a capped multi-source merge
as a lower-bound duplicate count, not an accurate one — `per_source["capped"]`
in the API response tells the caller which case they're in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pubmed import Article

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def _norm_doi(doi: str) -> str:
    d = (doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.rstrip("/")


def _norm_title(title: str) -> str:
    t = _PUNCT_RE.sub("", (title or "").lower())
    return _WS_RE.sub(" ", t).strip()


@dataclass
class MergeResult:
    articles: list[Article]
    sources: dict[str, list[str]] = field(default_factory=dict)   # pmid|doi|title-key -> [source names]
    matched_by: list[dict] = field(default_factory=list)          # audit trail
    identified: dict[str, int] = field(default_factory=dict)      # source -> count before merge
    duplicates_removed: int = 0

    def prisma_counts(self) -> dict:
        total_identified = sum(self.identified.values())
        return {
            "identified_by_source": dict(self.identified),
            "total_identified": total_identified,
            "duplicates_removed": self.duplicates_removed,
            "unique_records": len(self.articles),
        }


def merge_sources(named_results: list[tuple[str, list[Article]]]) -> MergeResult:
    """named_results: [("pubmed", [Article, ...]), ("europepmc", [...]), ...]"""
    result = MergeResult(articles=[])
    doi_index: dict[str, int] = {}
    pmid_index: dict[str, int] = {}
    title_index: dict[str, int] = {}
    result.identified = {name: len(arts) for name, arts in named_results}

    for name, articles in named_results:
        for art in articles:
            doi_key = _norm_doi(art.doi)
            pmid_key = art.pmid.strip()
            title_key = (_norm_title(art.title), art.year.strip()) if art.title else None

            idx = None
            rule = None
            if doi_key and doi_key in doi_index:
                idx, rule = doi_index[doi_key], "doi"
            elif pmid_key and pmid_key in pmid_index:
                idx, rule = pmid_index[pmid_key], "pmid"
            elif title_key and title_key in title_index:
                idx, rule = title_index[title_key], "title+year"

            if idx is not None:
                result.duplicates_removed += 1
                existing = result.articles[idx]
                # prefer the more complete record on the fields we display
                if not existing.doi and art.doi:
                    existing.doi = art.doi
                if not existing.abstract and art.abstract:
                    existing.abstract = art.abstract
                if not existing.pmid and art.pmid:
                    existing.pmid = art.pmid
                result.matched_by.append({"rule": rule, "source": name,
                                          "kept_pmid": existing.pmid, "kept_doi": existing.doi})
                continue

            result.articles.append(art)
            new_idx = len(result.articles) - 1
            if doi_key:
                doi_index[doi_key] = new_idx
            if pmid_key:
                pmid_index[pmid_key] = new_idx
            if title_key:
                title_index[title_key] = new_idx

    result.articles.sort(key=lambda a: (a.pmid.isdigit() is False, int(a.pmid) if a.pmid.isdigit() else 0,
                                        a.title))
    return result
