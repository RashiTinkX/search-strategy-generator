"""
PubMed retrieval via NCBI E-utilities.

Deterministic: the set of PMIDs matching a fixed query is stable (barring NCBI
index updates). We use the history server (esearch usehistory=y) then page
through efetch to pull *every* matching record — no silent cap.

Rate limits: 10 req/s with an API key, 3 req/s without. We self-throttle.
"""
from __future__ import annotations

import csv
import io
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

CSV_FIELDS = [
    "pmid", "doi", "title", "authors", "journal", "year",
    "pub_types", "mesh_terms", "abstract", "url",
]


@dataclass
class Article:
    pmid: str = ""
    doi: str = ""
    title: str = ""
    abstract: str = ""
    journal: str = ""
    year: str = ""
    authors: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    pub_types: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def to_row(self) -> dict:
        return {
            "pmid": self.pmid,
            "doi": self.doi,
            "title": self.title,
            "authors": "; ".join(self.authors),
            "journal": self.journal,
            "year": self.year,
            "pub_types": "; ".join(self.pub_types),
            "mesh_terms": "; ".join(self.mesh_terms),
            "abstract": self.abstract,
            "url": self.url,
        }


class PubMed:
    def __init__(self, api_key: str = "", email: str = "", tool: str = "deterministic-lit-search"):
        self.api_key = api_key or ""
        self.email = email or ""
        self.tool = tool
        self._min_interval = 1.0 / (10 if self.api_key else 3) + 0.02
        self._last = 0.0
        self._session = requests.Session()

    def _throttle(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self._min_interval:
            time.sleep(self._min_interval - dt)
        self._last = time.monotonic()

    def _params(self, **kw) -> dict:
        p = {"tool": self.tool}
        if self.email:
            p["email"] = self.email
        if self.api_key:
            p["api_key"] = self.api_key
        p.update(kw)
        return p

    def _request(self, path: str, params: dict, method: str = "GET") -> requests.Response:
        for attempt in range(5):
            self._throttle()
            try:
                if method == "POST":
                    r = self._session.post(f"{EUTILS}/{path}", data=params, timeout=60)
                else:
                    r = self._session.get(f"{EUTILS}/{path}", params=params, timeout=60)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("E-utilities request failed after retries")

    # ---- search -------------------------------------------------------

    def search(self, query: str) -> dict:
        """esearch with history server. Returns count, webenv, query_key, translation."""
        params = self._params(db="pubmed", term=query, usehistory="y", retmode="json", retmax=0)
        r = self._request("esearch.fcgi", params, method="POST")
        data = r.json()["esearchresult"]
        return {
            "count": int(data.get("count", 0)),
            "webenv": data.get("webenv", ""),
            "query_key": data.get("querykey", ""),
            "translation": data.get("querytranslation", ""),
            "warnings": data.get("warninglist", {}),
            "errors": data.get("errorlist", {}),
        }

    # ---- fetch --------------------------------------------------------

    def fetch_all(self, webenv: str, query_key: str, count: int,
                  batch: int = 200, max_records: int | None = None,
                  progress=None) -> list[Article]:
        target = count if max_records is None else min(count, max_records)
        out: list[Article] = []
        start = 0
        while start < target:
            n = min(batch, target - start)
            params = self._params(
                db="pubmed", WebEnv=webenv, query_key=query_key,
                retstart=start, retmax=n, retmode="xml",
            )
            r = self._request("efetch.fcgi", params, method="POST")
            out.extend(self._parse(r.content))
            start += n
            if progress:
                progress(min(start, target), target)
        # stable order
        out.sort(key=lambda a: int(a.pmid) if a.pmid.isdigit() else 0)
        return out

    # ---- XML parsing --------------------------------------------------

    @staticmethod
    def _text(el) -> str:
        return "".join(el.itertext()).strip() if el is not None else ""

    @classmethod
    def _parse(cls, xml_bytes: bytes) -> list[Article]:
        root = ET.fromstring(xml_bytes)
        articles: list[Article] = []
        for pa in root.findall(".//PubmedArticle"):
            art = Article()
            medline = pa.find("MedlineCitation")
            if medline is None:
                continue
            art.pmid = cls._text(medline.find("PMID"))
            article = medline.find("Article")
            if article is not None:
                art.title = cls._text(article.find("ArticleTitle"))
                # abstract (may have multiple labeled sections)
                abstracts = article.findall(".//Abstract/AbstractText")
                parts = []
                for a in abstracts:
                    label = a.get("Label")
                    txt = cls._text(a)
                    parts.append(f"{label}: {txt}" if label else txt)
                art.abstract = " ".join(p for p in parts if p)
                journal = article.find("Journal")
                if journal is not None:
                    art.journal = cls._text(journal.find("Title"))
                    y = journal.find(".//JournalIssue/PubDate/Year")
                    if y is not None:
                        art.year = cls._text(y)
                    else:
                        md = journal.find(".//JournalIssue/PubDate/MedlineDate")
                        art.year = cls._text(md)[:4]
                for author in article.findall(".//AuthorList/Author"):
                    last = cls._text(author.find("LastName"))
                    init = cls._text(author.find("Initials"))
                    coll = cls._text(author.find("CollectiveName"))
                    if last:
                        art.authors.append(f"{last} {init}".strip())
                    elif coll:
                        art.authors.append(coll)
                for pt in article.findall(".//PublicationTypeList/PublicationType"):
                    t = cls._text(pt)
                    if t:
                        art.pub_types.append(t)
            for mh in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
                t = cls._text(mh)
                if t:
                    art.mesh_terms.append(t)
            # DOI / ids
            for aid in pa.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if aid.get("IdType") == "doi":
                    art.doi = cls._text(aid)
            if not art.doi:
                for eloc in pa.findall(".//Article/ELocationID"):
                    if eloc.get("EIdType") == "doi":
                        art.doi = cls._text(eloc)
            articles.append(art)
        return articles


def to_csv(articles: list[Article]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    for a in articles:
        w.writerow(a.to_row())
    return buf.getvalue()


def to_jsonl(articles: list[Article]) -> str:
    import json
    return "\n".join(json.dumps(a.to_row(), ensure_ascii=False) for a in articles)
