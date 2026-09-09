"""
No-network regression checks for the facet/closure/multi-source pipeline
(facets.py, closure.py, dedupe.py, europepmc.py's pure query translation).

Mirrors the style of test_quality_features.py: fast, deterministic, no API
keys required. closure.py tests use the real local MeSH index (data/mesh.sqlite)
since that is itself a static, offline fixture -- no network call happens.
"""
from __future__ import annotations

import unittest

from backend import closure, dedupe, europepmc, facets, query_builder
from backend.mesh_index import get_index
from backend.pubmed import Article


class HeuristicFacetTests(unittest.TestCase):
    def test_splits_on_conjunctions_and_guesses_roles(self):
        q = "Effect of RNA sequencing on hippocampal neurons in Alzheimer Disease patients"
        facets_out = facets.heuristic_facets(q)
        names = [f["name"] for f in facets_out]
        self.assertTrue(any("RNA sequencing" in n for n in names))
        self.assertTrue(any("Alzheimer Disease" in n for n in names))

    def test_deterministic_across_runs(self):
        q = "Effect of metformin on cardiovascular outcomes in type 2 diabetes patients"
        first = facets.heuristic_facets(q)
        for _ in range(5):
            self.assertEqual(facets.heuristic_facets(q), first)


class VotingTests(unittest.TestCase):
    def test_majority_span_wins_over_minority(self):
        # 2 of 3 runs agree on the same (start,end); the 3rd run disagrees
        # slightly -- majority should win, and agreement count is exact.
        runs = [
            [{"name": "RNA sequencing", "role": "method", "start": 5, "end": 6}],
            [{"name": "RNA sequencing", "role": "method", "start": 5, "end": 6}],
            [{"name": "the RNA sequencing step", "role": "method", "start": 4, "end": 7}],
        ]
        voted = facets._cluster_and_vote(runs, min_runs_agree=2)
        self.assertEqual(len(voted), 1)
        self.assertEqual((voted[0]["start"], voted[0]["end"]), (5, 6))
        self.assertEqual(voted[0]["agreement"], 3)  # all 3 runs overlap this span

    def test_below_threshold_cluster_is_dropped(self):
        runs = [
            [{"name": "a", "role": "other", "start": 0, "end": 0}],
            [],
            [],
        ]
        voted = facets._cluster_and_vote(runs, min_runs_agree=2)
        self.assertEqual(voted, [])

    def test_role_normalizer_rejects_echoed_enum(self):
        self.assertEqual(facets._normalize_pico_role(
            "population|intervention|comparator|outcome|method|context"), "other")
        self.assertEqual(facets._normalize_pico_role("Population"), "population")
        self.assertEqual(facets._normalize_pico_role("garbage"), "other")


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.ix = get_index()

    def test_generic_standalone_term_excluded(self):
        q = "Alzheimer Disease patients"
        facet = {"name": q, "role": "population", "start": 0, "end": 2}
        block = closure.build_facet(self.ix, q, facet)
        labels = [m["options"][0]["label"] for m in block["mesh"]]
        self.assertIn("Alzheimer Disease", labels)
        self.assertNotIn("Patients", labels)  # generic-term guard

    def test_subsumption_pruning_keeps_broader_only(self):
        d_broad = self.ix.exact("Nervous System")
        d_narrow = self.ix.exact("Neurons")
        if not d_broad or not d_narrow:
            self.skipTest("index fixture missing expected descriptors")
        kept = closure.prune_subsumed(d_broad + d_narrow)
        kept_labels = {d.label for d in kept}
        if closure._is_descendant(d_narrow[0], d_broad[0]):
            self.assertNotIn("Neurons", kept_labels)
            self.assertIn("Nervous System", kept_labels)

    def test_deterministic_repeat(self):
        q = "RNA sequencing of hippocampal neurons"
        facet = {"name": q, "role": "method", "start": 0, "end": 5}
        first = closure.build_facet(self.ix, q, facet)
        for _ in range(3):
            again = closure.build_facet(self.ix, q, facet)
            self.assertEqual(first["mesh"], again["mesh"])
            self.assertEqual(first["freetext"], again["freetext"])


class DedupeTests(unittest.TestCase):
    def test_doi_match_normalizes_prefix_and_case(self):
        a = Article(pmid="1", doi="10.1/ABC", title="T", year="2020")
        b = Article(pmid="", doi="https://doi.org/10.1/abc", title="Different", year="2020")
        merged = dedupe.merge_sources([("pubmed", [a]), ("europepmc", [b])])
        self.assertEqual(merged.prisma_counts()["unique_records"], 1)
        self.assertEqual(merged.duplicates_removed, 1)
        self.assertEqual(merged.matched_by[0]["rule"], "doi")

    def test_title_year_fallback_when_no_doi_or_pmid(self):
        a = Article(pmid="", doi="", title="A Study of Things!", year="2019")
        b = Article(pmid="", doi="", title="a study of things", year="2019")
        c = Article(pmid="", doi="", title="a study of things", year="2020")  # different year
        merged = dedupe.merge_sources([("s1", [a]), ("s2", [b, c])])
        self.assertEqual(merged.prisma_counts()["unique_records"], 2)

    def test_no_false_merge_across_unrelated_records(self):
        a = Article(pmid="1", doi="10.1/x", title="First", year="2020")
        b = Article(pmid="2", doi="10.1/y", title="Second", year="2021")
        merged = dedupe.merge_sources([("pubmed", [a]), ("europepmc", [b])])
        self.assertEqual(merged.duplicates_removed, 0)
        self.assertEqual(merged.prisma_counts()["unique_records"], 2)


class EuropePMCTranslateTests(unittest.TestCase):
    def test_translate_is_deterministic_and_covers_filters(self):
        concepts = [query_builder.Concept("c", mesh=["Alzheimer Disease"], freetext=["AD"])]
        filters = query_builder.Filters(date_from="2020", date_to="2021")
        q1 = europepmc.translate_query(concepts, filters)
        q2 = europepmc.translate_query(concepts, filters)
        self.assertEqual(q1, q2)
        self.assertIn('MESH:"Alzheimer Disease"', q1)
        self.assertIn("FIRST_PDATE:[2020 TO 2021]", q1)

    def test_empty_concepts_raise(self):
        with self.assertRaises(ValueError):
            europepmc.translate_query([query_builder.Concept("c")], query_builder.Filters())


if __name__ == "__main__":
    unittest.main()
