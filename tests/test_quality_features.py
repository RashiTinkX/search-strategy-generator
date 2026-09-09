"""No-network regression checks for portfolio construction and PubMed safeguards."""
from __future__ import annotations

import unittest
from datetime import date

from backend.app import _portfolio
from backend.pubmed import PubMed


class QualityFeatureTests(unittest.TestCase):
    def test_portfolio_excludes_context_and_keeps_optional_variants(self):
        concepts = [
            {"name": "disease", "mesh": ["Alzheimer Disease"], "freetext": [], "role": "required"},
            {"name": "method", "mesh": ["Sequence Analysis, RNA"], "freetext": [], "role": "optional"},
            {"name": "outcome", "mesh": [], "freetext": ["memory"], "role": "contextual"},
        ]
        result = _portfolio(concepts, {}, strict=False)
        # With exactly one optional facet, primary and sensitivity_1 are the
        # same query. The portfolio intentionally keeps only unique hashes.
        self.assertEqual([v["name"] for v in result["variants"]], ["primary", "core"])
        self.assertNotIn('"memory"', result["primary"]["query"])
        self.assertIn("outcome", result["contextual_concepts"])

    def test_date_scope_is_disjoint_at_boundaries(self):
        query = '"Microglia"[MeSH Terms]'
        left = PubMed._dated_query(query, date(2020, 1, 1), date(2020, 6, 30))
        right = PubMed._dated_query(query, date(2020, 7, 1), date(2020, 12, 31))
        self.assertIn('"2020/06/30"', left)
        self.assertIn('"2020/07/01"', right)


if __name__ == "__main__":
    unittest.main()
