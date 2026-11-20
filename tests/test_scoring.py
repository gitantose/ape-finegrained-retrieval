"""Known-answer checks against pinned upstream operations; no model download."""

import os
import unittest

import numpy as np
import torch
from ape import APEScore
from ape.propsearch.benchmark import _multi_vector_ranks, _rank, _retrieval_metrics

from ape_retrieval.model import configure_reference


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(os.environ.get("APE_TEST_DEVICE", "cpu"))
        configure_reference({"dtype": "float32", "tf32": False,
                             "seed": 42, "cpu_threads": 4}, str(cls.device))
        torch.set_default_device(cls.device)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_device("cpu")

    def test_h2p_direction_normalization_and_permutation(self):
        query = torch.tensor([[2., 0.]])
        candidate = torch.tensor([[3., 0.], [0., 4.]])
        self.assertEqual(APEScore(query, candidate).h_min_to_p(), 1.)
        self.assertEqual(APEScore(candidate, query).h_min_to_p(), 0.)
        self.assertEqual(APEScore(query, candidate.flip(0)).h_min_to_p(), 1.)
        self.assertEqual(APEScore(torch.empty(0, 2), candidate).h_min_to_p(), 0.)

    def test_exact_ties_and_adjacent_float(self):
        one = torch.tensor(1., dtype=torch.float32)
        above = torch.nextafter(one, torch.tensor(float("inf")))
        scores = torch.stack([one, one, above, torch.tensor(0.)])[None]
        self.assertEqual(_rank(scores, torch.tensor([0])).item(), 2)

    def test_ha_maxsim_and_h2p_have_distinct_reductions(self):
        query, document = torch.eye(2), torch.tensor([[3., 0.]])
        scorer = APEScore(query, document)
        self.assertEqual(scorer.hungarian(), 1.)
        self.assertEqual(scorer.maxsim(), .75)
        self.assertEqual(scorer.h_min_to_p(), 0.)
        reverse = APEScore(document, query)
        self.assertEqual(reverse.hungarian(), scorer.hungarian())
        self.assertEqual(reverse.maxsim(), scorer.maxsim())
        self.assertEqual(reverse.h_min_to_p(), 1.)

    def test_hungarian_global_assignment_and_symmetric_permutation(self):
        query = torch.eye(2)
        document = torch.tensor([[.8, .6], [.6, -.8]])
        # Optimal assignment pairs q0 with d1 and q1 with d0, not greedy q0-d0.
        scorer = APEScore(query, document)
        self.assertAlmostEqual(scorer.hungarian(), .6, places=6)
        for rule in ("hungarian", "maxsim"):
            expected = scorer.score(rule)
            self.assertAlmostEqual(APEScore(document, query).score(rule), expected)
            self.assertAlmostEqual(APEScore(query.flip(0), document.flip(0)).score(rule), expected)
            self.assertEqual(APEScore(torch.empty(0, 2), document).score(rule), 0.)

    def test_variable_slots_duplicate_positions_and_known_ranks(self):
        queries = {"q": torch.tensor([[1., 0.], [0., 1.]])}
        corpus = {"partial": torch.tensor([[1., 0.]]),
                  "complete": torch.eye(2),
                  "opposite": -torch.eye(2)}
        positions = ["partial", "complete", "complete", "opposite"]
        scores = torch.tensor([[APEScore(queries["q"], corpus[c]).h_min_to_p()
                                for c in positions]])
        self.assertEqual(scores.tolist(), [[0., 1., 1., 0.]])
        for gold, expected in [(0, 3), (1, 1), (2, 1), (3, 3)]:
            ranks = _multi_vector_ranks(
                queries, ["q"], corpus, positions, torch.tensor([gold]),
                self.device, chunk_size=1, all_rules=False,
            )["h_min_to_p"]
            np.testing.assert_array_equal(ranks, [expected])
        metrics = _retrieval_metrics(np.array([1, 3]))
        self.assertAlmostEqual(metrics["MRR"], 2 / 3)
        self.assertEqual(metrics["R@1"], 50.)
        self.assertEqual(metrics["R@5"], 100.)


if __name__ == "__main__":
    unittest.main()
