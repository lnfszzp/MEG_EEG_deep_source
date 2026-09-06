import inspect
import unittest

import numpy as np
from scipy import sparse

from algorithms.external_metrics import external_full_head_metrics
from metrics.user_metrics import An_auc, An_cal_AUC, DLE_an
from metrics.user_metrics.An_cal_AUC import _grow_parcels
from metrics.user_metrics._common import mesh_adjacency


class UserMetricsRegressionTest(unittest.TestCase):
    def test_an_auc_keeps_canonical_api(self):
        self.assertEqual(
            str(inspect.signature(An_auc)),
            "(data, alpha=0.05, flag='logit', nboot=1000, return_ci=False, rng=None)",
        )
        self.assertIs(inspect.signature(DLE_an).parameters["index_base"].default, inspect.Parameter.empty)

    def test_auc_perfect_reversed_and_tie_order_invariance(self):
        perfect = np.array([[0, 0.1], [0, 0.2], [1, 0.8], [1, 0.9]])
        reversed_scores = perfect.copy()
        reversed_scores[:, 1] = 1 - reversed_scores[:, 1]
        tied = np.array([[1, 0.5], [0, 0.5], [1, 0.5], [0, 0.5]])

        self.assertAlmostEqual(An_auc(perfect), 1.0)
        self.assertAlmostEqual(An_auc(reversed_scores), 0.0)
        self.assertAlmostEqual(An_auc(tied), 0.5)
        self.assertAlmostEqual(An_auc(tied), An_auc(tied[::-1]))

    def test_layer_metrics_split_at_n_surface_without_index_guessing(self):
        n_surface = 3
        positions = np.array(
            [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0], [11.0, 0, 0], [12.0, 0, 0]]
        )
        source = np.array([[0.0], [0.0], [1.0], [0.0], [1.0], [0.0]])
        result = external_full_head_metrics(
            source,
            source,
            positions,
            true_groups=[np.array([2]), np.array([4])],
            n_surf=n_surface,
        )

        for layer in ("surface", "deep"):
            self.assertAlmostEqual(result[f"{layer}_auc"], 1.0)
            self.assertAlmostEqual(result[f"{layer}_rmse"], 0.0)
            self.assertAlmostEqual(result[f"{layer}_sd_mm"], 0.0)
            self.assertAlmostEqual(result[f"{layer}_dle_mm"], 0.0)

        deep_positions = positions[n_surface:]
        deep_source = source[n_surface:]
        self.assertAlmostEqual(
            DLE_an(deep_source, [np.array([2])], deep_positions, index_base=1), 0.0
        )

    def test_mesh_adjacency_formats_and_small_disconnected_parcels(self):
        vertices = np.zeros((4, 3))
        sparse_graph = sparse.eye(4, format="csr")
        dense_graph = np.eye(4)
        triangles = np.array([[0, 1, 2], [0, 2, 3]])

        self.assertEqual((mesh_adjacency(vertices, sparse_graph) != sparse_graph).nnz, 0)
        self.assertEqual((mesh_adjacency(vertices, dense_graph) != sparse_graph).nnz, 0)
        triangle_graph = mesh_adjacency(vertices, triangles)
        self.assertTrue(triangle_graph[0, 1] and triangle_graph[1, 0])

        paired = sparse.csr_matrix(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
        )
        parcels = _grow_parcels(paired, [0, 2], [], source_count=1, count=5)
        self.assertEqual(len(parcels), 5)
        self.assertTrue(all(len(parcel) >= 2 for parcel in parcels))
        with self.assertRaisesRegex(ValueError, "cannot grow"):
            _grow_parcels(sparse.eye(2, format="csr"), [0, 1], [], source_count=1, count=1)
        with self.assertRaisesRegex(ValueError, "cannot grow"):
            _grow_parcels(paired, [0, 2], [], source_count=2, count=1)

    def test_an_cal_auc_runs_on_sparse_graph_with_small_seed_pools(self):
        count = 30
        graph = sparse.diags(
            [np.ones(count - 1), np.ones(count), np.ones(count - 1)],
            offsets=[-1, 0, 1],
            format="csr",
        )
        truth = np.zeros((count, 1))
        truth[0] = 1
        cortex = {"Vertices": np.zeros((count, 3)), "Faces": graph}

        self.assertAlmostEqual(An_cal_AUC(truth, truth, cortex, threshold=0.01), 1.0)


if __name__ == "__main__":
    unittest.main()
