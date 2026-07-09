import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protected_multilayer import (
    adaptive_sisses_threshold,
    build_candidate_mask,
    component_refit_select_v3,
    component_refit_select_v4_deep_rescue,
    component_refit_select_v5_compact_deep_prior,
    component_refit_select_v6_sisses_refit,
    component_refit_select,
    evidence_aware_compact_mask,
    graph_hop_mask,
    numpy_knn_expand_deep,
    residual_deep_scores,
    residual_deep_timecourse,
    ridge_refit,
    ridge_refit_system,
    sisses_style_refit_system,
    shrink_surface_core,
    source_amplitude,
)


class ProtectedMultilayerTests(unittest.TestCase):
    def test_candidate_mask_uses_meg_surface_and_joint_deep(self):
        n_surf = 6
        n_sources = 9
        vert_conn = np.eye(n_sources)
        for a, b in [(0, 1), (1, 2), (3, 4), (4, 5)]:
            vert_conn[a, b] = 1
            vert_conn[b, a] = 1

        joint = np.zeros((n_sources, 4))
        joint[7] = 10.0
        joint[0] = 9.0
        joint[5] = 0.8

        meg = np.zeros_like(joint)
        meg[0] = 10.0
        meg[1] = 8.0
        meg[5] = 1.0

        masks = build_candidate_mask(joint, meg, vert_conn, n_surf)
        self.assertTrue(masks["final"][0])
        self.assertTrue(masks["final"][1])
        self.assertFalse(masks["final"][5])
        self.assertTrue(masks["final"][7])

    def test_ridge_refit_recovers_simple_support(self):
        t = np.linspace(0, 1, 12)
        true_wave = np.sin(2 * np.pi * t)
        gain = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.2], [1.0, 1.0, 0.0]])
        source = np.zeros((3, t.size))
        source[1] = true_wave
        data = gain @ source
        eeg = {"F": data, "Gain": gain}
        meg = {"F": data, "Gain": gain}
        support = np.array([False, True, False])
        estimated = ridge_refit(eeg, meg, support, ridge_fraction=1e-8)
        corr = np.corrcoef(estimated[1], true_wave)[0, 1]
        self.assertGreater(corr, 0.999)
        self.assertEqual(np.count_nonzero(np.linalg.norm(estimated, axis=1)), 1)

    def test_ridge_refit_system_can_use_prior(self):
        b = np.zeros((2, 3))
        l = np.eye(2)
        support = np.array([True, False])
        prior = np.zeros((2, 3))
        prior[0] = 4.0
        estimated = ridge_refit_system(b, l, support, ridge_fraction=10.0, prior=prior)
        self.assertGreater(estimated[0, 0], 3.0)
        self.assertEqual(estimated[1, 0], 0.0)

    def test_adaptive_sisses_threshold_uses_active_count_only(self):
        source = np.zeros((120, 3))
        source[:120] = 1.0
        self.assertEqual(adaptive_sisses_threshold(source), 0.115)

        source = np.zeros((60, 3))
        source[:60] = 1.0
        self.assertEqual(adaptive_sisses_threshold(source), 0.105)

        source = np.zeros((30, 3))
        source[:30] = 1.0
        self.assertEqual(adaptive_sisses_threshold(source), 0.110)

    def test_evidence_compact_only_compacts_mixed_layer_maps(self):
        source = np.zeros((8, 3))
        source[0:4] = 1.0
        source[6:8] = 0.8
        vert_conn = np.eye(8)
        vert_conn[0, 1] = vert_conn[1, 0] = 1
        vert_conn[2, 3] = vert_conn[3, 2] = 1
        mask = evidence_aware_compact_mask(source, vert_conn, 5)
        self.assertLess(mask.sum(), 6)
        self.assertTrue(mask[6] or mask[7])

    def test_source_amplitude_uses_active_window(self):
        source = np.zeros((2, 220))
        source[0, :200] = 10.0
        source[0, 200:] = 10.0
        source[1, 200:] = 2.0
        amp = source_amplitude(source)
        self.assertEqual(amp[0], 0.0)
        self.assertEqual(amp[1], 2.0)

    def test_component_refit_select_keeps_useful_component(self):
        t = np.linspace(0, 1, 20)
        wave = np.sin(2 * np.pi * t)
        gain = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.5, 0.2, 1.0]])
        source = np.zeros((3, t.size))
        source[1] = wave
        data = gain @ source
        sisses = source.copy()
        vert_conn = np.eye(3)
        fitted, support = component_refit_select(
            {"F": data, "Gain": gain},
            {"F": data, "Gain": gain},
            sisses,
            vert_conn,
            2,
            loose_rel=0.05,
            complexity_weight=1.0,
        )
        self.assertTrue(support[1])
        self.assertGreater(np.corrcoef(fitted[1], wave)[0, 1], 0.99)

    def test_graph_hop_mask_shrinks_to_two_hops_inside_component(self):
        adjacency = np.eye(6)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
            adjacency[a, b] = adjacency[b, a] = 1
        allowed = np.ones(6, dtype=bool)
        mask = graph_hop_mask(2, adjacency, allowed, hops=2)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0, 1, 2, 3, 4])

    def test_component_refit_v3_can_relocate_deep_neighbor(self):
        t = np.linspace(0, 1, 24)
        wave = np.sin(2 * np.pi * t)
        gain = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.1],
            ]
        )
        source = np.zeros((4, t.size))
        source[3] = wave
        data = gain @ source
        sisses = np.zeros_like(source)
        sisses[2] = 2.0 * wave
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0.001, 0, 0]], dtype=float)
        fitted, support = component_refit_select_v3(
            {"F": data, "Gain": gain},
            {"F": data, "Gain": gain},
            sisses,
            np.eye(4),
            2,
            vertices,
            loose_rel=0.05,
            deep_k=2,
        )
        self.assertTrue(support[3])
        self.assertGreater(np.corrcoef(fitted[3], wave)[0, 1], 0.99)

    def test_residual_deep_scores_prefers_matching_deep_field(self):
        residual = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
        leadfield = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        scores = residual_deep_scores(residual, leadfield, n_surf=2)
        self.assertGreater(scores[0], scores[1])

    def test_residual_deep_timecourse_recovers_projection(self):
        wave = np.array([1.0, -2.0, 3.0])
        leadfield = np.array([[0.0, 2.0], [0.0, 0.0]])
        residual = leadfield[:, [1]] @ wave[None, :]
        estimated = residual_deep_timecourse(residual, leadfield, n_surf=1, deep_id=0)
        self.assertTrue(np.allclose(estimated, wave))

    def test_shrink_surface_core_keeps_peak_limited_core(self):
        fitted = np.zeros((6, 220))
        fitted[:, 200:] = np.array([[1.0], [0.8], [0.6], [0.2], [0.1], [0.05]])
        support = np.ones(6, dtype=bool)
        vert_conn = np.eye(6)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
            vert_conn[a, b] = vert_conn[b, a] = 1
        shrunk = shrink_surface_core(support, fitted, vert_conn, n_surf=6, core_rel=0.5, min_keep=2, max_keep=3)
        self.assertEqual(np.flatnonzero(shrunk).tolist(), [0, 1, 2])

    def test_numpy_knn_expand_deep_adds_nearby_deep_points(self):
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0.001, 0, 0],
                [1.0, 0, 0],
            ],
            dtype=float,
        )
        expanded = numpy_knn_expand_deep(np.array([0]), vertices, n_surf=2, deep_k=2)
        self.assertEqual(expanded.tolist(), [0, 1])

    def test_component_refit_v4_rescues_deep_from_surface_residual(self):
        t = np.linspace(0, 1, 30)
        surface_wave = np.sin(2 * np.pi * t)
        deep_wave = np.cos(2 * np.pi * t)
        gain = np.array(
            [
                [1.0, 0.0, 0.4],
                [0.0, 1.0, 1.0],
                [0.2, 0.0, 0.3],
            ]
        )
        truth = np.zeros((3, t.size))
        truth[0] = surface_wave
        truth[2] = deep_wave
        data = gain @ truth
        sisses = np.zeros_like(truth)
        sisses[0] = surface_wave
        vertices = np.array([[0, 0, 0], [0.001, 0, 0], [0, 0, 0]], dtype=float)
        fitted, support = component_refit_select_v4_deep_rescue(
            {"F": data, "Gain": gain},
            {"F": data, "Gain": gain},
            sisses,
            np.eye(3),
            2,
            vertices,
            loose_rel=0.05,
            deep_rescue_top=1,
            deep_k=1,
        )
        self.assertTrue(support[0])
        self.assertTrue(support[2])
        self.assertGreater(np.corrcoef(fitted[2], deep_wave)[0, 1], 0.95)

    def test_component_refit_v4_does_not_add_deep_without_residual_gain(self):
        t = np.linspace(0, 1, 30)
        wave = np.sin(2 * np.pi * t)
        gain = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.2], [0.2, 0.0, 1.0]])
        source = np.zeros((3, t.size))
        source[0] = wave
        data = gain @ source
        vertices = np.array([[0, 0, 0], [0.001, 0, 0], [0, 0, 0]], dtype=float)
        _fitted, support = component_refit_select_v4_deep_rescue(
            {"F": data, "Gain": gain},
            {"F": data, "Gain": gain},
            source,
            np.eye(3),
            2,
            vertices,
            loose_rel=0.05,
            deep_rescue_top=1,
            deep_k=1,
        )
        self.assertTrue(support[0])
        self.assertFalse(support[2])

    def test_component_refit_v5_uses_residual_prior_for_rescued_deep(self):
        t = np.linspace(0, 1, 30)
        surface_wave = np.sin(2 * np.pi * t)
        deep_wave = np.cos(2 * np.pi * t)
        gain = np.array([[1.0, 0.0, 0.3], [0.0, 1.0, 1.0], [0.1, 0.0, 0.5]])
        truth = np.zeros((3, t.size))
        truth[0] = surface_wave
        truth[2] = deep_wave
        data = gain @ truth
        sisses = np.zeros_like(truth)
        sisses[0] = surface_wave
        vertices = np.array([[0, 0, 0], [0.001, 0, 0], [0, 0, 0]], dtype=float)
        fitted, support = component_refit_select_v5_compact_deep_prior(
            {"F": data, "Gain": gain},
            {"F": data, "Gain": gain},
            sisses,
            np.eye(3),
            2,
            vertices,
            deep_rescue_top=1,
            deep_k=1,
            max_deep_points=1,
        )
        self.assertTrue(support[2])
        self.assertGreater(source_amplitude(fitted)[2], 0.2)

    def test_sisses_style_refit_system_suppresses_weak_tail(self):
        t = np.linspace(0, 1, 40)
        wave = np.sin(2 * np.pi * t)
        leadfield = np.eye(3)
        truth = np.zeros((3, t.size))
        truth[0] = wave
        data = leadfield @ truth
        support = np.array([True, True, False])
        prior = np.zeros_like(truth)
        prior[0] = wave
        prior[1] = 0.2 * wave
        vert_conn = np.eye(3)
        vert_conn[0, 1] = vert_conn[1, 0] = 1
        fitted = sisses_style_refit_system(
            data,
            leadfield,
            support,
            vert_conn,
            n_surf=2,
            prior=prior,
            lambda_amp=0.05,
            lambda_edge=0.05,
            lambda_prior=0.01,
            max_iter=40,
            weight_iter=2,
        )
        amp = source_amplitude(fitted, noise_samples=0)
        self.assertGreater(amp[0], 0.5)
        self.assertLess(amp[1], 0.1)

    def test_component_refit_v6_returns_three_sparse_variants(self):
        t = np.linspace(0, 1, 30)
        wave = np.sin(2 * np.pi * t)
        gain = np.eye(3)
        source = np.zeros((3, t.size))
        source[0] = wave
        data = gain @ source
        vertices = np.zeros((3, 3))
        variants = component_refit_select_v6_sisses_refit(
            {"F": data, "Gain": gain},
            {"F": data, "Gain": gain},
            source,
            np.eye(3),
            2,
            vertices,
            admm_iters=10,
            max_weight_itr=1,
        )
        self.assertEqual(set(variants), {"weak", "mid", "strong"})
        self.assertTrue(all(variants[name][1][0] for name in variants))


if __name__ == "__main__":
    unittest.main()
