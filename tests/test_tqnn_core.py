"""
Tests for TQNN core components.

Covers the TQNNPerceptron classifier, spin-network encoding, and topological
defect (noise) injection from tqnn_helpers.
"""

import sys
import os
import numpy as np
import pytest

# Ensure the helpers module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Code', 'Static Visualization'))

from tqnn_helpers import (
    TQNNPerceptron,
    create_spin_network_from_pattern,
    add_topological_defect,
    N_LARGE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_patterns():
    """Three 10x10 binary patterns: vertical line, horizontal line, cross."""
    size = 10
    vertical = np.zeros((size, size))
    vertical[:, size // 2] = 1

    horizontal = np.zeros((size, size))
    horizontal[size // 2, :] = 1

    cross = np.zeros((size, size))
    for i in range(size):
        cross[i, i] = 1
        cross[i, size - 1 - i] = 1

    patterns = [vertical, horizontal, cross]
    labels = ["Vertical", "Horizontal", "Cross"]
    return patterns, labels


@pytest.fixture
def trained_tqnn(simple_patterns):
    """A TQNNPerceptron trained on the simple patterns."""
    patterns, labels = simple_patterns
    tqnn = TQNNPerceptron()
    tqnn.train(patterns, labels)
    return tqnn


# ---------------------------------------------------------------------------
# Spin-network encoding
# ---------------------------------------------------------------------------

class TestSpinNetworkEncoding:
    def test_output_shape(self):
        pattern = np.zeros((8, 8))
        spins = create_spin_network_from_pattern(pattern)
        assert spins.shape == (64,)

    def test_zero_pattern_encodes_to_n_large(self):
        pattern = np.zeros((5, 5))
        spins = create_spin_network_from_pattern(pattern)
        np.testing.assert_array_equal(spins, N_LARGE)

    def test_ones_pattern_offset(self):
        pattern = np.ones((5, 5))
        spins = create_spin_network_from_pattern(pattern)
        np.testing.assert_array_equal(spins, N_LARGE + 10)

    def test_mixed_pattern(self):
        pattern = np.array([[0, 1], [1, 0]])
        spins = create_spin_network_from_pattern(pattern)
        expected = np.array([N_LARGE, N_LARGE + 10, N_LARGE + 10, N_LARGE])
        np.testing.assert_array_equal(spins, expected)


# ---------------------------------------------------------------------------
# Topological defect / noise injection
# ---------------------------------------------------------------------------

class TestTopologicalDefect:
    def test_zero_noise_returns_copy(self):
        pattern = np.eye(5)
        noisy = add_topological_defect(pattern, 0.0)
        np.testing.assert_array_equal(noisy, pattern)
        # Verify it is a copy, not the same object
        assert noisy is not pattern

    def test_full_noise_flips_all(self):
        pattern = np.zeros((4, 4))
        noisy = add_topological_defect(pattern, 1.0)
        assert np.sum(noisy) == 16  # all flipped 0 -> 1

    def test_partial_noise_flip_count(self):
        np.random.seed(42)
        pattern = np.zeros((10, 10))
        noisy = add_topological_defect(pattern, 0.2)
        # 20% of 100 pixels = 20 flips
        assert np.sum(noisy) == 20

    def test_shape_preserved(self):
        pattern = np.ones((7, 3))
        noisy = add_topological_defect(pattern, 0.5)
        assert noisy.shape == pattern.shape


# ---------------------------------------------------------------------------
# TQNNPerceptron
# ---------------------------------------------------------------------------

class TestTQNNPerceptron:
    def test_train_stores_prototypes(self, trained_tqnn):
        assert set(trained_tqnn.class_labels) == {"Vertical", "Horizontal", "Cross"}
        assert len(trained_tqnn.prototypes) == 3

    def test_predict_clean_pattern(self, trained_tqnn, simple_patterns):
        """Clean patterns should be classified correctly."""
        patterns, labels = simple_patterns
        for pattern, expected_label in zip(patterns, labels):
            predicted, log_probs = trained_tqnn.predict(pattern)
            assert predicted == expected_label

    def test_predict_returns_all_classes(self, trained_tqnn, simple_patterns):
        patterns, _ = simple_patterns
        _, log_probs = trained_tqnn.predict(patterns[0])
        assert set(log_probs.keys()) == {"Vertical", "Horizontal", "Cross"}

    def test_log_probs_are_negative(self, trained_tqnn, simple_patterns):
        """Log probabilities (Gaussian exponents) should generally be <= 0."""
        patterns, _ = simple_patterns
        _, log_probs = trained_tqnn.predict(patterns[0])
        for lp in log_probs.values():
            assert lp <= 0.0

    def test_noisy_classification_degrades(self, trained_tqnn, simple_patterns):
        """At high noise the correct-class log-prob should be lower than at zero noise."""
        np.random.seed(0)
        patterns, labels = simple_patterns
        clean_pattern = patterns[0]
        correct_label = labels[0]

        _, clean_lp = trained_tqnn.predict(clean_pattern)
        noisy_pattern = add_topological_defect(clean_pattern, 0.4)
        _, noisy_lp = trained_tqnn.predict(noisy_pattern)

        assert noisy_lp[correct_label] < clean_lp[correct_label]

    def test_untrained_predict(self):
        """Predicting before training should return None."""
        tqnn = TQNNPerceptron()
        predicted, log_probs = tqnn.predict(np.zeros((5, 5)))
        assert predicted is None
        assert log_probs == {}
