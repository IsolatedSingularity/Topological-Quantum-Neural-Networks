"""
Tests for the TQNNProcessor computation backend and Jones polynomial invariants.
"""

import numpy as np
import pytest
from sympy import Symbol, simplify

from tqnn.processor import (
    TQNNProcessor,
    SpinNetworkState,
    SpinNetworkMode,
    ClassPrototype,
)
from tqnn.invariants import jonesPolynomial, bracketPolynomial, verifyKnownValues


# ---------------------------------------------------------------------------
# TQNNProcessor
# ---------------------------------------------------------------------------

class TestTQNNProcessor:
    @pytest.fixture
    def processor(self) -> TQNNProcessor:
        proc = TQNNProcessor()
        proc.num_classes = 3
        proc.class_names = ["Circle", "Triangle", "Square"]
        proc.prototypes = {
            name: ClassPrototype(
                label=name,
                mean_spins=np.random.RandomState(i).rand(16) * 2,
                std_spins=np.random.RandomState(i).rand(16) * 0.1 + 0.01,
                n_samples=10,
            )
            for i, name in enumerate(proc.class_names)
        }
        return proc

    def test_pattern_to_spin_network(self, processor: TQNNProcessor) -> None:
        pattern = np.random.rand(4, 4)
        state = processor.pattern_to_spin_network(pattern)
        assert isinstance(state, SpinNetworkState)
        assert state.spin_labels.shape == (16,)

    def test_compute_six_j_known_value(self) -> None:
        """wigner_6j(1,1,1,1,1,1) = 1/6 (exact via sympy)."""
        proc = TQNNProcessor()
        result = proc.compute_six_j_symbol(1, 1, 1, 1, 1, 1)
        assert abs(result - 1.0 / 6.0) < 1e-10

    def test_compute_six_j_triangle_violation(self) -> None:
        """Triangle condition violation should return 0."""
        proc = TQNNProcessor()
        result = proc.compute_six_j_symbol(1, 1, 5, 1, 1, 1)
        assert result == 0.0

    def test_classification_returns_valid_class(self, processor: TQNNProcessor) -> None:
        pattern = np.random.rand(4, 4)
        processor.pattern_to_spin_network(pattern)
        amplitudes = processor.compute_all_class_amplitudes()
        assert len(amplitudes) == 3
        predicted, confidence = processor.get_predicted_class()
        assert predicted in processor.class_names

    def test_recoupling_matrix_square(self, processor: TQNNProcessor) -> None:
        pattern = np.random.rand(4, 4)
        processor.pattern_to_spin_network(pattern)
        matrix = processor.compute_recoupling_matrix()
        assert matrix.shape[0] == matrix.shape[1]

    def test_noise_robustness(self, processor: TQNNProcessor) -> None:
        """Classification should be consistent under small noise."""
        np.random.seed(42)
        pattern = np.random.rand(4, 4)
        processor.pattern_to_spin_network(pattern)
        processor.compute_all_class_amplitudes()
        predicted, _ = processor.get_predicted_class()

        noisy = pattern + np.random.normal(0, 0.01, pattern.shape)
        processor.pattern_to_spin_network(noisy)
        processor.compute_all_class_amplitudes()
        noisyPredicted, _ = processor.get_predicted_class()
        assert predicted == noisyPredicted


# ---------------------------------------------------------------------------
# Jones polynomial invariants
# ---------------------------------------------------------------------------

class TestJonesPolynomial:
    def test_unknot(self) -> None:
        result = jonesPolynomial([], 2)
        assert simplify(result - 1) == 0

    def test_trefoil(self) -> None:
        t = Symbol("t")
        result = jonesPolynomial([1, 1, 1], 2)
        expected = -t**4 + t**3 + t
        assert simplify(result - expected) == 0

    def test_figure_eight(self) -> None:
        t = Symbol("t")
        result = jonesPolynomial([1, -2, 1, -2], 3)
        expected = t**2 - t + 1 - t**(-1) + t**(-2)
        assert simplify(result - expected) == 0

    def test_verify_known_values_all_pass(self) -> None:
        results = verifyKnownValues()
        for knot, passed in results.items():
            assert passed, f"Jones polynomial verification failed for {knot}"

    def test_bracket_unknot(self) -> None:
        result = bracketPolynomial([], 2)
        assert simplify(result - 1) == 0

    def test_invalid_strand_count(self) -> None:
        with pytest.raises(ValueError):
            bracketPolynomial([1], 1)

    def test_invalid_crossing_index(self) -> None:
        with pytest.raises(ValueError):
            bracketPolynomial([3], 2)
