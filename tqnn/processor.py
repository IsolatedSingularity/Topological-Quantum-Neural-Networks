"""
TQNN Processor: computation backend for Topological Quantum Neural Networks.

This module implements the Marciano-Zappala framework:
- Spin-network encoding of input patterns
- TQFT transition amplitude computation
- Semi-classical limit extraction
- 6j-symbol (Racah-Wigner) coefficient calculation via exact Racah formula
- Physical scalar product evaluation

Extracted from tqnn.simulation.gui to allow independent testing and reuse.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

from sympy.physics.wigner import wigner_6j as _sympy_wigner_6j


class SpinNetworkMode(Enum):
    """Types of spin-network encoding available"""
    HEXAGONAL = "Hexagonal Lattice (Lulli et al.)"
    STAR = "Star Graph (Perceptron)"
    GRID = "Desingularized Grid"


@dataclass
class SpinNetworkState:
    """
    Represents a spin-network state for TQNN computation

    Following Marciano et al.: spin-networks are graphs Gamma with edges labeled by
    SU(2) irreps j_e and vertices by intertwiners iota_v.

    Attributes:
        n_edges: Number of edges in the spin-network
        spin_labels: Array of spin values j_i on each edge (half-integers)
        quantum_dimensions: Delta_j = 2j + 1 for each edge
        intertwiner_labels: Labels at vertices (simplified)
        N_large: Large spin parameter for semi-classical limit
        transition_amplitude: Complex amplitude A from TQFT evaluation
        log_amplitude: log|A|^2 for numerical stability
    """
    n_edges: int
    spin_labels: np.ndarray
    quantum_dimensions: np.ndarray
    intertwiner_labels: np.ndarray = None
    N_large: int = 1000
    transition_amplitude: complex = 1.0
    log_amplitude: float = 0.0

    def __post_init__(self):
        if self.intertwiner_labels is None:
            self.intertwiner_labels = np.zeros(self.n_edges // 3 + 1)
        # Compute quantum dimensions Delta_j = 2j + 1
        self.quantum_dimensions = 2 * self.spin_labels + 1


@dataclass
class ClassPrototype:
    """
    Prototype for a class in TQNN classification

    Following the "training-free" approach: prototypes are defined by
    mean and std of spin colors from training examples.
    """
    label: str
    mean_spins: np.ndarray
    std_spins: np.ndarray
    n_samples: int = 0


class TQNNProcessor:
    """
    Handles TQFT-based computation for Topological Quantum Neural Networks

    This class implements the Marciano-Zappala framework:
    - Spin-network encoding of input patterns
    - TQFT transition amplitude computation
    - Semi-classical limit extraction
    - 6j-symbol (Racah-Wigner) coefficient calculation
    - Physical scalar product evaluation
    """

    def __init__(self, grid_size: int = 16, N_large: int = 1000):
        """Initialize the TQNN processor"""
        self.grid_size = grid_size
        self.N_large = N_large  # Large spin for semi-classical limit
        self.current_state = None
        self.network_mode = SpinNetworkMode.HEXAGONAL

        # Spin-network components
        self.spin_labels = np.array([])
        self.quantum_dimensions = np.array([])
        self.vertex_positions: List = []
        self.edge_connections: List = []

        # Class prototypes for classification
        self.prototypes: Dict[str, ClassPrototype] = {}
        self._initialize_default_prototypes()

        # TQFT computation results
        self.transition_amplitudes: Dict = {}  # Per-class amplitudes
        self.log_probabilities: Dict = {}
        self.six_j_symbols = np.array([])  # Racah-Wigner coefficients

        # Semi-classical analysis
        self.semiclassical_weights = np.array([])
        self.classical_activation = 0.0

        # History tracking
        self.amplitude_history: deque = deque(maxlen=100)
        self.weight_convergence_history: deque = deque(maxlen=50)

        # Hexagonal lattice for visualization
        self.hex_positions: List = []
        self.hex_spins: List = []
        self._generate_hexagonal_lattice()

    def _initialize_default_prototypes(self) -> None:
        """Initialize default class prototypes for demonstration"""
        # Create simple prototype classes
        n_edges = (self.grid_size * self.grid_size) // 4  # Approximate

        # Prototype A: Concentrated pattern (e.g., vertical line)
        mean_a = np.ones(n_edges) * self.N_large
        mean_a[n_edges//3:2*n_edges//3] += 10  # Higher spins in middle
        std_a = np.ones(n_edges) * 2.0
        self.prototypes['Class A'] = ClassPrototype('Class A', mean_a, std_a, 10)

        # Prototype B: Spread pattern (e.g., horizontal line)
        mean_b = np.ones(n_edges) * self.N_large
        mean_b[::3] += 10  # Higher spins distributed
        std_b = np.ones(n_edges) * 2.0
        self.prototypes['Class B'] = ClassPrototype('Class B', mean_b, std_b, 10)

        # Prototype C: Diagonal pattern
        mean_c = np.ones(n_edges) * self.N_large
        for i in range(n_edges):
            if i % 4 == 0:
                mean_c[i] += 10
        std_c = np.ones(n_edges) * 2.0
        self.prototypes['Class C'] = ClassPrototype('Class C', mean_c, std_c, 10)

    def _generate_hexagonal_lattice(self) -> None:
        """Generate hexagonal lattice positions for visualization"""
        self.hex_positions = []
        hex_size = 4  # Number of hexagons per side

        for q in range(-hex_size, hex_size + 1):
            for r in range(-hex_size, hex_size + 1):
                if abs(q + r) <= hex_size:
                    # Axial to pixel coordinates
                    x = 1.5 * q
                    y = np.sqrt(3) * (r + q/2)
                    self.hex_positions.append((x, y, q, r))

        self.hex_spins = np.zeros(len(self.hex_positions))

    def pattern_to_spin_network(self, pattern: np.ndarray) -> SpinNetworkState:
        """
        Convert a drawn pattern into a spin-network state

        Following Marciano et al.: j_i = N + floor(x_i) where x_i is pixel intensity
        This encoding places the system in the semi-classical regime.

        Args:
            pattern: 2D array representing the drawn pattern (values 0-1)

        Returns:
            SpinNetworkState object with spin labels and quantum dimensions
        """
        # Flatten and subsample pattern to get edge values
        flat_pattern = pattern.flatten()

        # Map to spin values: j_i = N_large + floor(x_i * scale)
        # Scale factor determines the "resolution" of spin encoding
        scale = 10.0
        spin_labels = self.N_large + np.floor(flat_pattern * scale)

        # Ensure half-integer spins (multiply by 0.5 for true SU(2))
        # For computational simplicity, we use integer approximation
        spin_labels = spin_labels.astype(float)

        # Compute quantum dimensions Delta_j = 2j + 1
        quantum_dims = 2 * spin_labels + 1

        # Update hexagonal lattice spins for visualization
        n_hex = len(self.hex_positions)
        if len(spin_labels) >= n_hex:
            self.hex_spins = spin_labels[:n_hex] - self.N_large  # Relative spins
        else:
            self.hex_spins = np.pad(spin_labels, (0, n_hex - len(spin_labels))) - self.N_large

        # Create state object
        state = SpinNetworkState(
            n_edges=len(spin_labels),
            spin_labels=spin_labels,
            quantum_dimensions=quantum_dims,
            N_large=self.N_large
        )

        self.current_state = state
        self.spin_labels = spin_labels
        self.quantum_dimensions = quantum_dims

        return state

    def compute_transition_amplitude(self, input_spins: np.ndarray,
                                    proto_mean: np.ndarray,
                                    proto_std: np.ndarray) -> Tuple[complex, float]:
        """
        Compute TQFT transition amplitude using the semi-classical formula

        From Marciano et al., the amplitude in the large-j limit is:
        A = prod_i Delta_{j_i} * exp(-(j_i - jbar_i)^2/(2 sigma_i^2)) * exp(-i xi_i j_i)

        Args:
            input_spins: Spin labels of input pattern
            proto_mean: Mean spin values for prototype
            proto_std: Std dev of spin values for prototype

        Returns:
            Tuple of (complex amplitude, log probability)
        """
        # Ensure arrays are same length
        min_len = min(len(input_spins), len(proto_mean), len(proto_std))
        j = input_spins[:min_len]
        j_bar = proto_mean[:min_len]
        sigma = proto_std[:min_len]

        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-6)

        # Quantum dimension contribution: log(Delta_j) = log(2j + 1)
        log_quantum_dim = np.sum(np.log(2 * j + 1))

        # Gaussian suppression term: -(j - jbar)^2/(2 sigma^2)
        gaussian_term = -np.sum((j - j_bar)**2 / (2 * sigma**2))

        # Phase term (visualization-only): not derived from TQFT; provides a
        # smooth complex phase for the amplitude display without affecting
        # the classification (log_amplitude ignores this term).
        xi = np.linspace(0, 0.1, min_len)
        phase_term = -np.sum(xi * j)

        # Log amplitude (for numerical stability)
        log_amplitude = log_quantum_dim + gaussian_term

        # Complex amplitude with phase
        amplitude = np.exp(log_amplitude / min_len) * np.exp(1j * phase_term / min_len)

        return amplitude, log_amplitude

    def compute_all_class_amplitudes(self) -> Dict[str, float]:
        """
        Compute transition amplitudes for all class prototypes

        Returns:
            Dictionary mapping class labels to log probabilities
        """
        if self.current_state is None:
            return {}

        self.transition_amplitudes = {}
        self.log_probabilities = {}

        for label, proto in self.prototypes.items():
            amplitude, log_prob = self.compute_transition_amplitude(
                self.current_state.spin_labels,
                proto.mean_spins,
                proto.std_spins
            )
            self.transition_amplitudes[label] = amplitude
            self.log_probabilities[label] = log_prob

        # Track amplitude history
        if self.log_probabilities:
            max_prob = max(self.log_probabilities.values())
            self.amplitude_history.append(max_prob)

        return self.log_probabilities

    def compute_six_j_symbol(self, j1: float, j2: float, j3: float,
                            j4: float, j5: float, j6: float) -> float:
        """
        Compute 6j-symbol (Racah-Wigner coefficient) via exact Racah formula.

        The 6j-symbol {j1 j2 j3; j4 j5 j6} represents recoupling of three
        angular momenta and is fundamental to spin-network evaluation.

        Uses sympy.physics.wigner.wigner_6j for the exact evaluation.
        """
        try:
            from sympy import Rational
            # Round to nearest half-integer for sympy compatibility
            def _to_half_int(x: float) -> float:
                return round(2 * x) / 2

            vals = [_to_half_int(v) for v in (j1, j2, j3, j4, j5, j6)]
            # Convert to sympy Rational for exact arithmetic
            rats = [Rational(int(2 * v), 2) for v in vals]
            result = _sympy_wigner_6j(*rats)
            return float(result)
        except (ValueError, TypeError):
            # Inadmissible spin configuration (triangle inequality violated)
            return 0.0

    def compute_recoupling_matrix(self) -> np.ndarray:
        """
        Compute matrix of 6j-symbols for current spin configuration.

        Each off-diagonal entry (i, k) is computed from a valid recoupling
        using Clebsch-Gordan admissible intermediate spins:
          j3 in {|j1-j2|, ..., j1+j2}  (integer steps)
        The recoupling matrix element is:
          U_{j12, j23} = (-1)^{j1+j2+j3+j4} sqrt((2j12+1)(2j23+1)) * {6j}

        Returns:
            Matrix of recoupling coefficients.
        """
        if self.current_state is None:
            return np.array([[1.0]])

        n_sample = min(8, len(self.spin_labels))
        sampled_spins = self.spin_labels[:n_sample] - self.N_large + 1
        sampled_spins = np.maximum(sampled_spins, 0.5)

        matrix = np.zeros((n_sample, n_sample))

        for i in range(n_sample):
            for k in range(n_sample):
                if i != k:
                    j1 = sampled_spins[i]
                    j2 = sampled_spins[k]
                    j4 = sampled_spins[(i + 1) % n_sample]
                    j5 = sampled_spins[(k + 1) % n_sample]
                    # Pick valid intermediate spins from the CG series
                    j3_min = abs(j1 - j2)
                    j3_max = j1 + j2
                    j6_min = abs(j4 - j5)
                    j6_max = j4 + j5
                    # Use the smallest admissible value (lowest coupling channel)
                    j3 = j3_min if j3_min > 0 else min(1.0, j3_max)
                    j6 = j6_min if j6_min > 0 else min(1.0, j6_max)
                    matrix[i, k] = self.compute_six_j_symbol(j1, j2, j3, j4, j5, j6)
                else:
                    matrix[i, k] = 1.0

        self.six_j_symbols = matrix
        return matrix

    def compute_semiclassical_weights(self) -> np.ndarray:
        """
        Extract semi-classical DNN weights from spin-network

        In the limit N_large -> inf, the TQNN reduces to a classical perceptron
        with weights w_i = j_i / N_large

        Returns:
            Array of classical weights
        """
        if self.current_state is None:
            return np.array([])

        # w_i = j_i / N_large (normalized to [0, 1] range)
        weights = (self.spin_labels - self.N_large) / 10.0  # Scale back
        weights = np.clip(weights, 0, 1)

        self.semiclassical_weights = weights

        # Track convergence to classical limit
        # As N_large increases, quantum corrections vanish
        quantum_correction = 1.0 / np.sqrt(self.N_large)
        self.weight_convergence_history.append(quantum_correction)

        return weights

    def compute_classical_activation(self) -> float:
        """
        Compute classical perceptron activation in semi-classical limit

        Returns:
            Activation value sigma(w*x + b)
        """
        if len(self.semiclassical_weights) == 0:
            return 0.5

        # Simple dot product with uniform "input" (the pattern itself acts as both)
        z = np.mean(self.semiclassical_weights)

        # Sigmoid activation
        self.classical_activation = 1.0 / (1.0 + np.exp(-10 * (z - 0.5)))

        return self.classical_activation

    def get_predicted_class(self) -> Tuple[str, float]:
        """
        Get predicted class based on maximum transition amplitude

        Returns:
            Tuple of (predicted class label, confidence)
        """
        if not self.log_probabilities:
            return "Unknown", 0.0

        # Softmax over log probabilities
        log_probs = np.array(list(self.log_probabilities.values()))
        labels = list(self.log_probabilities.keys())

        # Numerical stability
        log_probs = log_probs - np.max(log_probs)
        probs = np.exp(log_probs / 100)  # Temperature scaling
        probs = probs / np.sum(probs)

        best_idx = np.argmax(probs)
        return labels[best_idx], probs[best_idx]
