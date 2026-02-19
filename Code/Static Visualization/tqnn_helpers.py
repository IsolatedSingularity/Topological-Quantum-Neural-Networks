"""
TQNN Sandbox Helpers

This module provides the core components for a Topological Quantum Neural Network (TQNN)
sandbox. It includes a simplified TQNN Perceptron classifier based on the
semi-classical limit formulation described in the project's reference papers.
"""
from __future__ import annotations

import numpy as np

# A large spin value for the semi-classical limit, as mentioned in the papers.
N_LARGE: int = 1000

def create_spin_network_from_pattern(pattern: np.ndarray) -> np.ndarray:
    """
    Encodes a 2D binary pattern into a vector of spin "colors".

    This is a simplified encoding for the sandbox. A real implementation would
    use a more complex mapping like the hexagonal spin-network construction.
    Here, we flatten the pattern and scale it to represent spin states.

    Args:
        pattern (np.array): A 2D numpy array representing the input pattern.

    Returns:
        np.array: A 1D vector of spin colors.
    """
    flat_pattern = pattern.flatten()
    # Scale binary pattern (0 or 1) to a simple spin representation
    # This simulates the j = N + floor(x_i) mapping from the papers
    return N_LARGE + flat_pattern * 10

def add_topological_defect(pattern: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Introduces noise into a pattern to simulate a topological defect.

    This works by flipping a certain percentage of the pixels in the pattern.

    Args:
        pattern (np.array): The original 2D pattern.
        noise_level (float): The fraction of pixels to flip (0.0 to 1.0).

    Returns:
        np.array: The pattern with added noise.
    """
    if noise_level == 0.0:
        return pattern.copy()

    noisy_pattern = pattern.copy()
    num_flips = int(noise_level * noisy_pattern.size)
    
    # Choose random indices to flip
    indices_to_flip = np.random.choice(noisy_pattern.size, num_flips, replace=False)
    row_indices, col_indices = np.unravel_index(indices_to_flip, noisy_pattern.shape)

    # Flip the chosen pixels (0 to 1, 1 to 0)
    noisy_pattern[row_indices, col_indices] = 1 - noisy_pattern[row_indices, col_indices]
    
    return noisy_pattern

class TQNNPerceptron:
    """
    A TQNN Perceptron classifier based on the semi-classical limit.

    This classifier learns prototypes of different classes and uses a TQFT-inspired
    amplitude calculation to classify new patterns. It does not use gradient-based
    training.
    """
    def __init__(self) -> None:
        """Initializes the TQNNPerceptron."""
        self.prototypes: dict[str, dict[str, np.ndarray]] = {}
        self.class_labels: list[str] = []

    def train(self, patterns: list[np.ndarray], labels: list[str]) -> None:
        """
        Computes class prototypes from training data.

        This method follows the "training-free" approach from the papers, where
        the mean and standard deviation of spin colors for each class are
        calculated and stored as prototypes.

        Args:
            patterns (list of np.array): A list of 2D training patterns.
            labels (list): A list of corresponding labels for the patterns.
        """
        print("Computing class prototypes for TQNN Perceptron...")
        unique_labels = sorted(list(set(labels)))
        self.class_labels = unique_labels

        for label in unique_labels:
            class_patterns = [patterns[i] for i, l in enumerate(labels) if l == label]
            
            # Convert all patterns in the class to spin-networks
            class_spins = [create_spin_network_from_pattern(p) for p in class_patterns]
            
            if not class_spins:
                continue

            # Calculate mean and standard deviation for the spin colors
            mean_spins = np.mean(class_spins, axis=0)
            std_spins = np.std(class_spins, axis=0)
            
            # Avoid zero standard deviation to prevent division by zero
            std_spins[std_spins == 0] = 1e-6
            
            self.prototypes[label] = {
                'mean': mean_spins,
                'std': std_spins
            }
        print("Prototypes computed successfully.")

    def _calculate_log_probability(self, input_spins: np.ndarray, proto_mean: np.ndarray, proto_std: np.ndarray) -> float:
        """
        Calculates the log of the transition probability amplitude.

        This is based on the semi-classical amplitude formula from the papers:
        A = product(Delta_j * exp(-(j-j_bar)^2 / (2*sigma^2)))
        where Delta_j is the quantum dimension (j+1 for SU(2)).
        The log probability is log(|A|^2). We ignore the complex phase for simplicity.

        Args:
            input_spins (np.array): The spin-network of the input pattern.
            proto_mean (np.array): The mean spin values for a class prototype.
            proto_std (np.array): The std dev of spin values for a class prototype.

        Returns:
            float: The log probability of the input belonging to the prototype's class.
        """
        # Quantum dimension term: log((j+1)^2) = 2 * log(j+1)
        # Since input_spins are large, log(j+1) is approx log(j)
        # This term is nearly constant across prototypes and can be ignored for argmax.

        # Gaussian term
        gaussian_term = -np.sum(((input_spins - proto_mean) ** 2) / (2 * proto_std ** 2))
        
        return gaussian_term

    def predict(self, pattern: np.ndarray) -> tuple[str | None, dict[str, float]]:
        """
        Classifies a new pattern.

        Args:
            pattern (np.array): The 2D pattern to classify.

        Returns:
            tuple: A tuple containing the predicted label and a dictionary of
                   log probabilities for each class.
        """
        input_spins = create_spin_network_from_pattern(pattern)
        
        log_probs = {}
        for label, proto in self.prototypes.items():
            log_probs[label] = self._calculate_log_probability(
                input_spins, proto['mean'], proto['std']
            )

        if not log_probs:
            return None, {}
            
        # The predicted label is the one with the highest log probability
        predicted_label = max(log_probs, key=log_probs.get)
        
        return predicted_label, log_probs 