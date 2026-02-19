# Topological Quantum Neural Networks

Interactive tools for topological quantum neural networks: real-time tensor network simulation, pattern classification with noise robustness, and 3D Tanner graph visualization.

<p align="center">
  <img src="./Plots/beta.gif?raw=true" alt="beta" width="34" height="34" />
</p>

![Real-Time Tensor Network Simulator](Plots/tqnn_overall.png)

## Overview

This project implements Topological Quantum Neural Networks (TQNNs), a machine learning framework where classical deep neural networks emerge as the semi-classical limit of a topological quantum field theory. Instead of training weights through gradient descent, TQNNs encode data into spin-networks and classify patterns by evaluating TQFT transition amplitudes, providing inherent noise robustness through topological protection. The repository includes interactive simulators, visualization tools, and a robustness testing sandbox, all built in Python.

## Quick Start

```bash
git clone https://github.com/IsolatedSingularity/Topological-Quantum-Neural-Networks.git
cd Topological-Quantum-Neural-Networks
pip install -r requirements.txt
```

Run any of the interactive applications:

```bash
# Real-time tensor network simulator (tkinter GUI)
python "Code/Real Time Simulation/interactive_tqnn_tensor_network.py"

# Interactive TQNN classifier (matplotlib GUI)
python "Code/Image Classification/interactive_tqnn_gui.py"

# 3D Tanner graph visualizer (matplotlib 3D)
python "Code/Interactive Quantum Tanner/interactive_3d_tanner_graph.py"

# Robustness sandbox (generates plots)
python "Code/Static Visualization/tqnn_sandbox.py"

# Static & animated visualizations (generates PNGs and GIFs)
python "Code/Static Visualization/static_visualizations.py"
python "Code/Static Visualization/animated_visualizations.py"
```

## Features

### Real-Time Tensor Network Simulator

A tkinter dark-themed GUI for real-time TQNN visualization. Draw patterns on a 16x16 canvas and watch them get encoded as hexagonal spin-networks with live updates across four synchronized panels: spin-network topology, TQFT transition amplitudes, 6j-symbol recoupling heatmap, and semi-classical weight distribution. Includes an adjustable $N_{\text{large}}$ parameter (100 to 5000) to explore the semi-classical limit, and a multi-page interactive tutorial.

![Tensor Network Simulator](Plots/tqnn_overall.png)

|  |  |
|:--:|:--:|
| ![Drawing Canvas](Plots/drawing.png) | ![Tutorial System](Plots/tqnn_tutorial.png) |
| Drawing canvas with hexagonal lattice mapping | Multi-page tutorial with color-coded sections |

```python
# From Code/Real Time Simulation/interactive_tqnn_tensor_network.py
class TQNNProcessor:
    def compute_transition_amplitude(self, input_spins: np.ndarray, 
                                    proto_mean: np.ndarray, 
                                    proto_std: np.ndarray) -> Tuple[complex, float]:
        """
        Compute TQFT transition amplitude using the semi-classical formula.
        From Marciano et al., the amplitude in the large-j limit is:
        A = prod_i Delta_{j_i} * exp(-(j_i - j_bar_i)^2/(2*sigma_i^2))
        """
        min_len = min(len(input_spins), len(proto_mean), len(proto_std))
        j = input_spins[:min_len]
        j_bar = proto_mean[:min_len]
        sigma = proto_std[:min_len]
        
        # Quantum dimension contribution: log(Delta_j) = log(2j + 1)
        log_quantum_dim = np.sum(np.log(2 * j + 1))
        
        # Gaussian suppression term: -(j - j_bar)^2 / (2 * sigma^2)
        gaussian_term = -np.sum((j - j_bar)**2 / (2 * sigma**2))
        
        log_amplitude = log_quantum_dim + gaussian_term
        amplitude = np.exp(log_amplitude / min_len)
        return amplitude, log_amplitude
```

---

### Interactive TQNN Classifier

A matplotlib-based interactive environment with six visualization panels. Supports four geometric pattern types (vertical, horizontal, cross, circle) with a real-time noise slider. Panels display classification confidence, anyonic braiding patterns, charge flow through spin-networks, and topological robustness metrics.

```python
# From Code/Image Classification/interactive_tqnn_gui.py
class TQNNVisualization:
    def draw_classification(self):
        """Draw classification results with real-time confidence"""
        if self.simulator.log_probabilities:
            labels = list(self.simulator.log_probabilities.keys())
            values = list(self.simulator.log_probabilities.values())
            
            bars = self.ax_classification.bar(
                range(len(labels)), 
                values,
                color=[seqCmap(0.7) if label == self.simulator.prediction_result 
                       else seqCmap(0.3) for label in labels]
            )
            
            correct_idx = labels.index(self.simulator.current_pattern_label)
            bars[correct_idx].set_color('green')
            bars[correct_idx].set_alpha(0.7)
```

![Interactive TQNN Environment](Plots/Interactive.png)

---

### 3D Tanner Graph Visualizer

An interactive 3D visualization of quantum LDPC Tanner graphs embedded on topological surfaces. Supports genus 0 through 5 with hyperbolic curvature, error injection, and belief propagation syndrome decoding with live animation. Features auto-rotation and adjustable surface parameters.

![3D Tanner Graph](Plots/QLDPC_Demo.png)

---

### Robustness Sandbox

A standalone testing environment that trains a TQNN Perceptron on geometric patterns and sweeps noise levels from 0% to 50%. Generates quantitative robustness plots and animated radial confidence GIFs demonstrating topological protection.

```python
# From Code/Static Visualization/tqnn_sandbox.py
def plot_degradation(results, target_label, plot_path):
    """Plots the degradation of classification confidence vs. noise level."""
    noise_levels = sorted(results.keys())
    palette = sns.color_palette("mako", n_colors=len(results[0.0]))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, label in enumerate(results[0.0].keys()):
        log_probs = [results[noise][label] for noise in noise_levels]
        ax.plot(noise_levels, log_probs, marker='o', 
                linestyle='--', label=f'Class: {label}', color=palette[i])
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
```

![TQNN Robustness](Plots/tqnn_robustness_sandbox.png)

---

### Static and Animated Visualizations

Pre-generated braiding animations, charge flow diagrams, logical gate structures, and circuit compositions.

|  |  |
|:--:|:--:|
| ![Braiding Animation](Plots/tqnn_braiding_animation.gif) | ![Circuit Animation](Plots/tqnn_complex_circuit_animation.gif) |
| Anyonic braiding (6 strands) | Toffoli gate circuit composition |
| ![Charge Flow](Plots/tqnn_charge_flow.png) | ![Logical Gate](Plots/tqnn_logical_gate.png) |
| Charge flow through spin-network | Logical gate structure |

---

## Project Structure

```
Code/
  Real Time Simulation/          # Tkinter tensor network simulator (1600+ lines)
  Image Classification/          # Matplotlib interactive classifier
  Interactive Quantum Tanner/    # 3D QLDPC Tanner graph visualizer
  Static Visualization/          # Sandbox, static plots, animated GIFs, helpers
Plots/                           # Generated figures and animations
References/                      # Source papers (Marciano, Lulli, Fields)
```

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| GUI | tkinter, matplotlib (interactive backends) |
| Computation | numpy, scipy (linear algebra, optimization) |
| Visualization | matplotlib, seaborn (`mako` / `cubehelix` palettes) |
| Graph Theory | networkx |

---

<details>
<summary><h2>Theoretical Background</h2></summary>

### Overview

The core thesis, inspired by the research of Marciano, Fields, Lulli, and others, is that conventional DNNs can be understood as the semi-classical limit of a more general TQNN framework. TQNNs leverage the properties of topological invariants, making them naturally resilient to local perturbations and noise (topological protection).

### Spin-Network Encoding

A TQNN processes information encoded in **spin-networks**: graphs whose edges are labeled by irreducible representations of a quantum group (e.g., $SU(2)_k$), called "colors" ($j$), and whose nodes represent intertwiners. Input data is encoded as:

$$j_i = N + \lfloor x_i \rfloor$$

where $N$ is a large integer placing the system in the semi-classical regime.

### TQFT Transition Amplitudes

The core operation is the evaluation of a TQFT functor $Z$, mapping a cobordism $M$ (the evolution of spin-networks) to a transition amplitude:

$$Z(M): Z(\Sigma_{\text{in}}) \to Z(\Sigma_{\text{out}})$$

For a class $c$ with prototype mean $\bar{j}$ and standard deviation $\sigma$:

$$A_c \propto \prod_{i} \Delta_{j_i} e^{-\frac{(j_i - j_{c,i})^2}{2\sigma_{c,i}^2}}$$

where $\Delta_{j_i} = 2j_i + 1$ is the quantum dimension. Classification uses:

$$\text{prediction} = \arg\max_{c} \left( \log|A_c|^2 \right)$$

### MPS Decomposition

The quantum state from drawn patterns is decomposed into Matrix Product State (MPS) form:

$$|\psi\rangle = \sum_{i_1,\ldots,i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} \cdots A^{[n]}_{i_n} |i_1 i_2 \cdots i_n\rangle$$

with entanglement entropy $S = -\sum_i \lambda_i^2 \log_2(\lambda_i^2)$ computed from Schmidt values.

### Braiding and Fusion

Anyonic braiding operations satisfy the Yang-Baxter equation:

$$(B \otimes I)(I \otimes B)(B \otimes I) = (I \otimes B)(B \otimes I)(I \otimes B)$$

Fusion of two spins follows $SU(2)$ rules:

$$j_1 \otimes j_2 = \bigoplus_{j_3=|j_1-j_2|}^{j_1+j_2} j_3$$

</details>

---

## Next Steps

- [ ] **Qiskit / PennyLane backend**: Integrate quantum circuit backends for hardware-ready execution
- [ ] **Unit test coverage**: Add pytest suite for core TQNN processor and spin-network encoding
- [ ] **GPU acceleration**: Profile and optimize spin-network evaluation with CuPy or JAX
- [ ] **PyPI packaging**: Package the core TQNN library for `pip install` distribution
- [ ] **CI/CD pipeline**: Add GitHub Actions for linting, testing, and artifact generation
- [ ] **Export functionality**: Add save/export for visualization states (PNG, JSON)

> [!NOTE]
> This implementation simulates topological quantum behavior on classical hardware. While it demonstrates the principles of topological robustness, it does not provide the computational advantages of a true quantum computer.

## Contributing

Contributions are welcome. To get started:

1. Fork the repository and create a feature branch
2. Install dependencies: `pip install -r requirements.txt`
3. Run the test suite: `pytest tests/`
4. Submit a pull request with a clear description of changes

Please follow the existing code style: use `seaborn` mako/cubehelix palettes for visualizations, include docstrings for public functions, and add type hints to function signatures.

## References

- Marciano, A., et al. *DNNs as the Semi-Classical Limit of TQNNs*.
- Lulli, M., et al. *Exact hexagonal spin-networks and TQNNs*.
- Fields, C., et al. *Sequential measurements, TQFTs, and TQNNs*.
