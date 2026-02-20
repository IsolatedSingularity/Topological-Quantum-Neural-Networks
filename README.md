# Topological Quantum Neural Networks

[![CI](https://github.com/IsolatedSingularity/Topological-Quantum-Neural-Networks/actions/workflows/ci.yml/badge.svg)](https://github.com/IsolatedSingularity/Topological-Quantum-Neural-Networks/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Interactive tools for topological quantum neural networks: real-time tensor network simulation, topological pattern classification, cobordism evolution, and robustness analysis.

[Overview](#overview) - [Quick Start](#quick-start) - [Features](#features) - [Architecture](#architecture) - [Tech Stack](#tech-stack) - [References](#references) - [Citation](#citation) - [Contact](#questions-and-contact)

![Real-Time Tensor Network Simulator](Plots/tqnn_overall.png)

## Overview

This toolkit provides interactive simulators and batch visualization pipelines for **Topological Quantum Neural Networks (TQNNs)**, a gradient-free classification framework where inputs are encoded as spin-networks and classified via [TQFT](https://arxiv.org/abs/2210.13741) transition amplitudes. Because the classification signal is a topological invariant rather than a set of learned weights, TQNNs exhibit inherent resilience to local noise, an effect called **topological protection**.

The library implements the full pipeline: spin-network encoding, amplitude evaluation, prototype-based classification, cobordism evolution, and noise-robustness analysis, all exposed through three interactive tkinter GUIs and a suite of static/animated visualization generators (~5,800 lines of Python).

### Why Topological?

Conventional quantum neural networks rely on parameterized circuits whose gradients vanish exponentially as qubit count grows (the **barren plateau** problem). TQNNs sidestep this entirely: classification is driven by topological invariants that are, by definition, insensitive to smooth local perturbations. The result is a model whose accuracy degrades **gracefully** under noise rather than collapsing abruptly, without requiring explicit quantum error correction.

## Quick Start

```bash
git clone https://github.com/IsolatedSingularity/Topological-Quantum-Neural-Networks.git
cd Topological-Quantum-Neural-Networks
pip install -r requirements.txt
```

Launch any of the three interactive GUIs:

```bash
python "Code/Real Time Simulation/interactive_tqnn_tensor_network.py"   # Tensor network simulator
python "Code/Image Classification/interactive_tqnn_classifier.py"       # 6-panel classifier
python "Code/Cobordism Viewer/cobordism_evolution_viewer.py"            # Cobordism viewer
```

Or regenerate all static plots and animations in one step:

```bash
python generate_all_plots.py
```

<p align="center">
  <img src="./Plots/beta.gif?raw=true" alt="beta" width="34" height="34" />
</p>

## Features

### Real-Time Tensor Network Simulator

Draw a pattern on a 16 × 16 canvas and watch it get encoded, in real time, as a hexagonal spin-network. Four synchronized panels update on every brush stroke:

| Panel | What it shows |
|---|---|
| **Spin-network** | Each pixel maps to a node on a hexagonal lattice; edge colors reflect the spin label $j_i = N + \lfloor x_i \rfloor$ |
| **Transition amplitudes** | Per-class log-probability bars computed from the TQFT amplitude formula |
| **6j-symbol heatmap** | Recoupling coefficients that govern how three incoming spins fuse at a vertex |
| **Semi-classical weights** | Distribution of the Gaussian suppression term across all spins |

The $N_{\text{large}}$ slider (100 to 5000) controls the semi-classical regime: higher values sharpen the amplitude peaks, showing how the TQFT limit converges to classical classification.

![Tensor Network Simulator](Plots/tqnn_overall.png)

|  |  |
|:--:|:--:|
| ![Drawing Canvas](Plots/drawing.png) | ![Tutorial System](Plots/tqnn_tutorial.png) |
| Drawing canvas with hexagonal lattice mapping | Multi-page tutorial with color-coded sections |

---

### Interactive TQNN Classifier

A tkinter dark-themed GUI with six live panels that let you explore how topological protection keeps classification accurate under noise. Select one of four geometric patterns (vertical, horizontal, cross, circle), drag the noise slider to inject topological defects, and watch how the TQNN's confidence degrades slowly, because the amplitude is a *global* topological invariant rather than a fragile local feature.

| Panel | Purpose |
|---|---|
| **Input + Noise** | Shows the pattern after random pixel flips (simulating local defects) |
| **Anyonic Braiding** | Animated world-lines of six quasi-particles, illustrating the braiding operations that underpin TQNN computation |
| **Charge Flow** | Directed graph showing how topological charge propagates from the input layer through hidden nodes to the output |
| **Classification** | Horizontal bar chart of per-class log-probabilities with the winning class highlighted |
| **Spin-Network** | Hexagonal lattice where each node color encodes spin magnitude $j_i$ |
| **Robustness** | Sweep of noise levels (0 to 50%) showing all class curves; the current noise level is marked |

![Interactive TQNN Classifier](Plots/topology.png)

---

### Cobordism Evolution Viewer

A cobordism is a manifold $M$ whose boundary splits into an *input* surface $\Sigma_{\text{in}}$ and an *output* surface $\Sigma_{\text{out}}$. The TQFT functor $Z$ maps $M$ to a linear map $Z(\Sigma_{\text{in}}) \to Z(\Sigma_{\text{out}})$: this is the "forward pass" of a TQNN.

This viewer animates the evolution of spin-network amplitudes as a cursor sweeps from $\Sigma_{\text{in}}$ to $\Sigma_{\text{out}}$ through three cobordism types:

| Cobordism | Topology | Effect on spins |
|---|---|---|
| **Cylinder** | Identity ($\Sigma \times [0,1]$) | Spins pass through with small thermal jitter |
| **Pair-of-Pants** | Splitting (genus 0, 3 boundaries) | Spins on the second half become pairwise averages, sharing information between output legs |
| **Genus Handle** | Non-trivial genus | Cyclic coupling mixes distant spins, simulating a loop in the manifold |

![Cobordism Evolution Viewer](Plots/cobordism.png)

---

### Robustness Sandbox

How much noise can a TQNN tolerate before it misclassifies? The sandbox answers this quantitatively: it trains a `TQNNPerceptron` on four geometric patterns, then sweeps noise from 0% to 50% while recording per-class log-probabilities. The resulting curve shows a gradual, graceful degradation (the hallmark of topological protection) rather than the abrupt cliff typical of local-feature classifiers.

![TQNN Robustness](Plots/tqnn_robustness_sandbox.png)

---

### Static and Animated Visualizations

Pre-generated visualizations of the topological structures underlying TQNN computation. Run `python generate_all_plots.py` to regenerate all of them.

|  |  |
|:--:|:--:|
| ![Braiding Animation](Plots/tqnn_braiding_animation.gif) | ![Circuit Animation](Plots/tqnn_complex_circuit_animation.gif) |
| Six anyonic world-lines exchanging positions under the Yang-Baxter braid relation $B_{i} B_{i+1} B_{i} = B_{i+1} B_{i} B_{i+1}$ | Decomposition of a 3-qubit QFT into elementary topological gates, animated layer by layer |
| ![Charge Flow](Plots/tqnn_charge_flow.png) | ![Logical Gate](Plots/tqnn_logical_gate.png) |
| Directed charge-flow graph showing how conserved topological charge propagates through a TQNN layer | Logical gate structure built from fused anyon pairs, with fusion channels labeled by output spin $j_3$ |

---

## Architecture

The codebase is organized around processor classes that implement the TQFT math and GUI classes that provide interactive frontends.

| Class | Module | Role |
|---|---|---|
| `TQNNProcessor` | `interactive_tqnn_tensor_network.py` | Spin-network encoder, 6j-symbol evaluator, TQFT amplitude computation, MPS decomposition |
| `TQNNSimulator` | `interactive_tqnn_classifier.py` | Pattern generation, noise injection, prototype-based classification pipeline |
| `CobordismProcessor` | `cobordism_evolution_viewer.py` | TQFT functor evaluation along cylinder, pair-of-pants, and genus-handle cobordisms |
| `TQNNPerceptron` | `tqnn_helpers.py` | Lightweight classifier: trains prototypes from labeled patterns, predicts via log-amplitude |
| `TQNNVisualizerGUI` | `interactive_tqnn_tensor_network.py` | 1600-line tkinter app: drawing canvas, 4 synchronized matplotlib panels, tutorial system |
| `TQNNClassifierGUI` | `interactive_tqnn_classifier.py` | 6-panel dark-themed dashboard for noise-robustness exploration |
| `CobordismViewerGUI` | `cobordism_evolution_viewer.py` | 4-panel animated viewer with cobordism type selection and parameter sliders |

All GUIs share a consistent dark theme (`#1a1a1a` background, `#00ff88` accent) with matplotlib figures embedded via `FigureCanvasTkAgg`.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ (~5,800 lines across 11 modules) |
| GUI | tkinter (custom dark theme), matplotlib embedded via `FigureCanvasTkAgg` |
| Computation | numpy, scipy (linear algebra, special functions, optimization) |
| Visualization | matplotlib, seaborn (`mako` / `cubehelix` palettes), GIF export via `PillowWriter` |
| Graph Theory | networkx (spin-network topology, charge-flow graphs) |
| Testing | pytest (20 tests), GitHub Actions CI (Python 3.10 / 3.11 / 3.12) |

---

<details>
<summary><h2>Theoretical Background</h2></summary>

The core thesis, inspired by [Marciano et al. (2022)](https://arxiv.org/abs/2210.13741), is that conventional DNNs can be understood as the semi-classical limit of a more general TQNN framework. TQNNs encode input data into **spin-networks** whose edges carry irreducible representations of $SU(2)_k$, then classify by evaluating TQFT transition amplitudes.

**Encoding.** Each input value $x_i$ maps to a spin label:

$$j_i = N + \lfloor x_i \rfloor$$

where $N$ is a large integer placing the system in the semi-classical regime.

**Classification.** The TQFT functor $Z$ maps a cobordism $M$ to a transition amplitude $Z(\Sigma_{\text{in}}) \to Z(\Sigma_{\text{out}})$. For a class $c$ with prototype spins $\bar{j}_c$ and spread $\sigma_c$:

$$A_c \propto \prod_{i} (2j_i + 1) \, e^{-\frac{(j_i - \bar{j}_{c,i})^2}{2\sigma_{c,i}^2}}$$

$$\text{prediction} = \arg\max_{c} \left( \log|A_c|^2 \right)$$

The quantum dimension factor $(2j_i + 1)$ and the topological nature of the amplitude are what give TQNNs their noise resilience.

</details>

---

## References

- Marciano, A., Zappala, E., Torda, A., Lulli, M., et al. [*DNNs as the Semi-Classical Limit of TQNNs: The Problem of Generalisation*](https://arxiv.org/abs/2210.13741). arXiv:2210.13741, 2022.
- Lulli, M., Marciano, A., Zappala, E. [*The Exact Evaluation of Hexagonal Spin-Networks and TQNNs*](https://arxiv.org/abs/2310.03632). arXiv:2310.03632, 2023.
- Fields, C., Glazebrook, J. F., Marciano, A. [*Sequential Measurements, TQFTs, and TQNNs*](https://arxiv.org/abs/2205.13184). arXiv:2205.13184, 2022.
- Baez, J. [*Spin Networks in Gauge Theory*](https://arxiv.org/abs/gr-qc/9411007). Advances in Mathematics **117**, 253-272, 1996.

## Citation

If you use this software in your work, please cite:

```bibtex
@misc{morais2024tqnn,
  author       = {Morais, Jeffrey},
  title        = {Topological Quantum Neural Networks},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/IsolatedSingularity/Topological-Quantum-Neural-Networks}},
}
```

## See Also

Other quantum computing projects by the same author:

| Repository | Description |
|---|---|
| [Quantum-Chemistry-Eigensolver](https://github.com/IsolatedSingularity/Quantum-Chemistry-Eigensolver) | Variational quantum eigensolver for molecular ground-state energies |
| [Bloc-Fantome](https://github.com/IsolatedSingularity/Bloc-Fantome) | Quantum-inspired blockchain exploration |
| [Leonne](https://github.com/btq-ag/Leonne) | Modular consensus networks for cryptographic proof |
| [QRiNG](https://github.com/btq-ag/QRiNG) | Quantum random number generation for consensus protocols |
| [QLDPC](https://github.com/btq-ag/QLDPC) | Quantum LDPC code construction and circuit builder |

## Questions and Contact

If you have questions, feedback, or ideas for collaboration, feel free to [open an issue](https://github.com/IsolatedSingularity/Topological-Quantum-Neural-Networks/issues/new) or reach out directly:

- GitHub: [IsolatedSingularity](https://github.com/IsolatedSingularity)
- Website: [jeffreymorais.netlify.app](https://jeffreymorais.netlify.app/)
- LinkedIn: [Jeffrey Morais](https://www.linkedin.com/in/jeffrey-morais/)
