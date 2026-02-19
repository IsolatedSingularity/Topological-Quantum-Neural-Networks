# Cobordism Evolution Viewer

Interactive tkinter GUI visualizing the TQFT cobordism functor **Z(M) : Z(Σ\_in) → Z(Σ\_out)**.

## Quick Start

```bash
cd "Code/Cobordism Viewer"
python cobordism_evolution_viewer.py
```

Auto-screenshot (for CI / README images):

```bash
python cobordism_evolution_viewer.py --screenshot ../../Plots/Cobordism_Demo.png
```

## Panels

| Panel | Description |
|---|---|
| **Σ\_in** | Ring spin-network with color-coded representation labels |
| **Cobordism M** | Cross-section of the cobordism surface with an animated evolution front |
| **Amplitude evolution** | Per-class log\|A\|² traced along the cobordism parameter *t* |
| **Σ\_out** | Horizontal bar chart of output amplitudes, with prediction highlight |

## Controls

- **Cobordism type** — Cylinder (identity), Pair-of-Pants (splitting), Genus Handle (non-trivial topology)
- **Input presets** — Vertical / Horizontal / Cross / Circle or Random
- **Spins** — Number of edges on the input ring (4–40)
- **σ** — Gaussian width in the amplitude formula
- **Pause / Play** — freeze or resume the animated front
