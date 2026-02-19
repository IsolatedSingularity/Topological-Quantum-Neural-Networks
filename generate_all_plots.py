#!/usr/bin/env python3
"""
Generate All Plots

Regenerates every static and animated visualization in the repository using the
Agg (non-interactive) backend.  Also produces reproducible screenshots of the
two tkinter GUIs in headless ``--screenshot`` mode, plus the robustness sandbox.

Run from the repo root:

    python generate_all_plots.py

Outputs are written to Plots/ by default.
"""

import os
import sys
import subprocess

# Use non-interactive backend so this works in headless CI environments
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(REPO_ROOT, 'Plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Add source directories to sys.path so imports resolve
sys.path.insert(0, os.path.join(REPO_ROOT, 'Code', 'Static Visualization'))

# ---------------------------------------------------------------------------
# Static visualizations (6 PNGs)
# ---------------------------------------------------------------------------
from static_visualizations import (
    plot_braiding_pattern,
    plot_large_braiding_pattern,
    plot_topological_charge_flow,
    plot_large_topological_charge_flow,
    plot_logical_gate_structure,
    plot_large_logical_gate_structure,
)

STATIC_PLOTS = [
    (plot_braiding_pattern,              'tqnn_braiding_pattern.png'),
    (plot_large_braiding_pattern,        'tqnn_large_braiding_pattern.png'),
    (plot_topological_charge_flow,       'tqnn_charge_flow.png'),
    (plot_large_topological_charge_flow, 'tqnn_large_charge_flow.png'),
    (plot_logical_gate_structure,        'tqnn_logical_gate.png'),
    (plot_large_logical_gate_structure,  'tqnn_large_logical_gate.png'),
]

# ---------------------------------------------------------------------------
# Animated visualizations (3 GIFs)
# ---------------------------------------------------------------------------
from animated_visualizations import (
    animate_braiding_pattern,
    animate_quantum_gate,
    animate_complex_quantum_circuit,
)

ANIMATED_PLOTS = [
    (animate_braiding_pattern,        'tqnn_braiding_animation.gif'),
    (animate_quantum_gate,            'tqnn_gate_animation.gif'),
    (animate_complex_quantum_circuit, 'tqnn_complex_circuit_animation.gif'),
]


def main() -> None:
    print("=" * 60)
    print("  Generating all plots")
    print("=" * 60)

    # --- Static ---
    print("\n--- Static Visualizations (6 PNGs) ---")
    for func, filename in STATIC_PLOTS:
        path = os.path.join(PLOTS_DIR, filename)
        try:
            func(path)
        except Exception as exc:
            print(f"  [FAIL] {filename}: {exc}")

    # --- Animated ---
    print("\n--- Animated Visualizations (3 GIFs) ---")
    for func, filename in ANIMATED_PLOTS:
        path = os.path.join(PLOTS_DIR, filename)
        try:
            func(path)
        except Exception as exc:
            print(f"  [FAIL] {filename}: {exc}")

    # --- Robustness sandbox ---
    print("\n--- Robustness Sandbox ---")
    try:
        from tqnn_sandbox import run_sandbox
        run_sandbox(
            os.path.join(PLOTS_DIR, 'tqnn_robustness_sandbox.png'),
            os.path.join(PLOTS_DIR, 'tqnn_robustness_animation.gif'),
        )
    except Exception as exc:
        print(f"  [FAIL] sandbox: {exc}")

    # --- GUI screenshots (--screenshot mode via subprocess) ---
    print("\n--- GUI Screenshots (tkinter --screenshot) ---")
    python = sys.executable
    gui_scripts = [
        (os.path.join(REPO_ROOT, 'Code', 'Image Classification',
                      'interactive_tqnn_classifier.py'),
         os.path.join(PLOTS_DIR, 'Classifier_Demo.png')),
        (os.path.join(REPO_ROOT, 'Code', 'Cobordism Viewer',
                      'cobordism_evolution_viewer.py'),
         os.path.join(PLOTS_DIR, 'Cobordism_Demo.png')),
    ]
    for script, out_path in gui_scripts:
        name = os.path.basename(script)
        try:
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'
            result = subprocess.run(
                [python, script, '--screenshot', out_path],
                timeout=30, capture_output=True, text=True,
                encoding='utf-8', errors='replace', env=env,
            )
            if result.returncode == 0:
                print(f"  [OK]   {name} -> {os.path.basename(out_path)}")
            else:
                print(f"  [FAIL] {name}: {result.stderr.strip()[:200]}")
        except Exception as exc:
            print(f"  [SKIP] {name}: {exc}")

    print("\n" + "=" * 60)
    print(f"  Done. Outputs are in {PLOTS_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
