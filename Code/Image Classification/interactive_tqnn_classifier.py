"""
Interactive TQNN Classifier

A tkinter dark-themed GUI for real-time topological quantum neural network
classification. Users can select geometric patterns, inject noise via a
slider, and observe how TQNN classification confidence degrades under
topological defects -- demonstrating that topological protection preserves
correct classification far beyond what a naive classifier would allow.

The six panels show:
  1. Input pattern with noise overlay
  2. Anyonic braiding world-lines
  3. Charge-flow network
  4. Classification confidence bars
  5. Spin-network representation
  6. Topological robustness sweep

GUI Framework: tkinter with custom dark theme (matching tensor network simulator)
Architecture: Event-driven with embedded matplotlib via FigureCanvasTkAgg
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import numpy as np
import tkinter as tk
from tkinter import ttk
from collections import deque
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

# Add project path for helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Static Visualization"))
from tqnn_helpers import TQNNPerceptron, add_topological_defect, create_spin_network_from_pattern

# ---------------------------------------------------------------------------
# Palette & theme constants (match tensor-network simulator exactly)
# ---------------------------------------------------------------------------
sns.set_style("darkgrid")
seqCmap = sns.color_palette("mako", as_cmap=True)
divCmap = sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)
altCmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=0.95, reverse=True, as_cmap=True)

DARK_BG      = "#1a1a1a"
DARK_AXES    = "#0a0a0a"
DARK_TEXT    = "#ffffff"
DARK_ACCENT  = "#00ff88"
DARK_GRID    = "#2d2d2d"
DARK_EDGE    = "#444444"
DARK_SUBTITLE = "#aaaaaa"

PATTERN_SIZE     = 8
MAX_NOISE        = 0.50
BRAIDING_STRANDS = 6
ANIMATION_MS     = 120


# ═══════════════════════════════════════════════════════════════════════════
# Simulation back-end
# ═══════════════════════════════════════════════════════════════════════════

class TQNNSimulator:
    """Core classifier + topological-property tracker."""

    def __init__(self, pattern_size: int = PATTERN_SIZE) -> None:
        self.pattern_size = pattern_size
        self.tqnn = TQNNPerceptron()
        self.patterns, self.labels = self._make_patterns()
        self.tqnn.train(self.patterns, self.labels)

        # State
        self.current_pattern: np.ndarray = self.patterns[0].copy()
        self.current_pattern_label: str = self.labels[0]
        self.current_noise: float = 0.0
        self.noisy_pattern: np.ndarray = self.current_pattern.copy()
        self.prediction: str = ""
        self.log_probs: Dict[str, float] = {}
        self.confidence_history: deque = deque(maxlen=80)
        self.topological_charge: float = 0.0
        self.braiding_t: int = 0
        self.charge_t: int = 0

        self._classify()

    # --- Pattern generation ---------------------------------------------------

    def _make_patterns(self) -> Tuple[List[np.ndarray], List[str]]:
        s = self.pattern_size
        vertical = np.zeros((s, s)); vertical[:, s // 2] = 1
        horizontal = np.zeros((s, s)); horizontal[s // 2, :] = 1
        cross = np.zeros((s, s))
        for i in range(s):
            cross[i, i] = 1; cross[i, s - 1 - i] = 1
        circle = np.zeros((s, s))
        cx, cy, r = s // 2, s // 2, s // 3
        for i in range(s):
            for j in range(s):
                if abs(np.sqrt((i - cx)**2 + (j - cy)**2) - r) < 1:
                    circle[i, j] = 1
        return [vertical, horizontal, cross, circle], ["Vertical", "Horizontal", "Cross", "Circle"]

    # --- Accessors / mutators -------------------------------------------------

    def select_pattern(self, idx: int) -> None:
        if 0 <= idx < len(self.patterns):
            self.current_pattern = self.patterns[idx].copy()
            self.current_pattern_label = self.labels[idx]
            self._classify()

    def set_noise(self, level: float) -> None:
        self.current_noise = max(0.0, min(MAX_NOISE, level))
        self._classify()

    def toggle_pixel(self, row: int, col: int) -> None:
        if 0 <= row < self.pattern_size and 0 <= col < self.pattern_size:
            self.current_pattern[row, col] = 1 - self.current_pattern[row, col]
            self._classify()

    # --- Internal classification -----------------------------------------------

    def _classify(self) -> None:
        self.noisy_pattern = add_topological_defect(self.current_pattern, self.current_noise)
        self.prediction, self.log_probs = self.tqnn.predict(self.noisy_pattern)
        if self.log_probs:
            self.confidence_history.append(max(self.log_probs.values()))
        self.topological_charge = float(np.sum(self.noisy_pattern)) / self.pattern_size**2

    def tick(self) -> None:
        self.braiding_t += 1
        self.charge_t += 1

    # --- Robustness sweep (precomputed for the rightmost panel) ---------------

    def robustness_sweep(self, n_levels: int = 25) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        levels = np.linspace(0, MAX_NOISE, n_levels)
        curves: Dict[str, list] = {lb: [] for lb in self.labels}
        for lv in levels:
            noisy = add_topological_defect(self.current_pattern, lv)
            _, lp = self.tqnn.predict(noisy)
            for lb in self.labels:
                curves[lb].append(lp.get(lb, -1000.0))
        return levels, {k: np.array(v) for k, v in curves.items()}


# ═══════════════════════════════════════════════════════════════════════════
# GUI application
# ═══════════════════════════════════════════════════════════════════════════

class TQNNClassifierGUI:
    """Tkinter dark-themed GUI with six embedded matplotlib panels."""

    def __init__(self, screenshot_path: Optional[str] = None) -> None:
        self.sim = TQNNSimulator()
        self.screenshot_path = screenshot_path
        self.auto_mode = False

        # --- Window ----------------------------------------------------------
        self.root = tk.Tk()
        self.root.title("TQNN Interactive Classifier")
        self.root.geometry("1700x1000")
        self.root.configure(bg=DARK_BG)
        self._apply_theme()

        # --- Layout ----------------------------------------------------------
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(main, style="Dark.TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        right = ttk.Frame(main, style="Dark.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_pattern_selector(left)
        self._build_controls(left)
        self._build_status(left)
        self._build_plots(right)

        # --- Animation loop --------------------------------------------------
        self._schedule_tick()

        # Screenshot mode: render one frame then save & exit
        if self.screenshot_path:
            self.root.after(600, self._take_screenshot)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Dark.TFrame", background=DARK_BG)
        s.configure("Dark.TLabel", background=DARK_BG, foreground=DARK_TEXT)
        s.configure("Dark.TLabelframe", background=DARK_BG, foreground=DARK_TEXT,
                    borderwidth=2, relief="solid")
        s.configure("Dark.TLabelframe.Label", background=DARK_BG, foreground=DARK_ACCENT,
                    font=("TkDefaultFont", 10, "bold"))
        s.configure("Dark.TButton", background=DARK_GRID, foreground=DARK_TEXT,
                    borderwidth=1, relief="raised")
        s.map("Dark.TButton",
              background=[("active", "#3d3d3d")],
              foreground=[("active", DARK_ACCENT)])
        s.configure("Dark.TCheckbutton", background=DARK_BG, foreground=DARK_TEXT)
        s.configure("Dark.TRadiobutton", background=DARK_BG, foreground=DARK_TEXT)

    # ------------------------------------------------------------------
    # Left panel: pattern selector
    # ------------------------------------------------------------------

    def _build_pattern_selector(self, parent: ttk.Frame) -> None:
        fr = ttk.LabelFrame(parent, text="Pattern Selector", style="Dark.TLabelframe", padding=10)
        fr.pack(fill=tk.X, pady=(0, 10))

        self.pattern_var = tk.IntVar(value=0)
        for idx, label in enumerate(self.sim.labels):
            rb = ttk.Radiobutton(fr, text=label, variable=self.pattern_var,
                                 value=idx, style="Dark.TRadiobutton",
                                 command=self._on_pattern_change)
            rb.pack(anchor=tk.W, pady=2)

    def _on_pattern_change(self) -> None:
        self.sim.select_pattern(self.pattern_var.get())

    # ------------------------------------------------------------------
    # Left panel: controls
    # ------------------------------------------------------------------

    def _build_controls(self, parent: ttk.Frame) -> None:
        fr = ttk.LabelFrame(parent, text="TQNN Controls", style="Dark.TLabelframe", padding=10)
        fr.pack(fill=tk.X, pady=(0, 10))

        # Noise slider
        ttk.Label(fr, text="Noise Level (topological defect rate):",
                 style="Dark.TLabel").pack(anchor=tk.W)
        self.noise_var = tk.DoubleVar(value=0.0)
        self.noise_scale = tk.Scale(fr, from_=0.0, to=MAX_NOISE, resolution=0.01,
                                    orient=tk.HORIZONTAL, variable=self.noise_var,
                                    bg=DARK_GRID, fg=DARK_ACCENT,
                                    highlightthickness=0, troughcolor=DARK_BG,
                                    command=self._on_noise_change)
        self.noise_scale.pack(fill=tk.X)

        # Buttons
        btn_fr = ttk.Frame(fr, style="Dark.TFrame")
        btn_fr.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_fr, text="Classify", style="Dark.TButton",
                   command=self._on_classify).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(btn_fr, text="Reset Noise", style="Dark.TButton",
                   command=self._on_reset).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # Auto mode
        self.auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fr, text="Auto sweep noise", variable=self.auto_var,
                       style="Dark.TCheckbutton").pack(anchor=tk.W, pady=(8, 0))

    def _on_noise_change(self, _val: str) -> None:
        self.sim.set_noise(self.noise_var.get())

    def _on_classify(self) -> None:
        self.sim._classify()

    def _on_reset(self) -> None:
        self.noise_var.set(0.0)
        self.sim.set_noise(0.0)

    # ------------------------------------------------------------------
    # Left panel: status
    # ------------------------------------------------------------------

    def _build_status(self, parent: ttk.Frame) -> None:
        fr = ttk.LabelFrame(parent, text="Classification Results", style="Dark.TLabelframe", padding=10)
        fr.pack(fill=tk.BOTH, expand=True)
        self.status_text = tk.Text(fr, height=14, bg=DARK_AXES, fg=DARK_ACCENT,
                                   insertbackground=DARK_ACCENT, selectbackground=DARK_GRID,
                                   font=("Courier", 9), wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self._update_status("TQNN Classifier ready.\nSelect a pattern and adjust noise.")

    def _update_status(self, msg: str) -> None:
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert(tk.END, msg)
        self.status_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Right panel: 6 matplotlib panels
    # ------------------------------------------------------------------

    def _build_plots(self, parent: ttk.Frame) -> None:
        self.fig = Figure(figsize=(13, 10), facecolor=DARK_BG)
        gs = self.fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32,
                                   top=0.94, bottom=0.06, left=0.06, right=0.97)

        self.ax_pattern  = self.fig.add_subplot(gs[0, 0])
        self.ax_braiding = self.fig.add_subplot(gs[0, 1])
        self.ax_charge   = self.fig.add_subplot(gs[0, 2])
        self.ax_classif  = self.fig.add_subplot(gs[1, 0])
        self.ax_spin     = self.fig.add_subplot(gs[1, 1])
        self.ax_robust   = self.fig.add_subplot(gs[1, 2])

        for ax in (self.ax_pattern, self.ax_braiding, self.ax_charge,
                   self.ax_classif, self.ax_spin, self.ax_robust):
            ax.set_facecolor(DARK_AXES)
            ax.tick_params(colors=DARK_TEXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(DARK_EDGE)

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Drawing helpers (called every tick)
    # ------------------------------------------------------------------

    def _draw_all(self) -> None:
        self._draw_pattern()
        self._draw_braiding()
        self._draw_charge()
        self._draw_classification()
        self._draw_spin_network()
        self._draw_robustness()
        self.canvas.draw_idle()
        self._refresh_status()

    # 1 - Input pattern
    def _draw_pattern(self) -> None:
        ax = self.ax_pattern; ax.clear(); ax.set_facecolor(DARK_AXES)
        if self.sim.noisy_pattern is not None:
            ax.imshow(self.sim.noisy_pattern, cmap="mako", interpolation="nearest", aspect="equal")
            ax.text(0.02, 0.98, f"Noise: {self.sim.current_noise:.2f}",
                    transform=ax.transAxes, fontsize=9, va="top", color=DARK_TEXT,
                    bbox=dict(boxstyle="round", facecolor=DARK_BG, edgecolor=DARK_EDGE, alpha=0.9))
            ax.text(0.02, 0.02, f"Pattern: {self.sim.current_pattern_label}",
                    transform=ax.transAxes, fontsize=9, va="bottom", color=DARK_TEXT,
                    bbox=dict(boxstyle="round", facecolor=DARK_BG, edgecolor=DARK_ACCENT, alpha=0.9))
        ax.set_title("Input Pattern + Noise", fontsize=10, fontweight="bold", color=DARK_TEXT)
        ax.set_xticks([]); ax.set_yticks([])

    # 2 - Anyonic braiding
    def _draw_braiding(self) -> None:
        ax = self.ax_braiding; ax.clear(); ax.set_facecolor(DARK_AXES)
        colors = altCmap(np.linspace(0.15, 0.90, BRAIDING_STRANDS))
        t = self.sim.braiding_t
        for i in range(BRAIDING_STRANDS):
            x = i * 0.8 - 2.0
            y_off = 0.45 * np.sin(0.12 * t + i * 0.55)
            ax.plot([x, x], [0, 2], color=colors[i], linewidth=3.5,
                    solid_capstyle="round", alpha=0.75)
            ax.add_patch(Circle((x, 1.0 + y_off), 0.10, color=colors[i], zorder=5))
            ax.text(x, -0.18, f"|q{i}>", ha="center", va="top", fontsize=8, color=DARK_TEXT)
        phase = (t // 45) % 3
        if phase == 1:
            ax.plot([-1, 1], [1.5, 1.5], "--", color=DARK_ACCENT, lw=2, alpha=0.7)
            ax.text(0, 1.72, "Braiding", ha="center", fontsize=9, color=DARK_TEXT,
                    bbox=dict(boxstyle="round", facecolor=DARK_BG, edgecolor=DARK_ACCENT, alpha=0.85))
        ax.set_xlim(-2.8, 2.8); ax.set_ylim(-0.5, 2.5); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Anyonic Braiding", fontsize=10, fontweight="bold", color=DARK_TEXT)

    # 3 - Charge-flow network
    def _draw_charge(self) -> None:
        ax = self.ax_charge; ax.clear(); ax.set_facecolor(DARK_AXES)
        G = nx.DiGraph()
        pos = {"in": (0, 1), "h1": (1, 1.6), "h2": (1, 0.4), "out": (2, 1)}
        q = self.sim.topological_charge
        charges = {"in": q, "h1": q * 0.6, "h2": q * 0.4, "out": q}
        G.add_edges_from([("in", "h1"), ("in", "h2"), ("h1", "out"), ("h2", "out")])
        nc = [divCmap(abs(charges[n])) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=nc, node_size=700, alpha=0.85, ax=ax,
                               edgecolors=DARK_EDGE, linewidths=1.5)
        pulse = 0.5 + 0.5 * np.sin(self.sim.charge_t * 0.15)
        ec = [seqCmap(pulse)] * len(G.edges())
        nx.draw_networkx_edges(G, pos, edge_color=ec, width=2.5, alpha=0.85, ax=ax,
                               arrows=True, arrowsize=18, arrowstyle="->",
                               connectionstyle="arc3,rad=0.1")
        for node, (x, y) in pos.items():
            ax.text(x, y - 0.28, f"Q={charges[node]:.2f}", ha="center", fontsize=7, color=DARK_TEXT)
        ax.set_title("Charge Flow Network", fontsize=10, fontweight="bold", color=DARK_TEXT)
        ax.axis("off")

    # 4 - Classification confidence
    def _draw_classification(self) -> None:
        ax = self.ax_classif; ax.clear(); ax.set_facecolor(DARK_AXES)
        if self.sim.log_probs:
            labels = list(self.sim.log_probs.keys())
            vals = list(self.sim.log_probs.values())
            bar_colors = [DARK_ACCENT if lb == self.sim.prediction else seqCmap(0.35) for lb in labels]
            ax.barh(labels, vals, color=bar_colors, edgecolor=DARK_EDGE, linewidth=0.8)
            ax.set_xlabel("Log Probability", fontsize=8, color=DARK_TEXT)
            ax.text(0.98, 0.02, f"Prediction: {self.sim.prediction}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    color=DARK_ACCENT, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor=DARK_BG, edgecolor=DARK_ACCENT, alpha=0.9))
        ax.set_title("Classification Confidence", fontsize=10, fontweight="bold", color=DARK_TEXT)
        ax.tick_params(colors=DARK_TEXT, labelsize=7)
        ax.grid(axis="x", color=DARK_GRID, alpha=0.4)

    # 5 - Spin-network representation
    def _draw_spin_network(self) -> None:
        ax = self.ax_spin; ax.clear(); ax.set_facecolor(DARK_AXES)
        if self.sim.noisy_pattern is None:
            return
        spins = create_spin_network_from_pattern(self.sim.noisy_pattern)
        n = int(np.sqrt(len(spins)))
        if n == 0:
            return
        grid = (spins[:n*n] - 1000).reshape(n, n)  # relative to N_LARGE

        # Hexagonal-ish node layout
        positions = {}
        idx = 0
        for r in range(n):
            for c in range(n):
                x = c + 0.5 * (r % 2)
                y = r * 0.87
                positions[idx] = (x, y)
                idx += 1
        G = nx.Graph()
        G.add_nodes_from(range(n * n))
        for r in range(n):
            for c in range(n):
                node = r * n + c
                if c + 1 < n:
                    G.add_edge(node, node + 1)
                if r + 1 < n:
                    G.add_edge(node, node + n)

        node_vals = grid.flatten()
        vmin, vmax = node_vals.min(), max(node_vals.max(), 1)
        node_colors = [seqCmap((v - vmin) / (vmax - vmin + 1e-9)) for v in node_vals]
        nx.draw_networkx_nodes(G, positions, node_color=node_colors, node_size=55,
                               ax=ax, edgecolors=DARK_EDGE, linewidths=0.5)
        nx.draw_networkx_edges(G, positions, edge_color="#555555", width=0.6, alpha=0.5, ax=ax)
        ax.set_title("Spin-Network $j_i = N + \\lfloor x_i \\rfloor$",
                     fontsize=10, fontweight="bold", color=DARK_TEXT)
        ax.axis("off")

    # 6 - Robustness sweep
    def _draw_robustness(self) -> None:
        ax = self.ax_robust; ax.clear(); ax.set_facecolor(DARK_AXES)
        levels, curves = self.sim.robustness_sweep(25)
        palette = sns.color_palette("mako", n_colors=len(self.sim.labels))
        for i, lb in enumerate(self.sim.labels):
            style = "-" if lb == self.sim.current_pattern_label else "--"
            ax.plot(levels, curves[lb], style, color=palette[i], lw=2, label=lb)
        # Mark current noise
        ax.axvline(self.sim.current_noise, color=DARK_ACCENT, ls=":", lw=1.5, alpha=0.7)
        ax.legend(fontsize=7, facecolor=DARK_BG, edgecolor=DARK_EDGE, labelcolor=DARK_TEXT)
        ax.set_xlabel("Noise Level", fontsize=8, color=DARK_TEXT)
        ax.set_ylabel("Log Probability", fontsize=8, color=DARK_TEXT)
        ax.set_title("Topological Robustness", fontsize=10, fontweight="bold", color=DARK_TEXT)
        ax.tick_params(colors=DARK_TEXT, labelsize=7)
        ax.grid(color=DARK_GRID, alpha=0.4)

    # ------------------------------------------------------------------
    # Status text refresh
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        lines = [
            "=" * 50,
            "  TQNN INTERACTIVE CLASSIFIER",
            "=" * 50,
            "",
            f"  Pattern : {self.sim.current_pattern_label}",
            f"  Noise   : {self.sim.current_noise:.2f}",
            f"  Predict : {self.sim.prediction}",
            f"  Charge  : {self.sim.topological_charge:.3f}",
            "",
        ]
        if self.sim.log_probs:
            lines.append("  Class Log-Probabilities:")
            for lb, lp in self.sim.log_probs.items():
                marker = " <--" if lb == self.sim.prediction else ""
                lines.append(f"    {lb:12s} {lp:12.2f}{marker}")
        lines += [
            "",
            "  Formulas:",
            "  j_i = N + floor(x_i)          (spin encoding)",
            "  A = prod Delta_j exp(-(j-j_bar)^2/2sigma^2)",
            "  prediction = argmax_c log|A_c|^2",
        ]
        self._update_status("\n".join(lines))

    # ------------------------------------------------------------------
    # Animation loop
    # ------------------------------------------------------------------

    def _schedule_tick(self) -> None:
        self.sim.tick()
        # Auto noise sweep
        if self.auto_var.get():
            nv = self.noise_var.get() + 0.005
            if nv > MAX_NOISE:
                nv = 0.0
            self.noise_var.set(nv)
            self.sim.set_noise(nv)
        self._draw_all()
        self.root.after(ANIMATION_MS, self._schedule_tick)

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def _take_screenshot(self) -> None:
        self._draw_all()
        self.fig.savefig(self.screenshot_path, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
        print(f"Screenshot saved to {self.screenshot_path}")
        self.root.destroy()

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._update_status(
            "=" * 50 + "\n"
            "  TQNN INTERACTIVE CLASSIFIER READY\n"
            "=" * 50 + "\n\n"
            "  Select a pattern and adjust the noise slider.\n"
            "  Observe how topological protection preserves\n"
            "  correct classification under local defects.\n\n"
            "  Key formulas:\n"
            "  j_i = N + floor(x_i)     (spin encoding)\n"
            "  A_c = prod Delta_j exp(-(j - j_bar)^2 / 2sigma^2)\n"
            "  prediction = argmax_c log|A_c|^2\n"
            "=" * 50
        )
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="TQNN Interactive Classifier")
    parser.add_argument("--screenshot", type=str, default=None,
                        help="Save a screenshot to the given path and exit")
    args = parser.parse_args()

    app = TQNNClassifierGUI(screenshot_path=args.screenshot)
    app.run()


if __name__ == "__main__":
    main()
