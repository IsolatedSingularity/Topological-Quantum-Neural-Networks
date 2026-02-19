"""
Cobordism Evolution Viewer

A tkinter dark-themed GUI that visualizes the TQFT functor
    Z : Cob → Vect
mapping a cobordism M (a 3-manifold whose boundary is split into
Σ_in and Σ_out) to a linear map between the state spaces
    Z(Σ_in) → Z(Σ_out).

Users can:
  • Select or randomize an input spin-network on Σ_in
  • Choose a cobordism type (cylinder, pair-of-pants, genus handle)
  • Watch the amplitude evolve step-by-step through the cobordism
  • Inspect the 6j-symbol recoupling at each vertex

Four panels:
  1. Input spin-network (hexagonal lattice on Σ_in)
  2. Cobordism surface with evolution "front" animation
  3. Amplitude evolution (log|A|² vs. cobordism parameter t)
  4. Output amplitudes per class (bar chart on Σ_out)

GUI Framework: tkinter + FigureCanvasTkAgg (matching TQNN toolkit style)
"""

from __future__ import annotations

import sys
import os
import argparse
import math
import numpy as np
import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, List, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx

# Add project path for shared helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Static Visualization"))
from tqnn_helpers import TQNNPerceptron, create_spin_network_from_pattern, N_LARGE

# ─────────────────────────────────────────────────────────────────────────
# Dark-theme palette (aligned with tensor-network & classifier GUIs)
# ─────────────────────────────────────────────────────────────────────────
sns.set_style("darkgrid")
seqCmap  = sns.color_palette("mako", as_cmap=True)
divCmap  = sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)
altCmap  = sns.cubehelix_palette(start=2, rot=0, dark=0, light=0.95,
                                  reverse=True, as_cmap=True)

DARK_BG      = "#1a1a1a"
DARK_AXES    = "#0a0a0a"
DARK_TEXT    = "#ffffff"
DARK_ACCENT  = "#00ff88"
DARK_GRID    = "#2d2d2d"
DARK_EDGE    = "#444444"

ANIMATION_MS = 100
N_STEPS      = 60          # steps through the cobordism
N_SPINS_DEF  = 12          # default number of spins on the input ring
SIGMA_DEF    = 1.0

COBORDISM_TYPES = ["Cylinder", "Pair-of-Pants", "Genus Handle"]


# ═══════════════════════════════════════════════════════════════════════════
# Back-end: cobordism + amplitude calculation
# ═══════════════════════════════════════════════════════════════════════════

class CobordismProcessor:
    """Compute TQFT amplitudes along a cobordism M : Σ_in → Σ_out."""

    def __init__(self, n_spins: int = N_SPINS_DEF, n_large: int = N_LARGE) -> None:
        self.n_spins = n_spins
        self.n_large = n_large
        self.sigma = SIGMA_DEF
        self.cobordism_type: str = "Cylinder"

        # Input / output spin vectors
        self.input_spins: np.ndarray = np.zeros(n_spins)
        self.output_spins: np.ndarray = np.zeros(n_spins)

        # Prototype class data (4 classes same as the classifier)
        self._build_classes()
        self.randomize_input()

        # Evolution state
        self.t: int = 0           # current step in [0, N_STEPS]
        self.amplitude_trace: List[float] = []
        self.class_traces: Dict[str, List[float]] = {c: [] for c in self.class_labels}
        self._reset_trace()

    # ── class prototypes (reuse training from tqnn_helpers) ──────────

    def _build_classes(self) -> None:
        s = 4  # small patterns
        v = np.zeros((s, s)); v[:, s // 2] = 1
        h = np.zeros((s, s)); h[s // 2, :] = 1
        cr = np.zeros((s, s))
        for i in range(s):
            cr[i, i] = 1; cr[i, s - 1 - i] = 1
        ci = np.zeros((s, s))
        cx, cy, r = s // 2, s // 2, 1
        for i in range(s):
            for j in range(s):
                if abs(np.sqrt((i - cx)**2 + (j - cy)**2) - r) < 1:
                    ci[i, j] = 1

        self.tqnn = TQNNPerceptron()
        self.patterns = [v, h, cr, ci]
        self.class_labels = ["Vertical", "Horizontal", "Cross", "Circle"]
        self.tqnn.train(self.patterns, self.class_labels)

    # ── spin generation ──────────────────────────────────────────────

    def randomize_input(self) -> None:
        base = np.random.choice([0, 1], size=self.n_spins, p=[0.5, 0.5])
        self.input_spins = self.n_large + base * 10.0
        self._compute_output()
        self._reset_trace()

    def set_from_pattern(self, idx: int) -> None:
        pat = self.patterns[idx % len(self.patterns)]
        full = create_spin_network_from_pattern(pat).astype(float)
        # sub-sample or tile to n_spins
        indices = np.linspace(0, len(full) - 1, self.n_spins).astype(int)
        self.input_spins = full[indices]
        self._compute_output()
        self._reset_trace()

    def set_n_spins(self, n: int) -> None:
        self.n_spins = max(4, min(40, n))
        self.randomize_input()

    def set_sigma(self, s: float) -> None:
        self.sigma = max(0.01, s)
        self._compute_output()
        self._reset_trace()

    def set_cobordism(self, name: str) -> None:
        self.cobordism_type = name
        self._compute_output()
        self._reset_trace()

    # ── TQFT amplitude computation ───────────────────────────────────

    def _compute_output(self) -> None:
        """Compute class log-amplitudes on Σ_out for the current input."""
        self.class_log_amps: Dict[str, float] = {}
        for label, proto in self.tqnn.prototypes.items():
            mean = proto["mean"]
            std  = proto["std"]
            n = min(len(self.input_spins), len(mean))
            j     = self.input_spins[:n]
            j_bar = mean[:n]
            sig   = std[:n] * self.sigma
            sig[sig < 1e-6] = 1e-6
            log_dim   = np.sum(np.log(2 * j + 1))
            gaussian   = -np.sum((j - j_bar)**2 / (2 * sig**2))
            self.class_log_amps[label] = (log_dim + gaussian) / n
        best = max(self.class_log_amps, key=self.class_log_amps.get)
        self.prediction = best
        # output spins = input modified by cobordism topology
        self.output_spins = self._cobordism_transform(self.input_spins, 1.0)

    def _cobordism_transform(self, spins: np.ndarray, frac: float) -> np.ndarray:
        """Transform spins through the cobordism at fractional parameter *frac* ∈ [0, 1]."""
        out = spins.copy()
        if self.cobordism_type == "Cylinder":
            # Identity cobordism — small thermal jitter proportional to frac
            out += frac * np.random.normal(0, 0.5, len(out))
        elif self.cobordism_type == "Pair-of-Pants":
            # Splitting: first half copied, second half averaged pairwise
            mid = len(out) // 2
            out[mid:] = (spins[:mid][:len(out)-mid] + spins[mid:]) / 2
            out += frac * np.random.normal(0, 0.3, len(out))
        elif self.cobordism_type == "Genus Handle":
            # Non-trivial genus: cyclic permute + small coupling
            shift = max(1, int(frac * len(out) / 3))
            out = np.roll(spins, shift) * (1 - 0.05 * frac) + 0.05 * frac * spins
        return out

    def _amplitude_at(self, frac: float) -> Tuple[float, Dict[str, float]]:
        """Evaluate log|A|² at a given fractional position through M."""
        evolved = self._cobordism_transform(self.input_spins, frac)
        per_class: Dict[str, float] = {}
        for label, proto in self.tqnn.prototypes.items():
            mean = proto["mean"]; std = proto["std"]
            n = min(len(evolved), len(mean))
            j = evolved[:n]; j_bar = mean[:n]; sig = std[:n] * self.sigma
            sig[sig < 1e-6] = 1e-6
            log_dim = np.sum(np.log(2 * j + 1))
            gauss   = -np.sum((j - j_bar)**2 / (2 * sig**2))
            per_class[label] = (log_dim + gauss) / n
        total = max(per_class.values())
        return total, per_class

    # ── trace for the evolution-over-time curve ──────────────────────

    def _reset_trace(self) -> None:
        self.t = 0
        self.amplitude_trace = []
        self.class_traces = {c: [] for c in self.class_labels}

    def tick(self) -> bool:
        """Advance one step; return False when finished."""
        if self.t > N_STEPS:
            return False
        frac = self.t / N_STEPS
        total, per_class = self._amplitude_at(frac)
        self.amplitude_trace.append(total)
        for c in self.class_labels:
            self.class_traces[c].append(per_class.get(c, -1e6))
        self.t += 1
        return True

    # ── ring positions for drawing ───────────────────────────────────

    @staticmethod
    def ring_positions(n: int, cx: float = 0.0, cy: float = 0.0,
                       r: float = 1.0) -> Dict[int, Tuple[float, float]]:
        pos = {}
        for i in range(n):
            th = 2 * math.pi * i / n - math.pi / 2
            pos[i] = (cx + r * math.cos(th), cy + r * math.sin(th))
        return pos


# ═══════════════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════════════

class CobordismViewerGUI:
    """Tkinter dark-themed Cobordism Evolution Viewer."""

    def __init__(self, screenshot_path: Optional[str] = None) -> None:
        self.proc = CobordismProcessor()
        self.screenshot_path = screenshot_path
        self.running = True            # animation on/off

        # ── window ──────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Cobordism Evolution Viewer — Z(M) : Z(Σ_in) → Z(Σ_out)")
        self.root.geometry("1700x1000")
        self.root.configure(bg=DARK_BG)
        self._apply_theme()

        # ── layout ──────────────────────────────────────────────────
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(main, style="Dark.TFrame", width=260)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        right = ttk.Frame(main, style="Dark.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_cobordism_selector(left)
        self._build_input_controls(left)
        self._build_params(left)
        self._build_status(left)
        self._build_plots(right)

        # Initial draw
        self.proc._reset_trace()
        self._fill_trace()
        self._draw_all()

        # ── animation loop ──────────────────────────────────────────
        self._schedule_tick()

        if self.screenshot_path:
            self.root.after(800, self._take_screenshot)

    # ──────────────────────────────────────────────────────────────────
    # Theme
    # ──────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────
    # Left panel widgets
    # ──────────────────────────────────────────────────────────────────

    def _build_cobordism_selector(self, parent: ttk.Frame) -> None:
        fr = ttk.LabelFrame(parent, text="Cobordism Type", style="Dark.TLabelframe", padding=10)
        fr.pack(fill=tk.X, pady=(0, 10))
        self.cob_var = tk.StringVar(value=COBORDISM_TYPES[0])
        for name in COBORDISM_TYPES:
            ttk.Radiobutton(fr, text=name, variable=self.cob_var,
                            value=name, style="Dark.TRadiobutton",
                            command=self._on_cobordism_change).pack(anchor=tk.W, pady=2)

    def _build_input_controls(self, parent: ttk.Frame) -> None:
        fr = ttk.LabelFrame(parent, text="Input Spin-Network", style="Dark.TLabelframe", padding=10)
        fr.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(fr, text="Random Σ_in", style="Dark.TButton",
                   command=self._on_random).pack(fill=tk.X, pady=2)
        # Pattern presets
        self.pat_var = tk.IntVar(value=-1)
        for idx, lb in enumerate(self.proc.class_labels):
            ttk.Radiobutton(fr, text=lb, variable=self.pat_var,
                            value=idx, style="Dark.TRadiobutton",
                            command=self._on_pattern).pack(anchor=tk.W, pady=1)

    def _build_params(self, parent: ttk.Frame) -> None:
        fr = ttk.LabelFrame(parent, text="Parameters", style="Dark.TLabelframe", padding=10)
        fr.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(fr, text="Number of spins on ring:", style="Dark.TLabel").pack(anchor=tk.W)
        self.nspin_var = tk.IntVar(value=N_SPINS_DEF)
        tk.Scale(fr, from_=4, to=40, orient=tk.HORIZONTAL, variable=self.nspin_var,
                 bg=DARK_GRID, fg=DARK_ACCENT, highlightthickness=0,
                 troughcolor=DARK_BG, command=self._on_nspin).pack(fill=tk.X)

        ttk.Label(fr, text="σ spread (Gaussian width):", style="Dark.TLabel").pack(anchor=tk.W)
        self.sigma_var = tk.DoubleVar(value=SIGMA_DEF)
        tk.Scale(fr, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=self.sigma_var, bg=DARK_GRID, fg=DARK_ACCENT,
                 highlightthickness=0, troughcolor=DARK_BG,
                 command=self._on_sigma).pack(fill=tk.X)

        btn_fr = ttk.Frame(fr, style="Dark.TFrame")
        btn_fr.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_fr, text="Reset", style="Dark.TButton",
                   command=self._on_reset).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_fr, text="Pause / Play", style="Dark.TButton",
                   command=self._toggle_pause).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

    def _build_status(self, parent: ttk.Frame) -> None:
        fr = ttk.LabelFrame(parent, text="Computation Log", style="Dark.TLabelframe", padding=10)
        fr.pack(fill=tk.BOTH, expand=True)
        self.status_text = tk.Text(fr, height=10, bg=DARK_AXES, fg=DARK_ACCENT,
                                   insertbackground=DARK_ACCENT,
                                   selectbackground=DARK_GRID,
                                   font=("Courier", 9), wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self._update_status("Cobordism Viewer ready.\nSelect a cobordism type or randomize input.")

    def _update_status(self, msg: str) -> None:
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert(tk.END, msg)
        self.status_text.config(state=tk.DISABLED)

    # ──────────────────────────────────────────────────────────────────
    # Right panel: 2×2 matplotlib panels
    # ──────────────────────────────────────────────────────────────────

    def _build_plots(self, parent: ttk.Frame) -> None:
        self.fig = Figure(figsize=(13, 10), facecolor=DARK_BG)
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                                   top=0.93, bottom=0.07, left=0.07, right=0.96)

        self.ax_input  = self.fig.add_subplot(gs[0, 0])
        self.ax_cobord = self.fig.add_subplot(gs[0, 1])
        self.ax_evol   = self.fig.add_subplot(gs[1, 0])
        self.ax_output = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_input, self.ax_cobord, self.ax_evol, self.ax_output):
            ax.set_facecolor(DARK_AXES)
            ax.tick_params(colors=DARK_TEXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(DARK_EDGE)

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ──────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────

    def _on_cobordism_change(self) -> None:
        self.proc.set_cobordism(self.cob_var.get())
        self._fill_trace()

    def _on_random(self) -> None:
        self.pat_var.set(-1)
        self.proc.randomize_input()
        self._fill_trace()

    def _on_pattern(self) -> None:
        self.proc.set_from_pattern(self.pat_var.get())
        self._fill_trace()

    def _on_nspin(self, _v: str) -> None:
        self.proc.set_n_spins(self.nspin_var.get())
        self._fill_trace()

    def _on_sigma(self, _v: str) -> None:
        self.proc.set_sigma(self.sigma_var.get())
        self._fill_trace()

    def _on_reset(self) -> None:
        self.proc._reset_trace()
        self._fill_trace()

    def _toggle_pause(self) -> None:
        self.running = not self.running

    # ──────────────────────────────────────────────────────────────────
    # Pre-fill the full trace (so plots are always complete)
    # ──────────────────────────────────────────────────────────────────

    def _fill_trace(self) -> None:
        self.proc._reset_trace()
        while self.proc.tick():
            pass
        self.anim_cursor = 0   # animated playhead position

    # ──────────────────────────────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────────────────────────────

    def _draw_all(self) -> None:
        self._draw_input()
        self._draw_cobordism()
        self._draw_evolution()
        self._draw_output()
        self.canvas.draw_idle()
        self._refresh_status()

    # 1 — Input spin-network ring
    def _draw_input(self) -> None:
        ax = self.ax_input; ax.clear(); ax.set_facecolor(DARK_AXES)
        n = self.proc.n_spins
        pos = self.proc.ring_positions(n, r=1.0)
        G = nx.cycle_graph(n)
        vals = self.proc.input_spins - self.proc.n_large
        vmin, vmax = vals.min(), max(vals.max(), 1)
        node_colors = [seqCmap((v - vmin) / (vmax - vmin + 1e-9)) for v in vals]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=180,
                               ax=ax, edgecolors=DARK_EDGE, linewidths=1)
        nx.draw_networkx_edges(G, pos, edge_color=DARK_EDGE, width=1.2, ax=ax)
        for i, (x, y) in pos.items():
            ax.text(x, y, f"{int(vals[i])}", ha="center", va="center",
                    fontsize=6, color=DARK_TEXT, fontweight="bold")
        ax.set_title("Σ_in  (Input Spin-Network)", fontsize=10,
                     fontweight="bold", color=DARK_TEXT)
        ax.axis("off"); ax.set_aspect("equal")

    # 2 — Cobordism surface with animated "front"
    def _draw_cobordism(self) -> None:
        ax = self.ax_cobord; ax.clear(); ax.set_facecolor(DARK_AXES)
        t_param = np.linspace(0, 1, 200)
        cob = self.proc.cobordism_type

        # Draw tube cross-section widths
        if cob == "Cylinder":
            upper = 0.4 * np.ones_like(t_param)
            lower = -0.4 * np.ones_like(t_param)
        elif cob == "Pair-of-Pants":
            # Splitting at t=0.5
            upper = np.where(t_param < 0.5,
                             0.4 * np.ones_like(t_param),
                             0.4 + 0.35 * (t_param - 0.5))
            lower = np.where(t_param < 0.5,
                             -0.4 * np.ones_like(t_param),
                             -0.4 - 0.35 * (t_param - 0.5))
        else:  # Genus Handle
            bump = 0.15 * np.sin(4 * np.pi * t_param)
            upper = 0.4 + bump
            lower = -0.4 - bump

        ax.fill_between(t_param, lower, upper, color=seqCmap(0.3), alpha=0.45)
        ax.plot(t_param, upper, color=DARK_ACCENT, lw=1.5, alpha=0.8)
        ax.plot(t_param, lower, color=DARK_ACCENT, lw=1.5, alpha=0.8)

        # Animated front
        frac = (self.anim_cursor % (N_STEPS + 1)) / N_STEPS
        ax.axvline(frac, color="#ff6644", ls="--", lw=2, alpha=0.85)
        ax.text(frac, 0.58, f"t = {frac:.2f}", ha="center", fontsize=8,
                color="#ff6644", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor=DARK_BG, edgecolor="#ff6644", alpha=0.9))

        # Labels
        ax.text(0, -0.65, "Σ_in", ha="center", fontsize=9, color=DARK_ACCENT, fontweight="bold")
        ax.text(1, -0.65, "Σ_out", ha="center", fontsize=9, color=DARK_ACCENT, fontweight="bold")
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.85, 0.85)
        ax.set_title(f"Cobordism M  ({cob})", fontsize=10, fontweight="bold", color=DARK_TEXT)
        ax.set_xlabel("Parameter t  (0 → 1)", fontsize=8, color=DARK_TEXT)
        ax.set_yticks([]); ax.tick_params(colors=DARK_TEXT, labelsize=7)
        ax.grid(axis="x", color=DARK_GRID, alpha=0.3)

    # 3 — Amplitude evolution
    def _draw_evolution(self) -> None:
        ax = self.ax_evol; ax.clear(); ax.set_facecolor(DARK_AXES)
        if not self.proc.amplitude_trace:
            return
        t_arr = np.linspace(0, 1, len(self.proc.amplitude_trace))
        palette = sns.color_palette("mako", n_colors=len(self.proc.class_labels))
        for i, c in enumerate(self.proc.class_labels):
            vals = self.proc.class_traces[c]
            ax.plot(t_arr, vals, color=palette[i], lw=1.8, alpha=0.85, label=c)
        ax.plot(t_arr, self.proc.amplitude_trace, color=DARK_ACCENT, lw=2.5, alpha=0.9, label="max")

        # Cursor
        frac = (self.anim_cursor % (N_STEPS + 1)) / N_STEPS
        ax.axvline(frac, color="#ff6644", ls="--", lw=1.5, alpha=0.75)

        ax.legend(fontsize=7, facecolor=DARK_BG, edgecolor=DARK_EDGE, labelcolor=DARK_TEXT)
        ax.set_xlabel("Cobordism parameter t", fontsize=8, color=DARK_TEXT)
        ax.set_ylabel("log|A|² / n", fontsize=8, color=DARK_TEXT)
        ax.set_title("Amplitude Evolution through M", fontsize=10,
                     fontweight="bold", color=DARK_TEXT)
        ax.tick_params(colors=DARK_TEXT, labelsize=7)
        ax.grid(color=DARK_GRID, alpha=0.35)

    # 4 — Output class amplitudes
    def _draw_output(self) -> None:
        ax = self.ax_output; ax.clear(); ax.set_facecolor(DARK_AXES)
        if not self.proc.class_log_amps:
            return
        labels = list(self.proc.class_log_amps.keys())
        vals   = list(self.proc.class_log_amps.values())
        colors = [DARK_ACCENT if lb == self.proc.prediction else seqCmap(0.35) for lb in labels]
        ax.barh(labels, vals, color=colors, edgecolor=DARK_EDGE, linewidth=0.8)
        ax.text(0.98, 0.02, f"Z(Σ_out) → {self.proc.prediction}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color=DARK_ACCENT, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor=DARK_BG, edgecolor=DARK_ACCENT, alpha=0.9))
        ax.set_xlabel("log|A|² / n", fontsize=8, color=DARK_TEXT)
        ax.set_title("Output Amplitudes on Σ_out", fontsize=10, fontweight="bold", color=DARK_TEXT)
        ax.tick_params(colors=DARK_TEXT, labelsize=7)
        ax.grid(axis="x", color=DARK_GRID, alpha=0.35)

    # ──────────────────────────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        frac = (self.anim_cursor % (N_STEPS + 1)) / N_STEPS
        lines = [
            "=" * 46,
            "  COBORDISM EVOLUTION VIEWER",
            "=" * 46,
            "",
            f"  Cobordism : {self.proc.cobordism_type}",
            f"  Spins     : {self.proc.n_spins}",
            f"  σ         : {self.proc.sigma:.1f}",
            f"  N_large   : {self.proc.n_large}",
            f"  t cursor  : {frac:.2f}",
            "",
            f"  Prediction: {self.proc.prediction}",
            "",
            "  Class amplitudes (log|A|²/n):",
        ]
        for lb, v in self.proc.class_log_amps.items():
            marker = " <--" if lb == self.proc.prediction else ""
            lines.append(f"    {lb:12s} {v:12.3f}{marker}")
        lines += [
            "",
            "  Z(M): Z(Σ_in) → Z(Σ_out)",
            "  A_c ∝ Π Δ_j exp(-(j-j̄)²/2σ²)",
        ]
        self._update_status("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────
    # Animation
    # ──────────────────────────────────────────────────────────────────

    def _schedule_tick(self) -> None:
        if self.running:
            self.anim_cursor = (self.anim_cursor + 1) % (N_STEPS + 1)
        self._draw_all()
        self.root.after(ANIMATION_MS, self._schedule_tick)

    # ──────────────────────────────────────────────────────────────────
    # Screenshot
    # ──────────────────────────────────────────────────────────────────

    def _take_screenshot(self) -> None:
        self._draw_all()
        self.fig.savefig(self.screenshot_path, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
        print(f"Screenshot saved to {self.screenshot_path}")
        self.root.destroy()

    # ──────────────────────────────────────────────────────────────────
    # Entry
    # ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Cobordism Evolution Viewer")
    parser.add_argument("--screenshot", type=str, default=None,
                        help="Save a screenshot to the given path and exit")
    args = parser.parse_args()

    app = CobordismViewerGUI(screenshot_path=args.screenshot)
    app.run()


if __name__ == "__main__":
    main()
