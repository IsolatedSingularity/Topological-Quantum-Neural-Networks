"""
Interactive TQNN (Topological Quantum Neural Network) GUI

This script provides an interactive visualization of key TQNN concepts:
- Topological braiding of anyonic world-lines
- Charge conservation in topological networks
- Pattern classification with noise robustness
- Semi-classical limit behavior
- Real-time interaction with TQNN perceptron

Users can:
- Inject topological defects (noise) into patterns
- Watch real-time classification confidence changes
- Adjust network parameters dynamically
- Observe anyonic braiding patterns
- Explore charge flow through the network
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon
import seaborn as sns
import networkx as nx
from collections import deque
import os
import sys

# Add the Code directory to path to import tqnn_helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Code'))
from tqnn_helpers import TQNNPerceptron, add_topological_defect, create_spin_network_from_pattern

# --- Visualization Parameters ---
seqCmap = sns.color_palette("mako", as_cmap=True)
divCmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
altCmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)

# Dark theme constants (matching tensor network simulator)
DARK_BG = '#1a1a1a'
DARK_AXES = '#0a0a0a'
DARK_TEXT = '#ffffff'
DARK_ACCENT = '#00ff88'
DARK_GRID = '#2d2d2d'
DARK_EDGE = '#444444'
DARK_SUBTITLE = '#aaaaaa'

# --- Simulation Parameters ---
ANIMATION_INTERVAL = 150  # Milliseconds between frames
PATTERN_SIZE = 8  # Size of input patterns
MAX_NOISE_LEVEL = 0.5
BRAIDING_STRANDS = 6


class TQNNSimulator:
    """
    Core TQNN simulation with topological properties
    """
    def __init__(self, pattern_size=PATTERN_SIZE):
        self.pattern_size = pattern_size
        self.current_pattern = None
        self.current_noise_level = 0.0
        self.noisy_pattern = None
        
        # Initialize TQNN perceptron
        self.tqnn = TQNNPerceptron()
        self.patterns, self.labels = self._create_training_patterns()
        self.tqnn.train(self.patterns, self.labels)
        
        # Current prediction state
        self.prediction_result = None
        self.log_probabilities = {}
        self.confidence_history = deque(maxlen=50)
        
        # Topological properties
        self.charge_conservation = True
        self.braiding_configuration = np.arange(BRAIDING_STRANDS)  # Initial ordering
        self.topological_charge = 0.0
        
        # Animation states
        self.braiding_time = 0
        self.charge_flow_time = 0
        
        # Select initial pattern
        self.select_pattern(0)
        
    def _create_training_patterns(self):
        """Create simple geometric patterns for training"""
        patterns = []
        labels = []
        
        # Vertical line pattern
        vertical = np.zeros((self.pattern_size, self.pattern_size))
        vertical[:, self.pattern_size // 2] = 1
        patterns.append(vertical)
        labels.append("Vertical")
        
        # Horizontal line pattern
        horizontal = np.zeros((self.pattern_size, self.pattern_size))
        horizontal[self.pattern_size // 2, :] = 1
        patterns.append(horizontal)
        labels.append("Horizontal")
        
        # Cross pattern
        cross = np.zeros((self.pattern_size, self.pattern_size))
        for i in range(self.pattern_size):
            cross[i, i] = 1
            cross[i, self.pattern_size - 1 - i] = 1
        patterns.append(cross)
        labels.append("Cross")
        
        # Circle pattern
        circle = np.zeros((self.pattern_size, self.pattern_size))
        center = self.pattern_size // 2
        radius = self.pattern_size // 3
        for i in range(self.pattern_size):
            for j in range(self.pattern_size):
                if abs(np.sqrt((i - center)**2 + (j - center)**2) - radius) < 1:
                    circle[i, j] = 1
        patterns.append(circle)
        labels.append("Circle")
        
        return patterns, labels
    
    def select_pattern(self, pattern_idx):
        """Select a pattern for testing"""
        if 0 <= pattern_idx < len(self.patterns):
            self.current_pattern = self.patterns[pattern_idx].copy()
            self.current_pattern_idx = pattern_idx
            self.current_pattern_label = self.labels[pattern_idx]
            self.update_noise()
    
    def update_noise(self):
        """Update noisy pattern and classification"""
        if self.current_pattern is not None:
            self.noisy_pattern = add_topological_defect(self.current_pattern, self.current_noise_level)
            self.prediction_result, self.log_probabilities = self.tqnn.predict(self.noisy_pattern)
            
            # Update confidence history
            if self.prediction_result:
                confidence = max(self.log_probabilities.values())
                self.confidence_history.append(confidence)
            
            # Update topological charge (simplified)
            self.topological_charge = np.sum(self.noisy_pattern) / (self.pattern_size ** 2)
    
    def set_noise_level(self, noise_level):
        """Set noise level and update"""
        self.current_noise_level = max(0, min(MAX_NOISE_LEVEL, noise_level))
        self.update_noise()
    
    def get_braiding_positions(self, time_step):
        """Get current positions of braiding strands"""
        # Simple braiding animation
        positions = []
        for i in range(BRAIDING_STRANDS):
            x = i * 0.8 - 2.0
            y = 0.5 * np.sin(0.1 * time_step + i * 0.5)
            positions.append((x, y))
        return positions
    
    def update_animation(self):
        """Update animation counters"""
        self.braiding_time += 1
        self.charge_flow_time += 1


class TQNNVisualization:
    """Interactive TQNN visualization interface"""
    
    def __init__(self):
        self.simulator = TQNNSimulator()
        self.setup_figure()
        self.setup_controls()
        
        # Animation state
        self.auto_classify = False
        self.show_braiding = True
        self.show_charge_flow = True
        self.show_spin_network = True
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
    def setup_figure(self):
        """Setup the main figure and subplots"""
        self.fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
        self.fig.suptitle("Interactive Topological Quantum Neural Network (TQNN)", 
                         fontsize=16, fontweight='bold', color=DARK_TEXT)
        
        # Create carefully spaced grid layout
        gs = self.fig.add_gridspec(3, 4, height_ratios=[1.5, 1.2, 0.6], 
                                  width_ratios=[1, 1, 1, 1.2],
                                  hspace=0.35, wspace=0.3, bottom=0.12, top=0.92)
        
        # Top row - Main visualizations
        self.ax_pattern = self.fig.add_subplot(gs[0, 0])
        self.ax_pattern.set_title("Input Pattern + Noise", fontsize=12, fontweight='bold', color=DARK_TEXT)
        self.ax_pattern.set_facecolor(DARK_AXES)
        
        self.ax_braiding = self.fig.add_subplot(gs[0, 1])
        self.ax_braiding.set_title("Anyonic Braiding", fontsize=12, fontweight='bold', color=DARK_TEXT)
        self.ax_braiding.set_facecolor(DARK_AXES)
        
        self.ax_network = self.fig.add_subplot(gs[0, 2])
        self.ax_network.set_title("Charge Flow Network", fontsize=12, fontweight='bold', color=DARK_TEXT)
        self.ax_network.set_facecolor(DARK_AXES)
        
        self.ax_classification = self.fig.add_subplot(gs[0, 3])
        self.ax_classification.set_title("Classification Confidence", fontsize=12, fontweight='bold', color=DARK_TEXT)
        self.ax_classification.set_facecolor(DARK_AXES)
        
        # Middle row - Analysis
        self.ax_spin_network = self.fig.add_subplot(gs[1, :2])
        self.ax_spin_network.set_title("Spin Network Representation", fontsize=12, fontweight='bold', color=DARK_TEXT)
        self.ax_spin_network.set_facecolor(DARK_AXES)
        
        self.ax_robustness = self.fig.add_subplot(gs[1, 2:])
        self.ax_robustness.set_title("Topological Robustness", fontsize=12, fontweight='bold', color=DARK_TEXT)
        self.ax_robustness.set_facecolor(DARK_AXES)
        
        # Bottom row - Controls and parameters
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.set_title("TQNN Parameters & Controls", fontsize=12, fontweight='bold', color=DARK_TEXT)
        self.ax_controls.set_facecolor(DARK_BG)
        self.ax_controls.axis('off')
        
    def setup_controls(self):
        """Setup interactive controls"""
        # Control positioning
        control_y = 0.02
        control_height = 0.04
        
        # Noise level slider
        ax_noise = plt.axes([0.1, control_y, 0.15, control_height])
        self.slider_noise = Slider(ax_noise, 'Noise Level', 0.0, MAX_NOISE_LEVEL, 
                                  valinit=0.0, valfmt='%.2f')
        self.slider_noise.on_changed(self.update_noise)
        
        # Pattern selection buttons
        button_width = 0.06
        button_spacing = 0.01
        start_x = 0.28
        
        ax_pattern_sel = plt.axes([start_x, control_y, button_width * 4 + button_spacing * 3, control_height])
        self.radio_pattern = RadioButtons(ax_pattern_sel, self.simulator.labels, active=0)
        self.radio_pattern.on_clicked(self.select_pattern)
        
        # Action buttons
        button_start_x = 0.55
        
        ax_classify = plt.axes([button_start_x, control_y, button_width, control_height])
        self.btn_classify = Button(ax_classify, 'Classify')
        self.btn_classify.on_clicked(self.classify_pattern)
        
        ax_auto = plt.axes([button_start_x + button_width + button_spacing, control_y, 
                           button_width, control_height])
        self.btn_auto = Button(ax_auto, 'Auto Mode')
        self.btn_auto.on_clicked(self.toggle_auto_mode)
        
        ax_reset = plt.axes([button_start_x + 2 * (button_width + button_spacing), control_y, 
                           button_width, control_height])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_simulation)
        
        # Display options
        ax_options = plt.axes([0.78, control_y, 0.18, control_height * 2])
        self.checkbox = CheckButtons(ax_options, 
                                   ['Show Braiding', 'Show Charge Flow', 'Show Spin Network'], 
                                   [True, True, True])
        self.checkbox.on_clicked(self.toggle_display_options)
        
    def on_mouse_click(self, event):
        """Handle mouse clicks on pattern"""
        if event.inaxes == self.ax_pattern and self.simulator.current_pattern is not None:
            # Convert click position to pattern coordinates
            if event.xdata is not None and event.ydata is not None:
                x_idx = int(event.xdata + 0.5)
                y_idx = int(event.ydata + 0.5)
                
                if 0 <= x_idx < self.simulator.pattern_size and 0 <= y_idx < self.simulator.pattern_size:
                    # Toggle pixel
                    self.simulator.current_pattern[y_idx, x_idx] = 1 - self.simulator.current_pattern[y_idx, x_idx]
                    self.simulator.update_noise()
    
    def draw_pattern(self):
        """Draw the current pattern with noise"""
        self.ax_pattern.clear()
        self.ax_pattern.set_facecolor(DARK_AXES)
        
        if self.simulator.noisy_pattern is not None:
            # Show noisy pattern
            self.ax_pattern.imshow(self.simulator.noisy_pattern, cmap='mako', 
                                 interpolation='nearest', aspect='equal')
            
            # Add noise level indicator
            noise_text = f"Noise: {self.simulator.current_noise_level:.2f}"
            self.ax_pattern.text(0.02, 0.98, noise_text, transform=self.ax_pattern.transAxes,
                               fontsize=10, verticalalignment='top', color=DARK_TEXT,
                               bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor=DARK_EDGE, alpha=0.9))
            
            # Add pattern label
            pattern_text = f"Pattern: {self.simulator.current_pattern_label}"
            self.ax_pattern.text(0.02, 0.02, pattern_text, transform=self.ax_pattern.transAxes,
                               fontsize=10, verticalalignment='bottom', color=DARK_TEXT,
                               bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor=DARK_ACCENT, alpha=0.9))
        
        self.ax_pattern.set_xlim(-0.5, self.simulator.pattern_size - 0.5)
        self.ax_pattern.set_ylim(-0.5, self.simulator.pattern_size - 0.5)
        self.ax_pattern.set_xticks([])
        self.ax_pattern.set_yticks([])
        
    def draw_braiding(self):
        """Draw anyonic braiding visualization"""
        self.ax_braiding.clear()
        self.ax_braiding.set_facecolor(DARK_AXES)
        
        if not self.show_braiding:
            self.ax_braiding.text(0.5, 0.5, "Braiding Display Off", 
                                transform=self.ax_braiding.transAxes,
                                ha='center', va='center', fontsize=12, color=DARK_SUBTITLE)
            return
        
        # Draw braiding strands
        positions = self.simulator.get_braiding_positions(self.simulator.braiding_time)
        colors = altCmap(np.linspace(0.1, 0.9, BRAIDING_STRANDS))
        
        # Draw world-lines
        for i, (x, y) in enumerate(positions):
            # Draw strand
            self.ax_braiding.plot([x, x], [0, 2], color=colors[i], linewidth=4, 
                                solid_capstyle='round', alpha=0.8)
            
            # Draw anyon
            circle = Circle((x, y + 1), 0.1, color=colors[i], alpha=0.9)
            self.ax_braiding.add_patch(circle)
            
            # Label
            self.ax_braiding.text(x, -0.2, f'|q{i}\u27E9', ha='center', va='top', fontsize=10, color=DARK_TEXT)
        
        # Draw crossing indicators
        crossing_time = (self.simulator.braiding_time // 50) % 3
        if crossing_time == 1:
            self.ax_braiding.plot([-1, 1], [1.5, 1.5], '--', color=DARK_ACCENT, linewidth=2, alpha=0.7)
            self.ax_braiding.text(0, 1.7, "Braiding Operation", ha='center', fontsize=10, color=DARK_TEXT,
                                bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor=DARK_ACCENT, alpha=0.8))
        
        self.ax_braiding.set_xlim(-2.5, 2.5)
        self.ax_braiding.set_ylim(-0.5, 2.5)
        self.ax_braiding.set_aspect('equal')
        self.ax_braiding.axis('off')
        
    def draw_charge_network(self):
        """Draw topological charge flow network"""
        self.ax_network.clear()
        self.ax_network.set_facecolor(DARK_AXES)
        
        if not self.show_charge_flow:
            self.ax_network.text(0.5, 0.5, "Charge Flow Display Off", 
                               transform=self.ax_network.transAxes,
                               ha='center', va='center', fontsize=12, color=DARK_SUBTITLE)
            return
        
        # Create a simple network
        G = nx.DiGraph()
        
        # Add nodes with charges
        node_positions = {
            'input': (0, 1),
            'h1': (1, 1.5),
            'h2': (1, 0.5),
            'output': (2, 1)
        }
        
        charges = {
            'input': self.simulator.topological_charge,
            'h1': self.simulator.topological_charge * 0.6,
            'h2': self.simulator.topological_charge * 0.4,
            'output': self.simulator.topological_charge
        }
        
        # Add edges
        G.add_edges_from([('input', 'h1'), ('input', 'h2'), ('h1', 'output'), ('h2', 'output')])
        
        # Draw network
        node_colors = [divCmap(abs(charges[node])) for node in G.nodes()]
        nx.draw_networkx_nodes(G, node_positions, node_color=node_colors, 
                              node_size=800, alpha=0.8, ax=self.ax_network)
        
        # Draw edges with flow animation
        edge_colors = ['red' if (self.simulator.charge_flow_time // 10) % 2 == 0 else 'blue' 
                      for _ in G.edges()]
        nx.draw_networkx_edges(G, node_positions, edge_color=edge_colors, 
                              width=2, alpha=0.7, ax=self.ax_network,
                              arrowsize=20, arrowstyle='->')
        
        # Draw labels
        labels = {node: f"{charges[node]:.2f}" for node in G.nodes()}
        nx.draw_networkx_labels(G, node_positions, labels, font_size=10, 
                               font_weight='bold', ax=self.ax_network, font_color=DARK_TEXT)
        
        # Conservation law
        self.ax_network.text(0.5, 0.05, "Charge Conservation: \u2211Q_in = \u2211Q_out", 
                           transform=self.ax_network.transAxes,
                           ha='center', va='bottom', fontsize=10, color=DARK_TEXT,
                           bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor=DARK_ACCENT, alpha=0.9))
        
        self.ax_network.set_xlim(-0.5, 2.5)
        self.ax_network.set_ylim(0, 2)
        self.ax_network.axis('off')
        
    def draw_classification(self):
        """Draw classification results"""
        self.ax_classification.clear()
        self.ax_classification.set_facecolor(DARK_AXES)
        
        if self.simulator.log_probabilities:
            # Bar chart of log probabilities
            labels = list(self.simulator.log_probabilities.keys())
            values = list(self.simulator.log_probabilities.values())
            
            bars = self.ax_classification.bar(range(len(labels)), values, 
                                            color=[seqCmap(0.7) if label == self.simulator.prediction_result 
                                                  else seqCmap(0.3) for label in labels])
            
            # Highlight correct class
            correct_idx = labels.index(self.simulator.current_pattern_label)
            bars[correct_idx].set_color('green')
            bars[correct_idx].set_alpha(0.7)
            
            self.ax_classification.set_xticks(range(len(labels)))
            self.ax_classification.set_xticklabels(labels, rotation=45, color=DARK_TEXT)
            self.ax_classification.set_ylabel("Log Probability", color=DARK_TEXT)
            self.ax_classification.tick_params(colors=DARK_TEXT)
            self.ax_classification.grid(True, alpha=0.2, color=DARK_GRID)
            
            # Add prediction result
            pred_text = f"Predicted: {self.simulator.prediction_result}"
            correct_text = f"Correct: {self.simulator.current_pattern_label}"
            accuracy = "\u2713" if self.simulator.prediction_result == self.simulator.current_pattern_label else "\u2717"
            
            self.ax_classification.text(0.02, 0.98, f"{pred_text}\n{correct_text} {accuracy}", 
                                      transform=self.ax_classification.transAxes,
                                      fontsize=10, verticalalignment='top', color=DARK_TEXT,
                                      bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor=DARK_EDGE, alpha=0.9))
        
    def draw_spin_network(self):
        """Draw spin network representation"""
        self.ax_spin_network.clear()
        self.ax_spin_network.set_facecolor(DARK_AXES)
        
        if not self.show_spin_network:
            self.ax_spin_network.text(0.5, 0.5, "Spin Network Display Off", 
                                    transform=self.ax_spin_network.transAxes,
                                    ha='center', va='center', fontsize=12, color=DARK_SUBTITLE)
            return
        
        if self.simulator.noisy_pattern is not None:
            # Create hexagonal lattice representation
            hex_size = 0.3
            for i in range(self.simulator.pattern_size):
                for j in range(self.simulator.pattern_size):
                    x = j * hex_size * 1.5
                    y = i * hex_size * np.sqrt(3) + (j % 2) * hex_size * np.sqrt(3) / 2
                    
                    # Color based on pattern value
                    intensity = self.simulator.noisy_pattern[i, j]
                    color = seqCmap(intensity)
                    
                    # Draw hexagon
                    angles = np.linspace(0, 2*np.pi, 7)
                    hex_x = x + hex_size * np.cos(angles)
                    hex_y = y + hex_size * np.sin(angles)
                    
                    hex_patch = Polygon(list(zip(hex_x, hex_y)), 
                                      facecolor=color, alpha=0.8, edgecolor=DARK_EDGE)
                    self.ax_spin_network.add_patch(hex_patch)
                    
                    # Add spin label
                    if intensity > 0.5:
                        self.ax_spin_network.text(x, y, '\u2191', ha='center', va='center', 
                                                fontsize=8, fontweight='bold', color=DARK_TEXT)
        
        self.ax_spin_network.set_aspect('equal')
        self.ax_spin_network.axis('off')
        
        # Add explanation
        self.ax_spin_network.text(0.02, 0.98, "Hexagonal Spin Network (\u2191 = spin up)", 
                                transform=self.ax_spin_network.transAxes,
                                fontsize=10, verticalalignment='top', color=DARK_TEXT,
                                bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor=DARK_EDGE, alpha=0.9))
        
    def draw_robustness(self):
        """Draw topological robustness analysis"""
        self.ax_robustness.clear()
        self.ax_robustness.set_facecolor(DARK_AXES)
        
        if len(self.simulator.confidence_history) > 1:
            # Plot confidence over time
            history = list(self.simulator.confidence_history)
            self.ax_robustness.plot(history, color=seqCmap(0.7), linewidth=2)
            self.ax_robustness.fill_between(range(len(history)), history, alpha=0.3, color=seqCmap(0.7))
            
            self.ax_robustness.set_xlabel("Time Steps", color=DARK_TEXT)
            self.ax_robustness.set_ylabel("Classification Confidence", color=DARK_TEXT)
            self.ax_robustness.set_ylim(min(history) - 0.5, max(history) + 0.5)
            self.ax_robustness.tick_params(colors=DARK_TEXT)
            self.ax_robustness.grid(True, alpha=0.2, color=DARK_GRID)
            
            # Add current noise level indicator
            current_conf = history[-1] if history else 0
            self.ax_robustness.text(0.98, 0.95, f"Current Confidence: {current_conf:.3f}", 
                                  transform=self.ax_robustness.transAxes,
                                  fontsize=10, verticalalignment='top', horizontalalignment='right',
                                  color=DARK_TEXT,
                                  bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor=DARK_EDGE, alpha=0.9))
        else:
            self.ax_robustness.text(0.5, 0.5, "Confidence History\n(Run classification to see)", 
                                  transform=self.ax_robustness.transAxes,
                                  ha='center', va='center', fontsize=12, color=DARK_SUBTITLE)
        
    def draw_controls_info(self):
        """Draw control information and parameters"""
        self.ax_controls.clear()
        self.ax_controls.set_facecolor(DARK_BG)
        self.ax_controls.axis('off')
        
        # Current state information
        state_text = (
            f"Current State: Pattern='{self.simulator.current_pattern_label}', "
            f"Noise={self.simulator.current_noise_level:.2f}, "
            f"Prediction='{self.simulator.prediction_result}', "
            f"Topological Charge={self.simulator.topological_charge:.3f}"
        )
        
        self.ax_controls.text(0.5, 0.7, state_text, ha='center', va='center',
                            transform=self.ax_controls.transAxes, fontsize=11,
                            color=DARK_TEXT,
                            bbox=dict(boxstyle='round', facecolor=DARK_BG,
                                      edgecolor=DARK_EDGE, alpha=0.9))
        
        # Instructions
        instructions = (
            "Instructions: Select pattern with radio buttons • Adjust noise with slider • "
            "Click pattern pixels to edit • Use 'Classify' for single prediction • "
            "'Auto Mode' for continuous classification • Toggle display options with checkboxes"
        )
        
        self.ax_controls.text(0.5, 0.3, instructions, ha='center', va='center',
                            transform=self.ax_controls.transAxes, fontsize=10,
                            style='italic', color=DARK_SUBTITLE)
        
    def update_noise(self, val):
        """Update noise level"""
        self.simulator.set_noise_level(val)
        
    def select_pattern(self, label):
        """Select pattern by label"""
        try:
            pattern_idx = self.simulator.labels.index(label)
            self.simulator.select_pattern(pattern_idx)
        except ValueError:
            pass
        
    def classify_pattern(self, event):
        """Perform classification"""
        self.simulator.update_noise()
        
    def toggle_auto_mode(self, event):
        """Toggle automatic classification"""
        self.auto_classify = not self.auto_classify
        self.btn_auto.label.set_text('Stop Auto' if self.auto_classify else 'Auto Mode')
        
    def reset_simulation(self, event):
        """Reset simulation"""
        self.simulator = TQNNSimulator()
        self.slider_noise.reset()
        
    def toggle_display_options(self, label):
        """Toggle display options"""
        if label == 'Show Braiding':
            self.show_braiding = not self.show_braiding
        elif label == 'Show Charge Flow':
            self.show_charge_flow = not self.show_charge_flow
        elif label == 'Show Spin Network':
            self.show_spin_network = not self.show_spin_network
        
    def update(self, frame):
        """Animation update function"""
        # Auto classify if enabled
        if self.auto_classify:
            if frame % 10 == 0:  # Every 10 frames
                # Slightly vary noise for demonstration
                noise_variation = 0.01 * np.sin(frame * 0.1)
                new_noise = max(0, min(MAX_NOISE_LEVEL, 
                                     self.simulator.current_noise_level + noise_variation))
                self.simulator.set_noise_level(new_noise)
                self.slider_noise.set_val(new_noise)
        
        # Update simulation
        self.simulator.update_animation()
        
        # Redraw all components
        self.draw_pattern()
        self.draw_braiding()
        self.draw_charge_network()
        self.draw_classification()
        self.draw_spin_network()
        self.draw_robustness()
        self.draw_controls_info()
        
        return []
    
    def run(self):
        """Start the interactive simulation"""
        ani = animation.FuncAnimation(self.fig, self.update, blit=False, 
                                    interval=ANIMATION_INTERVAL, cache_frame_data=False)
        plt.show()


if __name__ == "__main__":
    print("--- Starting Interactive TQNN Visualization ---")
    print("Topological Quantum Neural Network Simulator")
    print("Based on semi-classical limit theory and topological robustness")
    print("\nFeatures:")
    print("• Interactive pattern editing and noise injection")
    print("• Real-time anyonic braiding visualization")
    print("• Topological charge conservation")
    print("• Spin network representation")
    print("• Classification confidence tracking")
    print("• Robustness analysis under topological defects")
    
    # Create and run visualization
    try:
        viz = TQNNVisualization()
        viz.run()
    except ImportError as e:
        print(f"\nError: Could not import TQNN helpers: {e}")
        print("Make sure the Code directory is accessible and contains tqnn_helpers.py")
    except Exception as e:
        print(f"\nError starting visualization: {e}")
    
    print("--- Visualization window closed ---")
