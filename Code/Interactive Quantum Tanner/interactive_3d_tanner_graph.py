"""
Interactive 3D Topological Tanner Graph Visualization for QLDPC Codes

This script provides an interactive 3D visualization of Quantum Low-Density Parity-Check
(QLDPC) codes through their Tanner graph representation. Users can:

- Explore 3D embedded Tanner graphs with topological constraints
- Manipulate geometric parameters in real-time
- Observe how topological genus affects error correction
- Visualize syndrome propagation through the graph structure
- Interact with hyperbolic geometry and surface codes
- Watch real-time error correction performance metrics

Key Features:
- Interactive 3D rotation and zoom
- Real-time topological invariant calculation
- Dynamic error injection and syndrome tracking
- Hyperbolic geometry exploration
- Surface code topology visualization
- Performance analysis under geometric constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
import seaborn as sns
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from collections import deque
import os
import sys
import time

# Add the Code directory to path to import tqnn_helpers if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Code'))

# --- Visualization Parameters ---
seqCmap = sns.color_palette("mako", as_cmap=True)
divCmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
altCmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)

# --- Dark Theme Constants ---
DARK_BG = '#1a1a1a'
DARK_AXES = '#0a0a0a'
DARK_TEXT = '#ffffff'
DARK_ACCENT = '#00ff88'
DARK_GRID = '#2d2d2d'
DARK_EDGE = '#444444'
DARK_SUBTITLE = '#aaaaaa'

# --- Simulation Parameters ---
ANIMATION_INTERVAL = 100  # Milliseconds between frames
DEFAULT_GENUS = 1
DEFAULT_HYPERBOLIC_CURVATURE = -0.5
MAX_VERTICES = 50


class TopologicalTannerGraph:
    """
    3D Topological Tanner Graph for QLDPC codes with geometric constraints
    """
    def __init__(self, n_data=24, n_check=12, genus=DEFAULT_GENUS):
        self.n_data = n_data  # Data vertices
        self.n_check = n_check  # Check vertices  
        self.genus = genus
        self.hyperbolic_curvature = DEFAULT_HYPERBOLIC_CURVATURE
        
        # Graph structure
        self.tanner_graph = nx.Graph()
        self.data_vertices = []
        self.check_vertices = []
        
        # 3D positions
        self.vertex_positions = {}
        self.surface_mesh = None
        
        # Topological properties
        self.euler_characteristic = 2 - 2 * self.genus
        self.distance_bounds = self._calculate_distance_bounds()
        
        # Error model
        self.vertex_errors = {}
        self.syndrome_state = {}
        self.error_correction_active = False
        
        # Animation state
        self.rotation_angle = 0
        self.syndrome_propagation_step = 0
        self.topology_evolution_step = 0
        
        # Performance metrics
        self.error_rate_history = deque(maxlen=100)
        self.distance_history = deque(maxlen=100)
        self.genus_history = deque(maxlen=100)
        
        # Initialize graph
        self._generate_topological_tanner_graph()
        self._embed_in_3d_surface()
        
    def _calculate_distance_bounds(self):
        """Calculate theoretical distance bounds based on topology"""
        # Quantum Singleton bound and topological constraints
        k = self.n_data - self.n_check  # Logical qubits (simplified)
        if k <= 0:
            return 1
        
        # Topological bound: d ≤ sqrt(n) for surface codes on genus g surface
        topological_bound = int(np.sqrt(self.n_data * (self.genus + 1)))
        singleton_bound = self.n_data - k + 1
        
        return min(topological_bound, singleton_bound)
    
    def _generate_topological_tanner_graph(self):
        """Generate Tanner graph with topological constraints"""
        self.tanner_graph.clear()
        
        # Create data vertices
        self.data_vertices = [f"d{i}" for i in range(self.n_data)]
        for v in self.data_vertices:
            self.tanner_graph.add_node(v, type='data', error=0, belief=0.5)
        
        # Create check vertices  
        self.check_vertices = [f"c{i}" for i in range(self.n_check)]
        for v in self.check_vertices:
            self.tanner_graph.add_node(v, type='check', syndrome=0)
        
        # Add edges with topological constraints
        self._add_topological_edges()
        
        # Initialize error and syndrome states
        self.vertex_errors = {v: 0 for v in self.data_vertices}
        self.syndrome_state = {v: 0 for v in self.check_vertices}
    
    def _add_topological_edges(self):
        """Add edges respecting topological surface constraints"""
        np.random.seed(42)  # For reproducibility
        
        # Each check connects to exactly 6 data vertices (hyperbolic tiling)
        check_degree = 6
        target_data_degree = 3  # Each data vertex in ~3 checks
        
        # Create adjacency based on surface topology
        for i, check in enumerate(self.check_vertices):
            # Select data vertices to connect based on geometric constraints
            available_data = [d for d in self.data_vertices 
                            if self.tanner_graph.degree(d) < target_data_degree + 1]
            
            if len(available_data) >= check_degree:
                connected_data = np.random.choice(available_data, check_degree, replace=False)
            else:
                connected_data = available_data + list(np.random.choice(
                    self.data_vertices, check_degree - len(available_data), replace=False))
            
            for data in connected_data:
                self.tanner_graph.add_edge(check, data)
    
    def _embed_in_3d_surface(self):
        """Embed Tanner graph in 3D surface of given genus"""
        # Generate surface mesh for given genus
        self._generate_surface_mesh()
        
        # Embed vertices on the surface
        for i, vertex in enumerate(self.data_vertices + self.check_vertices):
            # Distribute vertices on hyperbolic surface
            if i < len(self.data_vertices):
                # Data vertices on main surface
                theta = 2 * np.pi * i / len(self.data_vertices)
                phi = np.pi * (0.3 + 0.4 * np.sin(theta * self.genus))
            else:
                # Check vertices at different level
                j = i - len(self.data_vertices)
                theta = 2 * np.pi * j / len(self.check_vertices) + np.pi/4
                phi = np.pi * 0.7
            
            # Apply hyperbolic geometry transformation
            x, y, z = self._hyperbolic_embedding(theta, phi)
            self.vertex_positions[vertex] = (x, y, z)
    
    def _generate_surface_mesh(self):
        """Generate triangulated surface mesh for given genus"""
        # Create a simplified torus-like surface that can be extended to higher genus
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, 2*np.pi, 20)
        U, V = np.meshgrid(u, v)
        
        # Base torus
        R = 2 + 0.5 * self.genus  # Major radius scales with genus
        r = 1  # Minor radius
        
        # Add genus-dependent modulation
        genus_modulation = 0.3 * np.sin(self.genus * V) * np.cos(self.genus * U)
        
        X = (R + r * np.cos(V) + genus_modulation) * np.cos(U)
        Y = (R + r * np.cos(V) + genus_modulation) * np.sin(U)  
        Z = r * np.sin(V) + 0.2 * self.genus * np.sin(2*V)
        
        self.surface_mesh = (X, Y, Z)
    
    def _hyperbolic_embedding(self, theta, phi):
        """Embed point in hyperbolic geometry with given curvature"""
        # Poincaré disk model mapping to 3D
        r = np.tanh(abs(self.hyperbolic_curvature) * phi)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.sqrt(1 - r**2) if r < 1 else 0
        
        # Scale and position based on surface
        scale = 2 + 0.5 * self.genus
        return (scale * x, scale * y, scale * z)
    
    def inject_error(self, vertex, error_type=1):
        """Inject error into vertex"""
        if vertex in self.data_vertices:
            self.vertex_errors[vertex] = error_type
            self.tanner_graph.nodes[vertex]['error'] = error_type
            self._update_syndrome()
    
    def _update_syndrome(self):
        """Update syndrome based on current errors"""
        for check in self.check_vertices:
            # XOR of connected data vertex errors
            connected_data = [v for v in self.tanner_graph.neighbors(check)]
            syndrome = sum(self.vertex_errors.get(v, 0) for v in connected_data) % 2
            self.syndrome_state[check] = syndrome
            self.tanner_graph.nodes[check]['syndrome'] = syndrome
    
    def propagate_syndrome(self):
        """Perform one step of syndrome propagation"""
        if self.syndrome_propagation_step < 10:
            # Simulate belief propagation-like message passing
            for data_vertex in self.data_vertices:
                neighbors = list(self.tanner_graph.neighbors(data_vertex))
                if neighbors:
                    # Update belief based on neighbor syndromes
                    syndrome_sum = sum(self.syndrome_state.get(n, 0) for n in neighbors)
                    belief = 0.1 + 0.8 * (syndrome_sum / len(neighbors))
                    self.tanner_graph.nodes[data_vertex]['belief'] = belief
            
            self.syndrome_propagation_step += 1
    
    def evolve_topology(self):
        """Evolve the topological structure"""
        self.topology_evolution_step += 1
        
        # Periodically change genus (for demonstration)
        if self.topology_evolution_step % 200 == 0:
            new_genus = (self.genus % 3) + 1
            self.set_genus(new_genus)
    
    def set_genus(self, new_genus):
        """Change the genus of the surface"""
        self.genus = max(0, min(5, new_genus))  # Limit range
        self.euler_characteristic = 2 - 2 * self.genus
        self.distance_bounds = self._calculate_distance_bounds()
        self._embed_in_3d_surface()
        
        # History will be updated by get_performance_metrics() to stay synchronized
    
    def set_hyperbolic_curvature(self, curvature):
        """Set hyperbolic curvature parameter"""
        self.hyperbolic_curvature = max(-2.0, min(0.0, curvature))
        self._embed_in_3d_surface()
    
    def clear_errors(self):
        """Clear all errors"""
        for vertex in self.data_vertices:
            self.vertex_errors[vertex] = 0
            self.tanner_graph.nodes[vertex]['error'] = 0
            self.tanner_graph.nodes[vertex]['belief'] = 0.5
        
        for vertex in self.check_vertices:
            self.syndrome_state[vertex] = 0
            self.tanner_graph.nodes[vertex]['syndrome'] = 0
        
        self.syndrome_propagation_step = 0
    
    def get_performance_metrics(self):
        """Calculate current performance metrics"""
        # Error rate
        total_errors = sum(1 for e in self.vertex_errors.values() if e > 0)
        error_rate = total_errors / len(self.data_vertices)
        
        # Distance estimate (simplified)
        distance_estimate = self.distance_bounds
        
        # Record history - ensure all histories are synchronized
        self.error_rate_history.append(error_rate)
        self.distance_history.append(distance_estimate)
        self.genus_history.append(self.genus)  # Always update genus history too
        
        return {
            'error_rate': error_rate,
            'distance': distance_estimate,
            'genus': self.genus,
            'euler_char': self.euler_characteristic,
            'n_vertices': len(self.data_vertices) + len(self.check_vertices),
            'n_edges': self.tanner_graph.number_of_edges()
        }


class TannerGraph3DVisualization:
    """Interactive 3D Tanner Graph Visualization"""
    
    def __init__(self):
        self.tanner_graph = TopologicalTannerGraph()
        self.setup_figure()
        self.setup_controls()
        
        # Visualization state
        self.show_surface = True
        self.show_edges = True
        self.show_labels = False
        self.auto_rotate = False  # Start without auto-rotation
        self.auto_syndrome = False
        
        # 3D view parameters
        self.elevation = 20
        self.azimuth = 45
        self.zoom_level = 1.0
        
        # Mouse interaction
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
    def setup_figure(self):
        """Setup the main figure and 3D plot"""
        self.fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
        self.fig.suptitle("Interactive 3D Topological Tanner Graph for QLDPC Codes", 
                         fontsize=16, fontweight='bold', color=DARK_TEXT)
        
        # Create layout with 3D plot and control panels
        gs = self.fig.add_gridspec(3, 3, height_ratios=[2.5, 1, 0.8], 
                                  width_ratios=[2, 1, 1],
                                  hspace=0.4, wspace=0.3, bottom=0.15, top=0.92)
        
        # Main 3D visualization
        self.ax_3d = self.fig.add_subplot(gs[0, :2], projection='3d')
        self.ax_3d.set_facecolor(DARK_AXES)
        self.ax_3d.set_title("3D Topological Tanner Graph", fontsize=14, fontweight='bold',
                             color=DARK_TEXT)
        
        # Performance metrics
        self.ax_metrics = self.fig.add_subplot(gs[0, 2])
        self.ax_metrics.set_facecolor(DARK_AXES)
        self.ax_metrics.set_title("Performance Metrics", fontsize=12, fontweight='bold',
                                  color=DARK_TEXT)
        
        # Topological properties
        self.ax_topology = self.fig.add_subplot(gs[1, :2])
        self.ax_topology.set_facecolor(DARK_AXES)
        self.ax_topology.set_title("Topological Properties Evolution", fontsize=12,
                                   fontweight='bold', color=DARK_TEXT)
        
        # Network analysis
        self.ax_network = self.fig.add_subplot(gs[1, 2])
        self.ax_network.set_facecolor(DARK_AXES)
        self.ax_network.set_title("Network Analysis", fontsize=12, fontweight='bold',
                                  color=DARK_TEXT)
        
        # Controls
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.set_facecolor(DARK_BG)
        self.ax_controls.set_title("Interactive Controls", fontsize=12, fontweight='bold',
                                   color=DARK_TEXT)
        self.ax_controls.axis('off')
        
    def setup_controls(self):
        """Setup interactive controls"""
        # Control positioning
        control_y = 0.02
        control_height = 0.03
        
        # Genus slider
        ax_genus = plt.axes([0.1, control_y + 0.04, 0.15, control_height])
        self.slider_genus = Slider(ax_genus, 'Genus', 0, 5, 
                                  valinit=DEFAULT_GENUS, valfmt='%d')
        self.slider_genus.on_changed(self.update_genus)
        
        # Curvature slider
        ax_curvature = plt.axes([0.1, control_y, 0.15, control_height])
        self.slider_curvature = Slider(ax_curvature, 'Curvature', -2.0, 0.0, 
                                      valinit=DEFAULT_HYPERBOLIC_CURVATURE, valfmt='%.2f')
        self.slider_curvature.on_changed(self.update_curvature)
        
        # Action buttons
        button_width = 0.08
        button_spacing = 0.01
        start_x = 0.3
        
        ax_inject = plt.axes([start_x, control_y + 0.04, button_width, control_height])
        self.btn_inject = Button(ax_inject, 'Inject Error')
        self.btn_inject.on_clicked(self.inject_random_error)
        
        ax_propagate = plt.axes([start_x + button_width + button_spacing, control_y + 0.04, 
                               button_width, control_height])
        self.btn_propagate = Button(ax_propagate, 'Propagate')
        self.btn_propagate.on_clicked(self.propagate_syndrome)
        
        ax_clear = plt.axes([start_x + 2*(button_width + button_spacing), control_y + 0.04, 
                           button_width, control_height])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self.clear_errors)
        
        ax_auto_syndrome = plt.axes([start_x, control_y, button_width, control_height])
        self.btn_auto_syndrome = Button(ax_auto_syndrome, 'Auto Syndrome')
        self.btn_auto_syndrome.on_clicked(self.toggle_auto_syndrome)
        
        ax_auto_rotate = plt.axes([start_x + button_width + button_spacing, control_y, 
                                 button_width, control_height])
        self.btn_auto_rotate = Button(ax_auto_rotate, 'Auto Rotate')
        self.btn_auto_rotate.on_clicked(self.toggle_auto_rotate)
        
        # Display options
        ax_options = plt.axes([0.65, control_y, 0.25, control_height * 2])
        self.checkbox = CheckButtons(ax_options, 
                                   ['Show Surface', 'Show Edges', 'Show Labels'], 
                                   [True, True, False])
        self.checkbox.on_clicked(self.toggle_display_options)
        
    def on_mouse_click(self, event):
        """Handle mouse clicks for vertex selection"""
        if event.inaxes == self.ax_3d:
            # Find closest vertex to click (simplified)
            if hasattr(event, 'xdata') and event.xdata is not None:
                # Inject error in random vertex for demonstration
                vertices = list(self.tanner_graph.data_vertices)
                if vertices:
                    vertex = np.random.choice(vertices)
                    self.tanner_graph.inject_error(vertex, 1)
    
    def update(self, frame):
        """Animation update function"""
        # Automatic actions
        if self.auto_syndrome:
            self.tanner_graph.propagate_syndrome()
        
        self.tanner_graph.evolve_topology()
        
        # Redraw all components
        self.draw_3d_graph()
        self.draw_metrics()
        self.draw_topology_evolution()
        self.draw_network_analysis()
        
        # Redraw canvas
        self.fig.canvas.draw_idle()
        
        return []

    def run(self):
        """Run the visualization"""
        ani = animation.FuncAnimation(self.fig, self.update, 
                                      frames=200, interval=ANIMATION_INTERVAL, blit=True)
        plt.show()

    def draw_3d_graph(self):
        """Draw the 3D Tanner graph"""
        self.ax_3d.clear()
        
        # Draw surface mesh if enabled
        if self.show_surface and self.tanner_graph.surface_mesh:
            X, Y, Z = self.tanner_graph.surface_mesh
            self.ax_3d.plot_surface(X, Y, Z, alpha=0.1, color=altCmap(0.3))
        
        # Draw vertices
        data_positions = []
        check_positions = []
        data_colors = []
        check_colors = []
        
        for vertex in self.tanner_graph.data_vertices:
            pos = self.tanner_graph.vertex_positions[vertex]
            data_positions.append(pos)
            
            # Color based on error state
            error = self.tanner_graph.vertex_errors.get(vertex, 0)
            belief = self.tanner_graph.tanner_graph.nodes[vertex].get('belief', 0.5)
            
            if error > 0:
                color = [1.0, 0.0, 0.0, 1.0]  # Red as RGBA
            else:
                color = seqCmap(belief)  # This returns RGBA tuple
            data_colors.append(color)
        
        for vertex in self.tanner_graph.check_vertices:
            pos = self.tanner_graph.vertex_positions[vertex]
            check_positions.append(pos)
            
            # Color based on syndrome
            syndrome = self.tanner_graph.syndrome_state.get(vertex, 0)
            if syndrome:
                color = [1.0, 0.5, 0.0, 1.0]  # Orange as RGBA
            else:
                color = divCmap(0.7)  # This returns RGBA tuple
            check_colors.append(color)
        
        # Plot vertices
        if data_positions:
            data_pos = np.array(data_positions)
            self.ax_3d.scatter(data_pos[:, 0], data_pos[:, 1], data_pos[:, 2], 
                             c=data_colors, s=100, marker='o', alpha=0.8, label='Data')
        
        if check_positions:
            check_pos = np.array(check_positions)
            self.ax_3d.scatter(check_pos[:, 0], check_pos[:, 1], check_pos[:, 2], 
                             c=check_colors, s=120, marker='s', alpha=0.8, label='Check')
        
        # Draw edges if enabled
        if self.show_edges:
            for edge in self.tanner_graph.tanner_graph.edges():
                pos1 = self.tanner_graph.vertex_positions[edge[0]]
                pos2 = self.tanner_graph.vertex_positions[edge[1]]
                
                # Color based on syndrome activity
                v1_syndrome = 0
                v2_syndrome = 0
                if edge[0] in self.tanner_graph.check_vertices:
                    v1_syndrome = self.tanner_graph.syndrome_state.get(edge[0], 0)
                if edge[1] in self.tanner_graph.check_vertices:
                    v2_syndrome = self.tanner_graph.syndrome_state.get(edge[1], 0)
                
                edge_color = 'red' if (v1_syndrome or v2_syndrome) else 'gray'
                edge_alpha = 0.8 if (v1_syndrome or v2_syndrome) else 0.3
                
                self.ax_3d.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                              color=edge_color, alpha=edge_alpha, linewidth=1)
        
        # Labels if enabled
        if self.show_labels:
            for vertex, pos in self.tanner_graph.vertex_positions.items():
                self.ax_3d.text(pos[0], pos[1], pos[2], vertex, fontsize=8)
        
        # Update view
        if self.auto_rotate:
            self.tanner_graph.rotation_angle += 2
            self.ax_3d.view_init(elev=self.elevation, 
                               azim=self.azimuth + self.tanner_graph.rotation_angle)
        else:
            self.ax_3d.view_init(elev=self.elevation, azim=self.azimuth)
        
        # Styling
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y') 
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.legend(loc='upper right')
        
        # Add topology info
        info_text = (f"Genus: {self.tanner_graph.genus} | "
                    f"χ = {self.tanner_graph.euler_characteristic} | "
                    f"Distance ≤ {self.tanner_graph.distance_bounds}")
        self.ax_3d.text2D(0.02, 0.98, info_text, transform=self.ax_3d.transAxes,
                        fontsize=10, verticalalignment='top', color=DARK_TEXT,
                        bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor=DARK_EDGE, alpha=0.9))
        
    def draw_metrics(self):
        """Draw performance metrics"""
        self.ax_metrics.clear()
        
        metrics = self.tanner_graph.get_performance_metrics()
        
        # Create a metrics display
        metric_names = ['Error Rate', 'Distance Bound', 'Genus', 'Euler χ', 'Vertices', 'Edges']
        metric_values = [
            f"{metrics['error_rate']:.3f}",
            f"{metrics['distance']}",
            f"{metrics['genus']}",
            f"{metrics['euler_char']}",
            f"{metrics['n_vertices']}",
            f"{metrics['n_edges']}"
        ]
        
        # Color based on values
        colors = [
            seqCmap(metrics['error_rate']),  # Error rate
            altCmap(0.7),  # Distance
            divCmap(metrics['genus']/5),  # Genus
            'gray',  # Euler char
            'lightblue',  # Vertices
            'lightgreen'  # Edges
        ]
        
        y_positions = np.arange(len(metric_names))
        bars = self.ax_metrics.barh(y_positions, [1]*len(metric_names), 
                                  color=colors, alpha=0.6)
        
        # Add text
        for i, (name, value) in enumerate(zip(metric_names, metric_values)):
            self.ax_metrics.text(0.05, i, f"{name}: {value}", 
                               va='center', fontsize=10, fontweight='bold')
        
        self.ax_metrics.set_yticks([])
        self.ax_metrics.set_xlim(0, 1)
        self.ax_metrics.set_ylim(-0.5, len(metric_names) - 0.5)
        self.ax_metrics.axis('off')
        
    def draw_topology_evolution(self):
        """Draw topological properties over time"""
        self.ax_topology.clear()
        
        if len(self.tanner_graph.genus_history) > 1:
            # Plot genus evolution
            history_genus = list(self.tanner_graph.genus_history)
            history_distance = list(self.tanner_graph.distance_history)
            history_error = list(self.tanner_graph.error_rate_history)
            
            # Ensure all histories have the same length by taking the minimum
            min_length = min(len(history_genus), len(history_distance), len(history_error))
            
            # Trim all arrays to the same length
            history_genus = history_genus[:min_length]
            history_distance = history_distance[:min_length]
            history_error = history_error[:min_length]
            
            if min_length > 1:  # Need at least 2 points to plot
                x_range = range(min_length)
                
                # Multiple y-axes for different scales
                ax2 = self.ax_topology.twinx()
                
                line1 = self.ax_topology.plot(x_range, history_genus, 'b-', 
                                            label='Genus', linewidth=2)
                line2 = self.ax_topology.plot(x_range, history_distance, 'g--', 
                                            label='Distance Bound', linewidth=2)
                line3 = ax2.plot(x_range, history_error, 'r:', 
                               label='Error Rate', linewidth=2)
                
                self.ax_topology.set_xlabel('Time Steps')
                self.ax_topology.set_ylabel('Topological Properties', color='b')
                ax2.set_ylabel('Error Rate', color='r')
                
                # Set reasonable y-axis limits
                self.ax_topology.set_ylim(0, max(5, max(history_genus) + 1, max(history_distance) + 1))
                ax2.set_ylim(0, max(1.0, max(history_error) * 1.2) if history_error else 1.0)
                
                # Combine legends
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                self.ax_topology.legend(lines, labels, loc='upper left')
                
                self.ax_topology.grid(True, alpha=0.3)
            else:
                self.ax_topology.text(0.5, 0.5, "Accumulating data...\n(Need more data points)", 
                                    ha='center', va='center', transform=self.ax_topology.transAxes,
                                    fontsize=12)
        else:
            self.ax_topology.text(0.5, 0.5, "Topology Evolution\n(Run simulation to see)", 
                                ha='center', va='center', transform=self.ax_topology.transAxes,
                                fontsize=12)
    
    def draw_network_analysis(self):
        """Draw network analysis"""
        self.ax_network.clear()
        
        # Basic network statistics
        G = self.tanner_graph.tanner_graph
        
        if G.number_of_nodes() > 0:
            # Degree distribution
            degrees = [G.degree(v) for v in G.nodes()]
            unique_degrees, counts = np.unique(degrees, return_counts=True)
            
            bars = self.ax_network.bar(unique_degrees, counts, 
                                     color=seqCmap(0.6), alpha=0.7)
            
            self.ax_network.set_xlabel('Vertex Degree')
            self.ax_network.set_ylabel('Count')
            self.ax_network.set_title('Degree Distribution', fontsize=10)
            self.ax_network.grid(True, alpha=0.3)
            
            # Add statistics
            avg_degree = np.mean(degrees)
            max_degree = np.max(degrees)
            self.ax_network.text(0.98, 0.95, f'Avg: {avg_degree:.1f}\nMax: {max_degree}', 
                               transform=self.ax_network.transAxes,
                               fontsize=9, verticalalignment='top', horizontalalignment='right',
                               color=DARK_TEXT,
                               bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor=DARK_EDGE, alpha=0.9))
        else:
            self.ax_network.text(0.5, 0.5, "Network Analysis", 
                               ha='center', va='center', transform=self.ax_network.transAxes)
    
    def draw_controls_info(self):
        """Draw control information"""
        self.ax_controls.clear()
        self.ax_controls.set_facecolor(DARK_BG)
        self.ax_controls.axis('off')
        
        # Current state
        state_text = (f"3D Topological Tanner Graph: "
                     f"Genus = {self.tanner_graph.genus}, "
                     f"Curvature = {self.tanner_graph.hyperbolic_curvature:.2f}, "
                     f"Active Errors = {sum(1 for e in self.tanner_graph.vertex_errors.values() if e > 0)}")
        
        self.ax_controls.text(0.5, 0.7, state_text, ha='center', va='center',
                            transform=self.ax_controls.transAxes, fontsize=11,
                            color=DARK_TEXT,
                            bbox=dict(boxstyle='round', facecolor=DARK_BG,
                                      edgecolor=DARK_EDGE, alpha=0.9))
        
        # Instructions
        instructions = (
            "Instructions: Adjust genus and curvature with sliders • "
            "Click 'Inject Error' to add errors • 'Propagate' for syndrome propagation • "
            "'Auto' modes for continuous animation • Toggle display options with checkboxes • "
            "Click on 3D plot to inject errors interactively"
        )
        
        self.ax_controls.text(0.5, 0.3, instructions, ha='center', va='center',
                            transform=self.ax_controls.transAxes, fontsize=10,
                            style='italic', color=DARK_SUBTITLE)
    
    def toggle_display_options(self, label):
        """Toggle display options based on checkbox"""
        if label == 'Show Surface':
            self.show_surface = not self.show_surface
        elif label == 'Show Edges':
            self.show_edges = not self.show_edges
        elif label == 'Show Labels':
            self.show_labels = not self.show_labels
        
        # Force a redraw when display options change
        try:
            self.draw_3d_graph()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating display: {e}")
            
    def update_genus(self, val):
        """Update genus from slider"""
        try:
            self.tanner_graph.set_genus(int(self.slider_genus.val))
            # Update metrics to keep history synchronized
            self.tanner_graph.get_performance_metrics()
            # Force a redraw after genus change
            self.draw_3d_graph()
            self.draw_metrics()
            self.draw_topology_evolution()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating genus: {e}")
        
    def update_curvature(self, val):
        """Update curvature from slider"""
        try:
            self.tanner_graph.set_hyperbolic_curvature(self.slider_curvature.val)
            # Force a redraw after curvature change
            self.draw_3d_graph()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating curvature: {e}")
        
    def inject_random_error(self, event):
        """Inject a random error"""
        try:
            vertices = list(self.tanner_graph.data_vertices)
            if vertices:
                vertex = np.random.choice(vertices)
                self.tanner_graph.inject_error(vertex, 1)
                print(f"Injected error in vertex: {vertex}")
                # Update metrics to keep history synchronized
                self.tanner_graph.get_performance_metrics()
                # Force a redraw
                self.draw_3d_graph()
                self.draw_metrics()
                self.draw_topology_evolution()
                self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error injecting error: {e}")
            import traceback
            traceback.print_exc()
            
    def propagate_syndrome(self, event):
        """Propagate syndrome for one step"""
        try:
            self.tanner_graph.propagate_syndrome()
            print("Syndrome propagated")
            # Update metrics to keep history synchronized
            self.tanner_graph.get_performance_metrics()
            # Force a redraw
            self.draw_3d_graph()
            self.draw_metrics()
            self.draw_topology_evolution()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error propagating syndrome: {e}")
            import traceback
            traceback.print_exc()
        
    def clear_errors(self, event):
        """Clear all errors"""
        try:
            self.tanner_graph.clear_errors()
            print("Errors cleared")
            # Force a redraw
            self.draw_3d_graph()
            self.draw_metrics()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error clearing errors: {e}")
            import traceback
            traceback.print_exc()
        
    def toggle_auto_syndrome(self, event):
        """Toggle automatic syndrome propagation"""
        self.auto_syndrome = not self.auto_syndrome
        self.btn_auto_syndrome.label.set_text('Stop Syndrome' if self.auto_syndrome else 'Auto Syndrome')
        
    def toggle_auto_rotate(self, event):
        """Toggle automatic rotation"""
        self.auto_rotate = not self.auto_rotate
        self.btn_auto_rotate.label.set_text('Stop Rotate' if self.auto_rotate else 'Auto Rotate')

    def on_mouse_click(self, event):
        """Handle mouse clicks for vertex selection"""
        if event.inaxes == self.ax_3d:
            # Find closest vertex to click (simplified)
            if hasattr(event, 'xdata') and event.xdata is not None:
                # Inject error in random vertex for demonstration
                vertices = list(self.tanner_graph.data_vertices)
                if vertices:
                    vertex = np.random.choice(vertices)
                    self.tanner_graph.inject_error(vertex, 1)
    
    def update(self, frame):
        """Animation update function"""
        # Automatic actions
        if self.auto_syndrome:
            self.tanner_graph.propagate_syndrome()
        
        self.tanner_graph.evolve_topology()
        
        # Redraw all components
        self.draw_3d_graph()
        self.draw_metrics()
        self.draw_topology_evolution()
        self.draw_network_analysis()
        
        # Redraw canvas
        self.fig.canvas.draw_idle()
        
        return []

    def run(self):
        """Run the visualization"""
        # Initial draw
        self.draw_3d_graph()
        self.draw_metrics()
        self.draw_topology_evolution()
        self.draw_network_analysis()
        
        # Start animation without blit (which can cause issues)
        self.ani = animation.FuncAnimation(self.fig, self.update, 
                                          frames=200, interval=ANIMATION_INTERVAL, 
                                          blit=False, repeat=True)
        plt.show()


if __name__ == "__main__":
    print("--- Starting Interactive 3D Topological Tanner Graph Visualization ---")
    print("3D QLDPC Tanner Graph with Topological Constraints")
    print("Exploring hyperbolic geometry and surface codes")
    print("\nFeatures:")
    print("• Interactive 3D graph embedded on surfaces of different genus")
    print("• Real-time topological parameter adjustment")
    print("• Hyperbolic geometry with curvature control")
    print("• Error injection and syndrome propagation")
    print("• Performance metrics and network analysis")
    print("• Topological invariant tracking")
    
    try:
        viz = TannerGraph3DVisualization()
        viz.run()
    except Exception as e:
        print(f"\nError starting visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("--- Visualization window closed ---")