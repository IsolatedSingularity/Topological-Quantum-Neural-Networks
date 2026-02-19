"""
Animated Visualizations for TQNN Concepts

This script generates and saves animated figures (GIFs) illustrating dynamic
concepts of Topological Quantum Neural Networks (TQNNs).

The animations include:
1.  Dynamic topological braiding of anyonic world-lines.
2.  Animated flow of topological charge through a network.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import networkx as nx
from matplotlib.lines import Line2D

# --- Setup: Color Palettes and Output Directory ---

# Use the approved color palettes from the instructions
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

# Define the output directory for plots
plots_dir = 'Plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

# --- Animation Functions ---

def animate_braiding_pattern(save_path: str) -> None:
    """
    Creates an animation of the braiding of 6 anyonic world-lines over time.
    The animation shows the strands moving and crossing, highlighting the
    dynamic nature of topological operations.
    """
    print("Generating braiding pattern animation...")
    n_strands = 6
    crossings = [
        ((0, 1), 0.1), ((2, 3), 0.2), ((4, 5), 0.3),
        ((1, 2), 0.4), ((3, 4), 0.5),
        ((0, 1), 0.6), ((2, 3), 0.7), ((4, 5), 0.8),
        ((1, 2), 0.9)
    ]
    
    fig, ax = plt.subplots(figsize=(10, 12), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AXES)
    ax.set_title(r'Animated Topological Braiding (6 Anyons)', fontsize=18, pad=20, color=DARK_TEXT)
    ax.set_xlabel('Spatial Dimension', fontsize=14, color=DARK_SUBTITLE)
    ax.set_ylabel(r'Time-like Evolution ($\tau$)', fontsize=14, color=DARK_SUBTITLE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlim(-2, 2)

    y_coords = np.array([0] + [c[1] for c in crossings] + [1])
    strand_pos = np.arange(n_strands)
    paths = {i: [] for i in range(n_strands)}
    current_pos = np.linspace(-1.5, 1.5, n_strands)

    for i in range(len(y_coords) - 1):
        y_start, y_end = y_coords[i], y_coords[i+1]
        y_segment = np.linspace(y_start, y_end, 50)

        next_pos = current_pos.copy()
        if i < len(crossings):
            s_over, s_under = crossings[i][0]
            loc_over = np.where(strand_pos == s_over)[0][0]
            loc_under = np.where(strand_pos == s_under)[0][0]
            next_pos[loc_over], next_pos[loc_under] = next_pos[loc_under], next_pos[loc_over]
            strand_pos[loc_over], strand_pos[loc_under] = strand_pos[loc_under], strand_pos[loc_over]

        for s_idx in range(n_strands):
            logical_idx = np.where(strand_pos == s_idx)[0][0]
            x_start, x_end = current_pos[logical_idx], next_pos[logical_idx]
            x_segment = np.linspace(x_start, x_end, 50)
            paths[s_idx].append(np.column_stack([x_segment, y_segment]))
        current_pos = next_pos

    full_paths = {s_idx: np.concatenate(paths[s_idx]) for s_idx in range(n_strands)}
    colors = altCmap(np.linspace(0.1, 0.9, n_strands))
    lines = [ax.plot([], [], color=colors[i], linewidth=8, solid_capstyle='round')[0] for i in range(n_strands)]

    # Add labels that are present from the start
    initial_positions = np.linspace(-1.5, 1.5, n_strands)
    final_permutation = strand_pos # This holds the final mapping
    for i in range(n_strands):
        # Initial state label
        ax.text(initial_positions[i], -0.1, fr'$|q_{i}\rangle$', ha='center', fontsize=12, color=DARK_TEXT)
        # Final state label, placed at the end x-coordinate of the strand
        final_x = full_paths[i][-1, 0]
        # Find which initial strand (j) ends up at this position
        final_logical_index = final_permutation[np.argmin(np.abs(current_pos - final_x))]
        ax.text(final_x, 1.1, fr'$|q_{{{final_logical_index}}}\rangle$', ha='center', fontsize=12, color=DARK_TEXT)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        # Cap the frame at 100 for calculation to create a pause effect
        progress_frame = min(frame, 100)
        num_points = len(full_paths[0])
        end_point = int(progress_frame * num_points / 100)
        for i, line in enumerate(lines):
            line.set_data(full_paths[i][:end_point, 0], full_paths[i][:end_point, 1])
        return lines

    # Add 25 extra frames for the pause
    ani = animation.FuncAnimation(fig, update, frames=125, init_func=init, blit=True, interval=50)
    ani.save(save_path, writer='pillow', fps=20, savefig_kwargs={'facecolor': DARK_BG})
    plt.close()
    print(f"Saved braiding animation to {save_path}")


def animate_quantum_gate(save_path: str) -> None:
    """
    Creates an animation showing the construction of a Toffoli (CCNOT) gate
    circuit, piece by piece, starting from a blank canvas.
    """
    print("Generating quantum gate construction animation...")
    G = nx.Graph()
    
    # Define node positions, labels, and colors
    nodes_data = {
        'ctrl1_in': {'pos': (0, 2), 'label': r'$|c_1\rangle_{in}$', 'color': seqCmap(0.2), 'type': 'I/O'},
        'ctrl2_in': {'pos': (0, 0), 'label': r'$|c_2\rangle_{in}$', 'color': seqCmap(0.2), 'type': 'I/O'},
        'targ_in': {'pos': (0, -2), 'label': r'$|t\rangle_{in}$', 'color': seqCmap(0.2), 'type': 'I/O'},
        'gate': {'pos': (2, 0), 'label': 'Toffoli', 'color': seqCmap(0.5), 'type': 'Gate'},
        'ctrl1_out': {'pos': (4, 2), 'label': r'$|c_1\rangle_{out}$', 'color': seqCmap(0.2), 'type': 'I/O'},
        'ctrl2_out': {'pos': (4, 0), 'label': r'$|c_2\rangle_{out}$', 'color': seqCmap(0.2), 'type': 'I/O'},
        'targ_out': {'pos': (4, -2), 'label': r'$|t\rangle_{out}$', 'color': seqCmap(0.2), 'type': 'I/O'},
    }
    for name, data in nodes_data.items():
        G.add_node(name, **data)

    # Define edges
    edges_data = [
        ('ctrl1_in', 'gate'), ('ctrl2_in', 'gate'), ('targ_in', 'gate'),
        ('gate', 'ctrl1_out'), ('gate', 'ctrl2_out'), ('gate', 'targ_out')
    ]

    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    node_colors = [data['color'] for data in G.nodes.values()]

    # Define the animation sequence
    input_nodes = ['ctrl1_in', 'ctrl2_in', 'targ_in']
    gate_node = ['gate']
    output_nodes = ['ctrl1_out', 'ctrl2_out', 'targ_out']
    input_edges = [e for e in edges_data if e[0] in input_nodes]
    output_edges = [e for e in edges_data if e[0] in gate_node]
    
    # Stages for the animation build-up
    stages = [
        [], # Start blank
        input_nodes,
        input_nodes + gate_node,
        input_nodes + gate_node, # Edges get added here
        input_nodes + gate_node + output_nodes,
        input_nodes + gate_node + output_nodes, # Edges get added here
    ]
    edge_stages = [
        [], [], [],
        input_edges,
        input_edges,
        input_edges + output_edges
    ]

    total_frames = len(stages) * 15 # 15 frames per stage

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)

    def update(frame):
        ax.clear()
        ax.set_facecolor(DARK_AXES)
        current_stage = min(frame // 15, len(stages) - 1)
        
        nodes_to_draw = stages[current_stage]
        edges_to_draw = edge_stages[current_stage]
        
        # Get colors for the nodes that are being drawn
        colors_to_draw = [G.nodes[n]['color'] for n in nodes_to_draw]
        labels_to_draw = {n: G.nodes[n]['label'] for n in nodes_to_draw}

        # Drawing
        if nodes_to_draw:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_to_draw, 
                                   node_color=colors_to_draw, node_size=4000, 
                                   ax=ax, edgecolors=DARK_EDGE)
            nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=14, font_color=DARK_TEXT)
        
        if edges_to_draw:
            nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, width=2.0, ax=ax, edge_color='#666666')

        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Input/Output Node',
                   markerfacecolor=seqCmap(0.2), markersize=15, markeredgecolor=DARK_EDGE),
            Line2D([0], [0], marker='o', color='w', label='Gate Operation',
                   markerfacecolor=seqCmap(0.5), markersize=15, markeredgecolor=DARK_EDGE)
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=2, fontsize=12,
                  facecolor=DARK_BG, edgecolor=DARK_EDGE, labelcolor=DARK_TEXT)

        ax.set_title('Animated TQNN Gate Construction (Toffoli)', fontsize=18, pad=20, color=DARK_TEXT)
        ax.set_xlim(-1, 5)
        ax.set_ylim(-3, 3)
        plt.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50)
    ani.save(save_path, writer='pillow', fps=20, savefig_kwargs={'facecolor': DARK_BG})
    plt.close()
    print(f"Saved quantum gate construction animation to {save_path}")


def animate_complex_quantum_circuit(save_path: str) -> None:
    """
    Animates the construction of a 3-qubit Quantum Fourier Transform (QFT)
    circuit, demonstrating a more complex topological structure.
    """
    print("Generating complex quantum circuit animation (3-Qubit QFT)...")
    G = nx.Graph()

    # Define nodes for the 3-qubit QFT circuit
    nodes_data = {
        # Qubit lines
        'q0_in': {'pos': (0, 2), 'label': r'$|q_0\rangle$'}, 'q0_out': {'pos': (6, 2), 'label': r'$|q_0\rangle_{out}$'},
        'q1_in': {'pos': (0, 0), 'label': r'$|q_1\rangle$'}, 'q1_out': {'pos': (6, 0), 'label': r'$|q_1\rangle_{out}$'},
        'q2_in': {'pos': (0, -2), 'label': r'$|q_2\rangle$'}, 'q2_out': {'pos': (6, -2), 'label': r'$|q_2\rangle_{out}$'},
        # Gates
        'H0': {'pos': (1, 2), 'label': 'H', 'type': 'Hadamard'},
        'CR1': {'pos': (2, 1), 'label': r'$R_2$', 'type': 'C-Phase'}, 'dot1': {'pos': (2,2)},
        'CR2': {'pos': (3, 0), 'label': r'$R_3$', 'type': 'C-Phase'}, 'dot2': {'pos': (3,2)},
        'H1': {'pos': (2.5, 0), 'label': 'H', 'type': 'Hadamard'},
        'CR3': {'pos': (4, -1), 'label': r'$R_2$', 'type': 'C-Phase'}, 'dot3': {'pos': (4,0)},
        'H2': {'pos': (5, -2), 'label': 'H', 'type': 'Hadamard'},
    }
    
    # Add nodes to graph
    for name, data in nodes_data.items(): G.add_node(name, **data)
    
    # Define edges and build stages
    edges_data = {
        'wires': [('q0_in', 'H0'), ('H0', 'dot1'), ('dot1', 'dot2'), ('dot2', 'q0_out'),
                  ('q1_in', 'CR1'), ('CR1', 'H1'), ('H1', 'dot3'), ('dot3', 'q1_out'),
                  ('q2_in', 'CR2'), ('CR2', 'CR3'), ('CR3', 'H2'), ('H2', 'q2_out')],
        'controls': [('dot1', 'CR1'), ('dot2', 'CR2'), ('dot3', 'CR3')]
    }
    
    all_nodes = list(nodes_data.keys())
    pos = nx.get_node_attributes(G, 'pos')
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=DARK_BG)

    def update(frame):
        ax.clear()
        ax.set_facecolor(DARK_AXES)
        
        num_nodes = len(all_nodes)
        nodes_to_draw_count = int((frame / 100) * num_nodes)
        nodes_to_draw = all_nodes[:nodes_to_draw_count]
        
        # Draw base wires first
        nx.draw_networkx_edges(G, pos, edgelist=edges_data['wires'], width=1.5, edge_color='#555555')

        # --- Efficiently draw nodes by shape to avoid warnings ---
        
        # 1. Collect styles for nodes to be drawn
        styles_to_draw = {}
        for node in nodes_to_draw:
            data = G.nodes[node]
            style = {'node_size': 2000, 'node_color': 'white', 'node_shape': 'o'}
            if 'type' in data:
                if data['type'] == 'Hadamard': style.update({'node_color': altCmap(0.8), 'node_shape': 's'})
                elif data['type'] == 'C-Phase': style.update({'node_color': seqCmap(0.5), 'node_shape': 's'})
            elif 'dot' in node: style.update({'node_color': DARK_TEXT, 'node_size': 100})
            styles_to_draw[node] = style
        
        # 2. Group nodes by shape
        nodes_by_shape = {}
        for node, style in styles_to_draw.items():
            shape = style['node_shape']
            if shape not in nodes_by_shape:
                nodes_by_shape[shape] = []
            nodes_by_shape[shape].append(node)
            
        # 3. Draw each group with a single call
        for shape, nodelist in nodes_by_shape.items():
            colors = [styles_to_draw[n]['node_color'] for n in nodelist]
            sizes = [styles_to_draw[n]['node_size'] for n in nodelist]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_shape=shape,
                                   node_color=colors, node_size=sizes, edgecolors=DARK_EDGE)

        # Draw labels and control lines
        labels_to_draw = {n: G.nodes[n]['label'] for n in nodes_to_draw if 'label' in G.nodes[n]}
        nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=14, font_color=DARK_TEXT)
        
        for u, v in edges_data['controls']:
            if u in nodes_to_draw and v in nodes_to_draw:
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1.5, edge_color=DARK_TEXT)
        
        # Legend
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label='Hadamard Gate', markerfacecolor=altCmap(0.8), markersize=15),
            Line2D([0], [0], marker='s', color='w', label='Controlled-Phase Gate', markerfacecolor=seqCmap(0.5), markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Control Point', markerfacecolor=DARK_TEXT, markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=3, fontsize=12,
                  facecolor=DARK_BG, edgecolor=DARK_EDGE, labelcolor=DARK_TEXT)

        ax.set_title('Animated Construction of a 3-Qubit QFT Circuit', fontsize=18, pad=20, color=DARK_TEXT)
        ax.set_xlim(-1, 7)
        ax.set_ylim(-3, 3)
        plt.axis('off')
        
    # Add 25 frames for pause at the end
    ani = animation.FuncAnimation(fig, update, frames=125, interval=60)
    ani.save(save_path, writer='pillow', fps=25, savefig_kwargs={'facecolor': DARK_BG})
    plt.close()
    print(f"Saved complex circuit animation to {save_path}")


if __name__ == '__main__':
    # Generate all animated visualizations
    print("--- Generating Animated Visualizations ---")
    animate_braiding_pattern(os.path.join(plots_dir, 'tqnn_braiding_animation.gif'))
    animate_quantum_gate(os.path.join(plots_dir, 'tqnn_gate_animation.gif'))
    animate_complex_quantum_circuit(os.path.join(plots_dir, 'tqnn_complex_circuit_animation.gif'))
    print("\nAll animated visualizations have been generated successfully.") 