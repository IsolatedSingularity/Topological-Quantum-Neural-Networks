"""
Static Visualizations for TQNN Concepts

This script generates and saves static figures illustrating key concepts
of Topological Quantum Neural Networks (TQNNs), as per the project instructions.

The visualizations include:
1.  Topological braiding of anyonic world-lines.
2.  Conservation of topological charge in a network.
3.  A logical gate structure with symmetry-protected nodes.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

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

# Create the output directory for plots if it doesn't exist
# Assumes this script is in the 'Code/' directory
plots_dir = 'Plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

# --- Visualization Functions ---

def plot_braiding_pattern(save_path: str) -> None:
    """
    Illustrates a simple braiding pattern of three anyonic world-lines,
    representing a fundamental topological operation in a TQNN. The braiding
    is robust to local deformations, encoding information topologically.
    """
    print("Generating braiding pattern visualization...")
    n_strands = 3
    # Define the sequence of crossings: ((over, under), y_position)
    crossings = [((0, 1), 0.25), ((1, 2), 0.75)]

    fig, ax = plt.subplots(figsize=(6, 8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AXES)

    y_coords = np.array([0] + [c[1] for c in crossings] + [1])
    strand_pos = np.arange(n_strands)
    paths = {i: [] for i in range(n_strands)}
    current_pos = np.linspace(-1, 1, n_strands)

    # Generate the paths for each strand
    for i in range(len(y_coords) - 1):
        y_start, y_end = y_coords[i], y_coords[i+1]
        y_segment = np.linspace(y_start, y_end, 50)

        # Determine the strand positions after this segment
        next_pos = current_pos.copy()
        if i < len(crossings):
            s_over, s_under = crossings[i][0]
            loc_over = np.where(strand_pos == s_over)[0][0]
            loc_under = np.where(strand_pos == s_under)[0][0]
            next_pos[loc_over], next_pos[loc_under] = next_pos[loc_under], next_pos[loc_over]
            strand_pos[loc_over], strand_pos[loc_under] = strand_pos[loc_under], strand_pos[loc_over]

        # Interpolate and store path segments
        for s_idx in range(n_strands):
            logical_idx = np.where(strand_pos == s_idx)[0][0]
            x_start, x_end = current_pos[logical_idx], next_pos[logical_idx]
            x_segment = np.linspace(x_start, x_end, 50)
            paths[s_idx].append(np.column_stack([x_segment, y_segment]))

        current_pos = next_pos

    # Concatenate path segments
    for s_idx in range(n_strands):
        paths[s_idx] = np.concatenate(paths[s_idx])

    # Plot the full paths first
    colors = altCmap(np.linspace(0.1, 0.9, n_strands))
    for s_idx in range(n_strands):
        path = paths[s_idx]
        ax.plot(path[:, 0], path[:, 1], color=colors[s_idx], linewidth=8, solid_capstyle='round')

    # Draw over/under crossings
    for (s_over, s_under), y_pos in crossings:
        over_path = paths[s_over]
        cross_idx_over = np.argmin(np.abs(over_path[:, 1] - y_pos))
        segment = over_path[max(0, cross_idx_over - 6):min(len(over_path), cross_idx_over + 6)]
        
        # Draw a dark background to create the illusion of a break
        ax.plot(segment[:, 0], segment[:, 1], color=DARK_AXES, linewidth=16, solid_capstyle='round', zorder=5)
        # Redraw the over-strand segment on top
        ax.plot(segment[:, 0], segment[:, 1], color=colors[s_over], linewidth=8, solid_capstyle='round', zorder=6)

    # Final styling for a publication-quality figure
    ax.set_title(r'Topological Braiding of Anyonic World-Lines', fontsize=18, pad=20, color=DARK_TEXT)
    ax.set_xlabel('Spatial Dimension', fontsize=14, color=DARK_SUBTITLE)
    ax.set_ylabel(r'Time-like Evolution ($\tau$)', fontsize=14, color=DARK_SUBTITLE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-1.5, 1.5)
    
    # Add input/output labels
    initial_positions = np.linspace(-1, 1, n_strands)
    final_permutation = strand_pos
    for i in range(n_strands):
        ax.text(initial_positions[i], -0.08, fr'Input $|q_{i}\rangle$', ha='center', fontsize=12, color=DARK_TEXT)
        final_x = paths[i][-1, 0]
        ax.text(final_x, 1.08, fr'Output $|q_{{{final_permutation[i]}}}\rangle$', ha='center', fontsize=12, color=DARK_TEXT)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved braiding pattern visualization to {save_path}")


def plot_large_braiding_pattern(save_path: str) -> None:
    """
    Illustrates a more complex braiding pattern with 6 anyonic world-lines,
    showcasing a longer time evolution and more intricate topological operations.
    """
    print("Generating large braiding pattern visualization...")
    n_strands = 6
    # Define a more complex sequence of crossings for a longer evolution
    crossings = [
        ((0, 1), 0.1), ((2, 3), 0.2), ((4, 5), 0.3),
        ((1, 2), 0.4), ((3, 4), 0.5),
        ((0, 1), 0.6), ((2, 3), 0.7), ((4, 5), 0.8),
        ((1, 2), 0.9)
    ]

    fig, ax = plt.subplots(figsize=(10, 12), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AXES)

    y_coords = np.array([0] + [c[1] for c in crossings] + [1])
    strand_pos = np.arange(n_strands)
    paths = {i: [] for i in range(n_strands)}
    current_pos = np.linspace(-1.5, 1.5, n_strands)

    # Generate the paths for each strand
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

    for s_idx in range(n_strands):
        paths[s_idx] = np.concatenate(paths[s_idx])

    colors = altCmap(np.linspace(0.1, 0.9, n_strands))
    for s_idx in range(n_strands):
        path = paths[s_idx]
        ax.plot(path[:, 0], path[:, 1], color=colors[s_idx], linewidth=8, solid_capstyle='round')

    for (s_over, s_under), y_pos in crossings:
        over_path = paths[s_over]
        cross_idx_over = np.argmin(np.abs(over_path[:, 1] - y_pos))
        segment = over_path[max(0, cross_idx_over - 6):min(len(over_path), cross_idx_over + 6)]
        
        ax.plot(segment[:, 0], segment[:, 1], color=DARK_AXES, linewidth=16, solid_capstyle='round', zorder=5)
        ax.plot(segment[:, 0], segment[:, 1], color=colors[s_over], linewidth=8, solid_capstyle='round', zorder=6)

    ax.set_title(r'Large Topological Braiding of 6 Anyonic World-Lines', fontsize=18, pad=20, color=DARK_TEXT)
    ax.set_xlabel('Spatial Dimension', fontsize=14, color=DARK_SUBTITLE)
    ax.set_ylabel(r'Time-like Evolution ($\tau$)', fontsize=14, color=DARK_SUBTITLE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-2, 2)
    
    initial_positions = np.linspace(-1.5, 1.5, n_strands)
    final_permutation = strand_pos
    for i in range(n_strands):
        ax.text(initial_positions[i], -0.05, fr'$|q_{i}\rangle$', ha='center', fontsize=12, color=DARK_TEXT)
        final_x = paths[i][-1, 0]
        ax.text(final_x, 1.05, fr'$|q_{{{final_permutation[i]}}}\rangle$', ha='center', fontsize=12, color=DARK_TEXT)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved large braiding pattern visualization to {save_path}")


def plot_topological_charge_flow(save_path: str) -> None:
    """
    Illustrates the flow and conservation of topological charge through a
    network, a key concept in TQNNs for preserving information robustly.
    """
    print("Generating topological charge flow visualization...")
    G = nx.DiGraph()

    # Define node positions and their topological charges
    G.add_node('A', pos=(0, 3), charge=1.0)
    G.add_node('B', pos=(0, 1), charge=-1.0)
    G.add_node('C', pos=(2, 4), charge=1.0)
    G.add_node('D', pos=(2, 2), charge=0.0)
    G.add_node('E', pos=(2, 0), charge=-1.0)
    G.add_node('F', pos=(4, 3), charge=1.0)
    G.add_node('G', pos=(4, 1), charge=-1.0)
    G.add_node('H', pos=(6, 2), charge=0.0, label='Output\n(Conserved)')

    # Edges represent the flow of information/charge
    G.add_edges_from([('A', 'C'), ('A', 'D'), ('B', 'D'), ('B', 'E'),
                      ('C', 'F'), ('D', 'F'), ('D', 'G'), ('E', 'G'),
                      ('F', 'H'), ('G', 'H')])

    pos = nx.get_node_attributes(G, 'pos')
    charges = nx.get_node_attributes(G, 'charge')
    node_colors = [charges[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AXES)

    # Draw nodes, colored by charge using the diverging colormap
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=divCmap,
                                   node_size=3000, alpha=0.9, vmin=-1.5, vmax=1.5, ax=ax)
    nodes.set_edgecolor(DARK_EDGE)

    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=25,
                           edge_color='#666666', width=2.0, ax=ax, node_size=3000)

    # Add charge value labels to nodes
    labels = {n: (f"${charges[n]:+.1f}" if 'label' not in G.nodes[n] else G.nodes[n]['label']) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='white', font_weight='bold')

    # Final styling
    ax.set_title('Conservation of Topological Charge in a TQNN', fontsize=18, pad=20, color=DARK_TEXT)
    ax.text(0.5, 0.05,
            'Charge is conserved at each interaction vertex, protecting information from local errors.',
            transform=fig.transFigure, ha='center', fontsize=12, style='italic', color=DARK_SUBTITLE)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved topological charge flow visualization to {save_path}")


def plot_large_topological_charge_flow(save_path: str) -> None:
    """
    Illustrates charge conservation in a larger, more complex TQNN,
    with multiple input and output layers and intermediate processing nodes.
    """
    print("Generating large topological charge flow visualization...")
    G = nx.DiGraph()
    
    # Define a larger network with more nodes and layers
    # Layer 1: Inputs
    G.add_node('I1', pos=(0, 6), charge=1.0)
    G.add_node('I2', pos=(0, 4), charge=-0.5)
    G.add_node('I3', pos=(0, 2), charge=0.5)
    G.add_node('I4', pos=(0, 0), charge=-1.0)

    # Layer 2: Hidden
    G.add_node('H1', pos=(2, 7), charge=0.5)
    G.add_node('H2', pos=(2, 5), charge=0.5)
    G.add_node('H3', pos=(2, 3), charge=-0.5)
    G.add_node('H4', pos=(2, 1), charge=-0.5)

    # Layer 3: Hidden
    G.add_node('H5', pos=(4, 6), charge=1.0)
    G.add_node('H6', pos=(4, 4), charge=-1.0)
    G.add_node('H7', pos=(4, 2), charge=0.0)

    # Layer 4: Outputs
    G.add_node('O1', pos=(6, 5), charge=0.0, label='Out 1\n(Conserved)')
    G.add_node('O2', pos=(6, 3), charge=0.0, label='Out 2\n(Conserved)')

    # Edges defining charge flow
    G.add_edges_from([
        ('I1', 'H1'), ('I1', 'H2'), ('I2', 'H2'), ('I2', 'H3'),
        ('I3', 'H3'), ('I3', 'H4'), ('I4', 'H4'),
        ('H1', 'H5'), ('H2', 'H5'), ('H2', 'H6'),
        ('H3', 'H6'), ('H3', 'H7'), ('H4', 'H7'),
        ('H5', 'O1'), ('H6', 'O1'), ('H6', 'O2'), ('H7', 'O2')
    ])
    
    pos = nx.get_node_attributes(G, 'pos')
    charges = nx.get_node_attributes(G, 'charge')
    node_colors = [charges[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AXES)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=divCmap,
                                   node_size=3500, alpha=0.9, vmin=-1.5, vmax=1.5, ax=ax)
    nodes.set_edgecolor(DARK_EDGE)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=25,
                           edge_color='#666666', width=2.0, ax=ax, node_size=3500)
    labels = {n: (f"${charges[n]:+.1f}" if 'label' not in G.nodes[n] else G.nodes[n]['label']) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='white', font_weight='bold')

    ax.set_title('Conservation of Topological Charge in a Large TQNN', fontsize=18, pad=20, color=DARK_TEXT)
    ax.text(0.5, 0.05,
            'Information is robustly preserved as charge is conserved through multiple network layers.',
            transform=fig.transFigure, ha='center', fontsize=12, style='italic', color=DARK_SUBTITLE)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved large topological charge flow visualization to {save_path}")


def plot_logical_gate_structure(save_path: str) -> None:
    """
    Depicts the structure of a logical gate (e.g., CNOT) with
    symmetry-protected nodes, which are immune to certain errors.
    """
    print("Generating logical gate structure visualization...")
    G = nx.Graph()

    # Define nodes for a CNOT gate representation
    G.add_node('ctrl_in', pos=(0, 1), label=r'$|c\rangle_{in}$', protected=True)
    G.add_node('targ_in', pos=(0, -1), label=r'$|t\rangle_{in}$', protected=False)
    G.add_node('ctrl_out', pos=(3, 1), label=r'$|c\rangle_{out}$', protected=True)
    G.add_node('targ_out', pos=(3, -1), label=r'$|t\rangle_{out}$', protected=False)
    G.add_node('gate', pos=(1.5, 0), label='CNOT\nOperation', protected=False)

    # Define edges
    G.add_edges_from([('ctrl_in', 'ctrl_out'), ('ctrl_in', 'gate'),
                      ('targ_in', 'gate'), ('gate', 'targ_out')])

    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')

    # Define node styles based on protection status
    node_styles = {}
    for node in G.nodes():
        if G.nodes[node]['protected']:
            node_styles[node] = {'shape': 'H', 'color': altCmap(0.8), 'size': 5000}
        elif node == 'gate':
            node_styles[node] = {'shape': 's', 'color': seqCmap(0.5), 'size': 4000}
        else:
            node_styles[node] = {'shape': 'o', 'color': seqCmap(0.2), 'size': 3000}

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AXES)

    # Draw nodes with their specified shapes and colors
    for shape in set(s['shape'] for s in node_styles.values()):
        nodelist = [n for n, s in node_styles.items() if s['shape'] == shape]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_shape=shape,
                               node_color=[node_styles[n]['color'] for n in nodelist],
                               node_size=[node_styles[n]['size'] for n in nodelist],
                               ax=ax, alpha=0.9, edgecolors=DARK_EDGE)

    nx.draw_networkx_edges(G, pos, edge_color='#666666', width=2.0, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_color=DARK_TEXT)

    # Final styling
    ax.set_title('TQNN Logical Gate with Symmetry-Protected Nodes', fontsize=18, pad=20, color=DARK_TEXT)
    ax.text(0.5, 0.05,
        r'Symmetry-protected nodes (hexagons) are immune to certain classes of errors.',
        transform=fig.transFigure, ha='center', fontsize=12, style='italic', color=DARK_SUBTITLE)

    plt.axis('off')
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved logical gate structure visualization to {save_path}")


def plot_large_logical_gate_structure(save_path: str) -> None:
    """
    Depicts a larger logical circuit, such as a Toffoli gate, with multiple
    control qubits and symmetry-protected nodes, including a clear legend.
    """
    print("Generating large logical gate structure visualization...")
    G = nx.Graph()
    from matplotlib.lines import Line2D

    # Define nodes for a Toffoli (CCNOT) gate representation
    G.add_node('ctrl1_in', pos=(0, 2), label=r'$|c_1\rangle_{in}$', protected=True)
    G.add_node('ctrl2_in', pos=(0, 0), label=r'$|c_2\rangle_{in}$', protected=True)
    G.add_node('targ_in', pos=(0, -2), label=r'$|t\rangle_{in}$', protected=False)
    
    G.add_node('ctrl1_out', pos=(4, 2), label=r'$|c_1\rangle_{out}$', protected=True)
    G.add_node('ctrl2_out', pos=(4, 0), label=r'$|c_2\rangle_{out}$', protected=True)
    G.add_node('targ_out', pos=(4, -2), label=r'$|t\rangle_{out}$', protected=False)
    
    G.add_node('gate', pos=(2, -1), label='Toffoli\n(CCNOT)', protected=False)

    # Define edges
    G.add_edges_from([
        ('ctrl1_in', 'ctrl1_out'), ('ctrl1_in', 'gate'),
        ('ctrl2_in', 'ctrl2_out'), ('ctrl2_in', 'gate'),
        ('targ_in', 'gate'), ('gate', 'targ_out')
    ])

    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')

    node_styles = {}
    for node in G.nodes():
        if G.nodes[node]['protected']:
            node_styles[node] = {'shape': 'H', 'color': altCmap(0.8), 'size': 5000, 'label': 'Protected Node'}
        elif 'gate' in node:
            node_styles[node] = {'shape': 's', 'color': seqCmap(0.5), 'size': 6000, 'label': 'Gate Operation'}
        else:
            node_styles[node] = {'shape': 'o', 'color': seqCmap(0.2), 'size': 4000, 'label': 'Standard Node'}

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AXES)

    for shape in set(s['shape'] for s in node_styles.values()):
        nodelist = [n for n, s in node_styles.items() if s['shape'] == shape]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_shape=shape,
                               node_color=[node_styles[n]['color'] for n in nodelist],
                               node_size=[node_styles[n]['size'] for n in nodelist],
                               ax=ax, alpha=0.9, edgecolors=DARK_EDGE)

    nx.draw_networkx_edges(G, pos, edge_color='#666666', width=2.0, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_color=DARK_TEXT)

    # Create a custom legend that does not overlap
    legend_elements = [
        Line2D([0], [0], marker='H', color='w', label='Symmetry-Protected Node',
               markerfacecolor=altCmap(0.8), markersize=15, markeredgecolor=DARK_EDGE),
        Line2D([0], [0], marker='s', color='w', label='Gate Operation',
               markerfacecolor=seqCmap(0.5), markersize=15, markeredgecolor=DARK_EDGE),
        Line2D([0], [0], marker='o', color='w', label='Standard Node',
               markerfacecolor=seqCmap(0.2), markersize=15, markeredgecolor=DARK_EDGE)
    ]
    legend = ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3, fontsize=12,
              facecolor=DARK_BG, edgecolor=DARK_EDGE, labelcolor=DARK_TEXT)

    ax.set_title('TQNN Toffoli Gate with Symmetry-Protected Nodes', fontsize=18, pad=20, color=DARK_TEXT)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"Saved large logical gate structure visualization to {save_path}")


if __name__ == '__main__':
    # Generate all static visualizations
    print("--- Generating Original Visualizations ---")
    plot_braiding_pattern(os.path.join(plots_dir, 'tqnn_braiding_pattern.png'))
    plot_topological_charge_flow(os.path.join(plots_dir, 'tqnn_charge_flow.png'))
    plot_logical_gate_structure(os.path.join(plots_dir, 'tqnn_logical_gate.png'))
    
    print("\n--- Generating Larger/More Complex Visualizations ---")
    plot_large_braiding_pattern(os.path.join(plots_dir, 'tqnn_large_braiding_pattern.png'))
    plot_large_topological_charge_flow(os.path.join(plots_dir, 'tqnn_large_charge_flow.png'))
    plot_large_logical_gate_structure(os.path.join(plots_dir, 'tqnn_large_logical_gate.png'))

    print("\nAll static visualizations have been generated successfully.") 