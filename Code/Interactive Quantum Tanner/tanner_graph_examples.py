"""
3D Tanner Graph Examples

This script provides simpler examples and demonstrations of the 3D Tanner graph
visualization concepts. These examples can be run independently to understand
specific aspects of the topological visualization.

Examples:
1. Basic 3D graph embedding
2. Surface topology comparison
3. Hyperbolic geometry demonstration
4. Error correction simulation
5. Performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import networkx as nx
from topological_utilities import *

# --- Setup visualization parameters ---
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


def example_1_basic_3d_embedding():
    """
    Example 1: Basic 3D graph embedding on different surfaces
    """
    print("Example 1: Basic 3D Graph Embedding")
    
    # Create a simple LDPC Tanner graph
    graph = generate_ldpc_tanner_graph(12, 8, 3, 4)
    
    # Create figure with subplots for different embeddings
    fig = plt.figure(figsize=(15, 5), facecolor=DARK_BG)
    
    surface_types = ['sphere', 'torus', 'hyperbolic']
    titles = ['Sphere Embedding', 'Torus Embedding', 'Hyperbolic Embedding']
    
    for i, (surface_type, title) in enumerate(zip(surface_types, titles)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.set_facecolor(DARK_AXES)
        
        # Embed graph on surface
        positions = embed_graph_on_surface(graph, surface_type, genus=1)
        
        # Separate data and check vertices
        data_vertices = [v for v in graph.nodes() if v.startswith('d')]
        check_vertices = [v for v in graph.nodes() if v.startswith('c')]
        
        # Plot data vertices
        data_pos = np.array([positions[v] for v in data_vertices])
        ax.scatter(data_pos[:, 0], data_pos[:, 1], data_pos[:, 2], 
                  c=seqCmap(0.7), s=100, marker='o', alpha=0.8, label='Data')
        
        # Plot check vertices
        check_pos = np.array([positions[v] for v in check_vertices])
        ax.scatter(check_pos[:, 0], check_pos[:, 1], check_pos[:, 2], 
                  c=divCmap(0.7), s=120, marker='s', alpha=0.8, label='Check')
        
        # Plot edges
        for edge in graph.edges():
            pos1 = positions[edge[0]]
            pos2 = positions[edge[1]]
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                   color='#666666', alpha=0.5, linewidth=1)
        
        ax.set_title(title, fontsize=12, fontweight='bold', color=DARK_TEXT)
        ax.legend(facecolor=DARK_BG, edgecolor=DARK_EDGE, labelcolor=DARK_TEXT)
        ax.set_xlabel('X', color=DARK_TEXT)
        ax.set_ylabel('Y', color=DARK_TEXT)
        ax.set_zlabel('Z', color=DARK_TEXT)
        ax.tick_params(colors=DARK_TEXT)
    
    plt.tight_layout()
    plt.savefig('plots/example_1_basic_embeddings.png', dpi=300, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.show()
    
    print("Example 1 completed. Saved as 'plots/example_1_basic_embeddings.png'")


def example_2_genus_comparison():
    """
    Example 2: Compare surfaces of different genus
    """
    print("Example 2: Genus Comparison")
    
    fig = plt.figure(figsize=(15, 10), facecolor=DARK_BG)
    
    genus_values = [0, 1, 2, 3]
    
    for i, genus in enumerate(genus_values):
        # Surface visualization
        ax1 = fig.add_subplot(2, 4, i+1, projection='3d')
        ax1.set_facecolor(DARK_AXES)
        X, Y, Z = generate_genus_g_surface(genus)
        ax1.plot_surface(X, Y, Z, alpha=0.6, color=altCmap(0.3))
        ax1.set_title(f'Genus {genus} Surface\n\u03C7 = {calculate_euler_characteristic(genus)}', 
                     fontsize=10, fontweight='bold', color=DARK_TEXT)
        ax1.axis('off')
        
        # Distance bounds visualization
        ax2 = fig.add_subplot(2, 4, i+5)
        ax2.set_facecolor(DARK_AXES)
        n_data = 24
        n_check = 12
        bounds = calculate_distance_bound(n_data, n_check, genus)
        
        bound_names = ['Singleton', 'Topological', 'LDPC', 'Hyperbolic']
        bound_values = [bounds['singleton'], bounds['topological'], 
                       bounds['ldpc'], bounds['hyperbolic']]
        
        bars = ax2.bar(bound_names, bound_values, 
                      color=[seqCmap(0.3), altCmap(0.5), divCmap(0.7), seqCmap(0.8)],
                      alpha=0.7)
        
        ax2.set_title(f'Distance Bounds (g={genus})', fontsize=10, color=DARK_TEXT)
        ax2.set_ylabel('Distance', color=DARK_TEXT)
        ax2.tick_params(axis='x', rotation=45, colors=DARK_TEXT)
        ax2.tick_params(axis='y', colors=DARK_TEXT)
        
        # Highlight practical bound
        practical_idx = bound_values.index(bounds['practical'])
        bars[practical_idx].set_color('red')
        bars[practical_idx].set_alpha(0.9)
    
    plt.tight_layout()
    plt.savefig('plots/example_2_genus_comparison.png', dpi=300, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.show()
    
    print("Example 2 completed. Saved as 'plots/example_2_genus_comparison.png'")


def example_3_hyperbolic_geometry():
    """
    Example 3: Hyperbolic geometry effects
    """
    print("Example 3: Hyperbolic Geometry")
    
    fig = plt.figure(figsize=(12, 8), facecolor=DARK_BG)
    
    curvature_values = [0.0, -0.5, -1.0, -1.5]
    
    for i, curvature in enumerate(curvature_values):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_facecolor(DARK_AXES)
        
        # Generate points in hyperbolic space
        n_points = 20
        theta_values = np.linspace(0, 2*np.pi, n_points)
        phi_values = np.linspace(0.1, 2.0, n_points)
        
        points = []
        for theta in theta_values:
            for phi in phi_values[:5]:  # Limit for visualization
                x, y, z = hyperbolic_to_euclidean(theta, phi, curvature)
                points.append([x, y, z])
        
        points = np.array(points)
        
        # Color by distance from origin
        distances = np.linalg.norm(points, axis=1)
        colors = seqCmap(distances / np.max(distances))
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=50, alpha=0.8)
        
        ax.set_title(f'Curvature \u03BA = {curvature}', fontsize=12, fontweight='bold',
                     color=DARK_TEXT)
        ax.set_xlabel('X', color=DARK_TEXT)
        ax.set_ylabel('Y', color=DARK_TEXT)
        ax.set_zlabel('Z', color=DARK_TEXT)
        ax.tick_params(colors=DARK_TEXT)
        
        # Add unit sphere for reference
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.1, color=DARK_EDGE)
    
    plt.tight_layout()
    plt.savefig('plots/example_3_hyperbolic_geometry.png', dpi=300, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.show()
    
    print("Example 3 completed. Saved as 'plots/example_3_hyperbolic_geometry.png'")


def example_4_error_correction_simulation():
    """
    Example 4: Error correction simulation
    """
    print("Example 4: Error Correction Simulation")
    
    # Create LDPC code
    graph = generate_ldpc_tanner_graph(21, 12, 3, 6)
    
    # Simulate error correction process
    error_rates = np.linspace(0, 0.3, 10)
    correction_success = []
    
    for error_rate in error_rates:
        # Inject random errors
        data_vertices = [v for v in graph.nodes() if v.startswith('d')]
        n_errors = int(error_rate * len(data_vertices))
        error_vertices = np.random.choice(data_vertices, n_errors, replace=False)
        
        # Simple syndrome calculation
        syndrome_weight = 0
        for check in [v for v in graph.nodes() if v.startswith('c')]:
            check_syndrome = 0
            for neighbor in graph.neighbors(check):
                if neighbor in error_vertices:
                    check_syndrome ^= 1
            syndrome_weight += check_syndrome
        
        # Success if syndrome weight is reasonable
        success_rate = max(0, 1 - syndrome_weight / len([v for v in graph.nodes() if v.startswith('c')]))
        correction_success.append(success_rate)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)
    
    # Error correction performance
    ax1.set_facecolor(DARK_AXES)
    ax1.plot(error_rates, correction_success, 'o-', linewidth=2, color=seqCmap(0.7))
    ax1.fill_between(error_rates, correction_success, alpha=0.3, color=seqCmap(0.7))
    ax1.set_xlabel('Error Rate', color=DARK_TEXT)
    ax1.set_ylabel('Correction Success Rate', color=DARK_TEXT)
    ax1.set_title('Error Correction Performance', fontweight='bold', color=DARK_TEXT)
    ax1.grid(True, alpha=0.3, color=DARK_GRID)
    ax1.set_ylim(0, 1)
    ax1.tick_params(colors=DARK_TEXT)
    
    # Graph properties
    metrics = calculate_graph_topology_metrics(graph)
    metric_names = ['Density', 'Avg Clustering', 'Min Degree', 'Max Degree']
    metric_values = [metrics['density'], metrics['average_clustering'], 
                    metrics['min_degree']/10, metrics['max_degree']/10]  # Normalized
    
    ax2.set_facecolor(DARK_AXES)
    bars = ax2.bar(metric_names, metric_values, 
                  color=[seqCmap(0.3), altCmap(0.5), divCmap(0.7), seqCmap(0.8)],
                  alpha=0.7)
    ax2.set_title('Graph Properties', fontweight='bold', color=DARK_TEXT)
    ax2.set_ylabel('Normalized Value', color=DARK_TEXT)
    ax2.tick_params(axis='x', rotation=45, colors=DARK_TEXT)
    ax2.tick_params(axis='y', colors=DARK_TEXT)
    
    plt.tight_layout()
    plt.savefig('plots/example_4_error_correction.png', dpi=300, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.show()
    
    print("Example 4 completed. Saved as 'plots/example_4_error_correction.png'")


def example_5_performance_analysis():
    """
    Example 5: Performance analysis across different parameters
    """
    print("Example 5: Performance Analysis")
    
    # Parameter ranges
    genus_range = range(0, 4)
    n_data_range = [12, 18, 24, 30]
    
    # Performance metrics
    distance_bounds = []
    edge_densities = []
    clustering_coeffs = []
    
    for genus in genus_range:
        genus_distances = []
        genus_densities = []
        genus_clustering = []
        
        for n_data in n_data_range:
            n_check = n_data // 2
            
            # Distance bounds
            bounds = calculate_distance_bound(n_data, n_check, genus)
            genus_distances.append(bounds['practical'])
            
            # Graph properties
            graph = generate_ldpc_tanner_graph(n_data, n_check, 3, 6)
            metrics = calculate_graph_topology_metrics(graph)
            genus_densities.append(metrics['density'])
            genus_clustering.append(metrics['average_clustering'])
        
        distance_bounds.append(genus_distances)
        edge_densities.append(genus_densities)
        clustering_coeffs.append(genus_clustering)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor=DARK_BG)
    
    # Distance bounds vs genus
    ax1 = axes[0, 0]
    ax1.set_facecolor(DARK_AXES)
    for i, n_data in enumerate(n_data_range):
        distances = [distance_bounds[g][i] for g in range(len(genus_range))]
        ax1.plot(genus_range, distances, 'o-', label=f'n={n_data}', 
                color=seqCmap(i/len(n_data_range)))
    ax1.set_xlabel('Genus', color=DARK_TEXT)
    ax1.set_ylabel('Distance Bound', color=DARK_TEXT)
    ax1.set_title('Distance vs Genus', fontweight='bold', color=DARK_TEXT)
    ax1.legend(facecolor=DARK_BG, edgecolor=DARK_EDGE, labelcolor=DARK_TEXT)
    ax1.grid(True, alpha=0.3, color=DARK_GRID)
    ax1.tick_params(colors=DARK_TEXT)
    
    # Edge density vs parameters
    ax2 = axes[0, 1]
    ax2.set_facecolor(DARK_AXES)
    X, Y = np.meshgrid(genus_range, n_data_range)
    Z = np.array(edge_densities).T
    im = ax2.contourf(X, Y, Z, cmap=altCmap)
    ax2.set_xlabel('Genus', color=DARK_TEXT)
    ax2.set_ylabel('Number of Data Qubits', color=DARK_TEXT)
    ax2.set_title('Edge Density', fontweight='bold', color=DARK_TEXT)
    ax2.tick_params(colors=DARK_TEXT)
    plt.colorbar(im, ax=ax2)
    
    # Clustering coefficient
    ax3 = axes[1, 0]
    ax3.set_facecolor(DARK_AXES)
    Z_clustering = np.array(clustering_coeffs).T
    im2 = ax3.contourf(X, Y, Z_clustering, cmap=divCmap)
    ax3.set_xlabel('Genus', color=DARK_TEXT)
    ax3.set_ylabel('Number of Data Qubits', color=DARK_TEXT)
    ax3.set_title('Clustering Coefficient', fontweight='bold', color=DARK_TEXT)
    ax3.tick_params(colors=DARK_TEXT)
    plt.colorbar(im2, ax=ax3)
    
    # Trade-off analysis
    ax4 = axes[1, 1]
    ax4.set_facecolor(DARK_AXES)
    for i, genus in enumerate(genus_range):
        distances = distance_bounds[i]
        densities = edge_densities[i]
        ax4.scatter(densities, distances, s=100, label=f'Genus {genus}',
                   color=altCmap(i/len(genus_range)), alpha=0.7)
    ax4.set_xlabel('Edge Density', color=DARK_TEXT)
    ax4.set_ylabel('Distance Bound', color=DARK_TEXT)
    ax4.set_title('Distance vs Density Trade-off', fontweight='bold', color=DARK_TEXT)
    ax4.legend(facecolor=DARK_BG, edgecolor=DARK_EDGE, labelcolor=DARK_TEXT)
    ax4.grid(True, alpha=0.3, color=DARK_GRID)
    ax4.tick_params(colors=DARK_TEXT)
    
    plt.tight_layout()
    plt.savefig('plots/example_5_performance_analysis.png', dpi=300, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.show()
    
    print("Example 5 completed. Saved as 'plots/example_5_performance_analysis.png'")


def run_all_examples():
    """Run all examples sequentially"""
    print("Running all 3D Tanner Graph examples...")
    print("=" * 50)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    try:
        example_1_basic_3d_embedding()
        print()
        
        example_2_genus_comparison()
        print()
        
        example_3_hyperbolic_geometry()
        print()
        
        example_4_error_correction_simulation()
        print()
        
        example_5_performance_analysis()
        print()
        
        print("=" * 50)
        print("All examples completed successfully!")
        print("Check the 'plots/' directory for saved visualizations.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("3D Tanner Graph Examples")
    print("Choose an example to run:")
    print("1. Basic 3D Embedding")
    print("2. Genus Comparison")
    print("3. Hyperbolic Geometry")
    print("4. Error Correction Simulation")
    print("5. Performance Analysis")
    print("6. Run All Examples")
    
    try:
        choice = input("Enter choice (1-6): ").strip()
        
        if choice == '1':
            example_1_basic_3d_embedding()
        elif choice == '2':
            example_2_genus_comparison()
        elif choice == '3':
            example_3_hyperbolic_geometry()
        elif choice == '4':
            example_4_error_correction_simulation()
        elif choice == '5':
            example_5_performance_analysis()
        elif choice == '6':
            run_all_examples()
        else:
            print("Invalid choice. Running all examples...")
            run_all_examples()
            
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Running all examples with default settings...")
        run_all_examples()