"""
TQNN Sandbox

This script provides an interactive sandbox to demonstrate the principles of
Topological Quantum Neural Networks (TQNNs), particularly their robustness
to local noise (topological defects).

It uses the TQNNPerceptron from the tqnn_helpers module to classify simple
geometric patterns and then visualizes how classification confidence degrades
as noise is introduced into the input patterns.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from tqnn_helpers import TQNNPerceptron, add_topological_defect

# --- Dark Theme Constants ---
DARK_BG = '#1a1a1a'
DARK_AXES = '#0a0a0a'
DARK_TEXT = '#ffffff'
DARK_ACCENT = '#00ff88'
DARK_GRID = '#2d2d2d'
DARK_EDGE = '#444444'
DARK_SUBTITLE = '#aaaaaa'

def create_simple_patterns(size=10):
    """
    Creates a set of simple geometric patterns for classification.
    
    Returns:
        tuple: A tuple containing a list of patterns (np.array) and a list
               of corresponding labels (string).
    """
    print(f"Generating simple patterns of size {size}x{size}...")
    
    # Pattern 1: A vertical line
    vertical_line = np.zeros((size, size))
    vertical_line[:, size // 2] = 1
    
    # Pattern 2: A horizontal line
    horizontal_line = np.zeros((size, size))
    horizontal_line[size // 2, :] = 1
    
    # Pattern 3: A simple 'X' cross
    cross = np.zeros((size, size))
    for i in range(size):
        cross[i, i] = 1
        cross[i, size - 1 - i] = 1

    patterns = [vertical_line, horizontal_line, cross]
    labels = ["Vertical Line", "Horizontal Line", "Cross"]
    
    print("Patterns generated.")
    return patterns, labels

def plot_degradation(results, target_label, plot_path):
    """
    Plots the degradation of classification confidence vs. noise level.
    
    Args:
        results (dict): A dictionary containing noise levels and corresponding
                        log probabilities.
        target_label (str): The correct label for the pattern being tested.
        plot_path (str): The path to save the generated plot.
    """
    print(f"Generating plot and saving to {plot_path}...")
    
    noise_levels = sorted(results.keys())
    
    # Use the specified color palette
    palette = sns.color_palette("mako", n_colors=len(results[0.0]))
    
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AXES)
    
    # Plot log probability for each class across noise levels
    for i, label in enumerate(results[0.0].keys()):
        log_probs = [results[noise][label] for noise in noise_levels]
        ax.plot(
            noise_levels,
            log_probs,
            marker='o',
            linestyle='--',
            label=f'Class: {label}',
            color=palette[i]
        )
        
    ax.set_title(f'TQNN Classification Confidence Degradation\n(Correct Class: "{target_label}")',
                 fontsize=16, color=DARK_TEXT)
    ax.set_xlabel("Noise Level (Fraction of Flipped Pixels)", fontsize=12, color=DARK_TEXT)
    ax.set_ylabel("Log Probability (Confidence)", fontsize=12, color=DARK_TEXT)
    ax.legend(title="Class Prototypes", fontsize=10,
              facecolor=DARK_BG, edgecolor=DARK_EDGE, labelcolor=DARK_TEXT)
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, color=DARK_GRID)
    ax.tick_params(colors=DARK_TEXT)
    
    # Ensure the Plots directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print("Plot generated successfully.")


def _normalize_log_probs(log_probs, temperature=1.0):
    """Converts log probabilities to a normalized probability distribution."""
    # Use a numerically stable softmax implementation.
    log_p_values = np.array(list(log_probs.values()))
    # Apply temperature to soften the distribution for visualization
    log_p_values /= temperature
    # Subtract the maximum log-probability for numerical stability before exponentiating.
    # This prevents underflow (all values becoming zero).
    log_p_values -= np.max(log_p_values)
    probs = np.exp(log_p_values)
    # Normalize to get a probability distribution.
    probs /= probs.sum()
    return probs

def animate_robustness_test(tqnn, test_pattern, test_label, noise_levels, animation_path):
    """
    Creates a GIF animation of the TQNN robustness test.

    The animation shows the noisy input pattern on the left and a radial plot
    of classification confidences on the right. The radial plot includes a
    "ghosting" effect to show the history of the classification.

    Args:
        tqnn (TQNNPerceptron): The trained TQNN classifier.
        test_pattern (np.array): The original, clean pattern.
        test_label (str): The correct label for the pattern.
        noise_levels (list): A list of noise levels to apply.
        animation_path (str): The path to save the GIF animation.
    """
    print(f"Generating animation and saving to {animation_path}...")
    
    fig = plt.figure(figsize=(16, 7), facecolor=DARK_BG)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor(DARK_AXES)
    ax2 = fig.add_subplot(1, 2, 2, polar=True) # Use a polar projection for the radial plot
    ax2.set_facecolor(DARK_AXES)

    palette = sns.color_palette("mako", n_colors=len(tqnn.class_labels))
    class_labels = tqnn.class_labels
    num_labels = len(class_labels)
    angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False).tolist()
    angles += angles[:1] # Close the loop

    # Pre-calculate all probabilities to get history for the trail effect
    print("Pre-calculating animation frames...")
    history = []
    for noise in noise_levels:
        pred_label, log_probs = tqnn.predict(add_topological_defect(test_pattern, noise))
        ordered_log_probs = {label: log_probs.get(label, -1000) for label in class_labels}
        # Use a higher temperature to create more visually distinct shapes
        probs = _normalize_log_probs(ordered_log_probs, temperature=20.0)
        history.append({'probs': probs, 'pred_label': pred_label})

    def update(frame):
        """Update function for each frame of the animation."""
        noise = noise_levels[frame]
        current_data = history[frame]
        pred_label = current_data['pred_label']
        
        # --- Left Subplot: Noisy Pattern ---
        ax1.clear()
        ax1.set_facecolor(DARK_AXES)
        ax1.imshow(add_topological_defect(test_pattern, noise), cmap='mako', interpolation='nearest')
        ax1.set_title(f"Input Pattern with Noise: {noise:.2f}\nPredicted: {pred_label}",
                      fontsize=14, color=DARK_TEXT)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # --- Right Subplot: Classification Confidence (Radial Plot) ---
        ax2.clear()
        
        # Draw historical trails ("ghost" polygons)
        trail_length = 5
        for i in range(1, trail_length):
            if frame - i < 0:
                break
            hist_probs = history[frame - i]['probs']
            hist_plot_values = np.concatenate((hist_probs, [hist_probs[0]]))
            alpha = 0.4 * (1 - (i / trail_length)) # Fade out older trails
            ax2.plot(angles, hist_plot_values, color=palette[1], linewidth=1.5, alpha=alpha)
            ax2.fill(angles, hist_plot_values, color=palette[1], alpha=alpha*0.5)

        # Draw the current, primary polygon
        current_probs = current_data['probs']
        plot_values = np.concatenate((current_probs, [current_probs[0]])) # Close the loop
        ax2.plot(angles, plot_values, color=palette[2], linewidth=2.5, zorder=10)
        ax2.fill(angles, plot_values, color=palette[2], alpha=0.4, zorder=10)

        # Configure the plot aesthetics
        ax2.set_facecolor(DARK_AXES)
        ax2.set_thetagrids(np.degrees(angles[:-1]), class_labels)
        ax2.set_title("TQNN Classification Confidence", fontsize=14, y=1.1, color=DARK_TEXT)
        ax2.set_rlabel_position(22.5)
        ax2.set_rlim(0, 1)
        ax2.set_yticklabels([]) # Hide radial tick labels
        ax2.tick_params(colors=DARK_TEXT)
        for label in ax2.get_xticklabels():
            label.set_color(DARK_TEXT)
            
        fig.tight_layout()

    # Create and save the animation
    anim = FuncAnimation(fig, update, frames=len(noise_levels), interval=150, repeat_delay=1000)
    # Ensure the Plots directory exists
    os.makedirs(os.path.dirname(animation_path), exist_ok=True)
    anim.save(animation_path, writer='pillow', fps=5,
              savefig_kwargs={'facecolor': DARK_BG})
    plt.close()
    
    print("Animation generated successfully.")


def main():
    """
    Main function to run the TQNN sandbox experiment.
    """
    # 1. Setup and Training
    patterns, labels = create_simple_patterns()
    tqnn = TQNNPerceptron()
    tqnn.train(patterns, labels)
    
    # 2. Select a pattern to test and introduce noise
    test_pattern_index = 0 # Use the "Vertical Line"
    test_pattern = patterns[test_pattern_index]
    test_label = labels[test_pattern_index]

    print(f"\n--- Testing Robustness for pattern: '{test_label}' ---")
    
    noise_levels = np.linspace(0, 0.5, 60) # Test noise from 0% to 50% over 60 steps
    degradation_results = {}

    # 3. Run classification across different noise levels
    for noise in noise_levels:
        noisy_pattern = add_topological_defect(test_pattern, noise)
        pred_label, log_probs = tqnn.predict(noisy_pattern)
        degradation_results[noise] = log_probs
        
        print(f"Noise: {noise:.2f} | Prediction: {pred_label} | Correct: {test_label}")

    # 4. Visualize and save the results
    plot_filename = "tqnn_robustness_sandbox.png"
    plot_path = os.path.join("Plots", plot_filename)
    plot_degradation(degradation_results, test_label, plot_path)
    
    # 5. Generate and save the animation
    animation_filename = "tqnn_robustness_animation.gif"
    animation_path = os.path.join("Plots", animation_filename)
    animate_robustness_test(tqnn, test_pattern, test_label, noise_levels, animation_path)

    print(f"\nExperiment complete. Check the output plot at '{plot_path}' and animation at '{animation_path}'.")

if __name__ == "__main__":
    main() 