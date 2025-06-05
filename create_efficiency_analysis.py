#!/usr/bin/env python3
"""Create model efficiency analysis: accuracy vs parameter count."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_efficiency_analysis():
    """Create accuracy vs parameter count scatter plot."""
    
    # Your experimental results
    models = ['MLP Small', 'MLP Large', 'GRU', 'LSTM', 'TCN', 'Transformer']
    accuracies = [88.22, 91.82, 91.06, 92.74, 88.27, 87.71]  # %
    parameters = [17163, 50699, 448011, 596235, 62731, 802955]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors for each model type
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#34495e']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    
    # Create scatter plot with different markers for each model
    for i, (model, acc, params, color, marker) in enumerate(zip(models, accuracies, parameters, colors, markers)):
        ax.scatter(params, acc, s=150, c=color, marker=marker, 
                  alpha=0.8, edgecolors='black', linewidth=1.5, 
                  label=model, zorder=5)
    
    # Add model labels with arrows
    annotations = [
        (17163, 88.22, 'MLP Small\n(Efficiency)'),
        (50699, 91.82, 'MLP Large\n(Efficiency Champion)'),
        (448011, 91.06, 'GRU'),
        (596235, 92.74, 'LSTM\n(Accuracy Leader)'),
        (62731, 88.27, 'TCN'),
        (802955, 87.71, 'Transformer\n(Poor Efficiency)')
    ]
    
    for params, acc, label in annotations:
        # Offset text to avoid overlap
        if 'Small' in label:
            xytext = (15, 15)
        elif 'Large' in label:
            xytext = (15, -25)
        elif 'GRU' in label:
            xytext = (-15, 15)
        elif 'LSTM' in label:
            xytext = (15, 15)
        elif 'TCN' in label:
            xytext = (15, 15)
        else:  # Transformer
            xytext = (-15, -25)
            
        ax.annotate(label, (params, acc), 
                   xytext=xytext, textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
    
    # Formatting
    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Efficiency Analysis: Accuracy vs Parameter Count', 
                fontsize=16, fontweight='bold')
    
    # Use log scale for x-axis to better show the range
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    
    # Set axis limits
    ax.set_xlim(10000, 1000000)
    ax.set_ylim(86, 94)
    
    # Add efficiency zones
    ax.axhspan(91, 94, alpha=0.1, color='green', label='High Accuracy Zone')
    ax.axvspan(10000, 100000, alpha=0.1, color='blue', label='High Efficiency Zone')
    
    # Add efficiency quadrant lines
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=100000, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add text annotations for quadrants
    ax.text(0.02, 0.98, 'High Efficiency\nLow Accuracy', transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    ax.text(0.98, 0.98, 'Low Efficiency\nHigh Accuracy', transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', horizontalalignment='right', alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))
    
    ax.text(0.02, 0.02, 'Low Efficiency\nLow Accuracy', transform=ax.transAxes, 
           fontsize=10, verticalalignment='bottom', alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    ax.text(0.98, 0.02, 'High Efficiency\nHigh Accuracy\n(Sweet Spot)', transform=ax.transAxes, 
           fontsize=10, verticalalignment='bottom', horizontalalignment='right', alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = Path('figures')
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / 'efficiency_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Efficiency analysis saved to {output_path / 'efficiency_analysis.png'}")
    plt.show()

def create_simple_efficiency_plot():
    """Create a cleaner version for the paper."""
    
    # Data
    models = ['MLP Small', 'MLP Large', 'GRU', 'LSTM', 'TCN', 'Transformer']
    accuracies = [88.22, 91.82, 91.06, 92.74, 88.27, 87.71]
    parameters = [17163, 50699, 448011, 596235, 62731, 802955]
    
    # Create clean figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Model colors and sizes
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#34495e']
    
    # Create scatter plot
    scatter = ax.scatter(parameters, accuracies, s=120, c=colors, 
                        alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add model labels
    for i, (model, acc, params) in enumerate(zip(models, accuracies, parameters)):
        ax.annotate(model, (params, acc), xytext=(5, 5), 
                   textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Number of Parameters', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Model Efficiency Analysis: Accuracy vs Parameter Count', 
                fontsize=15, fontweight='bold')
    
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Highlight key models
    # MLP Large - efficiency champion
    ax.annotate('Efficiency Champion', xy=(50699, 91.82), 
               xytext=(30000, 93), fontsize=12, fontweight='bold', color='green',
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # LSTM - accuracy leader  
    ax.annotate('Accuracy Leader', xy=(596235, 92.74), 
               xytext=(300000, 94), fontsize=12, fontweight='bold', color='purple',
               arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    
    # Transformer - poor efficiency
    ax.annotate('Poor Parameter\nEfficiency', xy=(802955, 87.71), 
               xytext=(600000, 86.5), fontsize=12, fontweight='bold', color='red',
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    plt.tight_layout()
    
    # Save clean version
    output_path = Path('figures')
    fig.savefig(output_path / 'efficiency_analysis_clean.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Clean efficiency analysis saved to {output_path / 'efficiency_analysis_clean.png'}")
    plt.show()

if __name__ == "__main__":
    print("📊 Creating model efficiency analysis...")
    
    # Create both versions
    create_efficiency_analysis()
    print("\n" + "="*50)
    create_simple_efficiency_plot() 