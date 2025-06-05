#!/usr/bin/env python3
"""Fix spacing issues in the efficiency analysis plot."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_fixed_efficiency_analysis():
    """Create efficiency analysis with proper spacing."""
    
    # Your experimental results
    models = ['MLP Small', 'MLP Large', 'GRU', 'LSTM', 'TCN', 'Transformer']
    accuracies = [88.22, 91.82, 91.06, 92.74, 88.27, 87.71]
    parameters = [17163, 50699, 448011, 596235, 62731, 802955]
    
    # Create figure with more space
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))  # Wider figure
    
    # Colors and markers for each model type
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#34495e']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    
    # Create scatter plot
    for i, (model, acc, params, color, marker) in enumerate(zip(models, accuracies, parameters, colors, markers)):
        ax.scatter(params, acc, s=150, c=color, marker=marker, 
                  alpha=0.8, edgecolors='black', linewidth=1.5, 
                  label=model, zorder=5)
    
    # Fixed annotations with better positioning
    annotations = [
        (17163, 88.22, 'MLP Small\n(Efficiency)', (20, 15)),
        (50699, 91.82, 'MLP Large\n(Efficiency Champion)', (20, -30)),
        (448011, 91.06, 'GRU', (-60, 15)),
        (596235, 92.74, 'LSTM\n(Accuracy Leader)', (20, 15)),
        (62731, 88.27, 'TCN', (20, 15)),
        (802955, 87.71, 'Transformer\n(Poor Efficiency)', (-120, -25))  # Move left and down
    ]
    
    for params, acc, label, offset in annotations:
        ax.annotate(label, (params, acc), 
                   xytext=offset, textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
    
    # Formatting
    ax.set_xlabel('Number of Parameters', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Efficiency Analysis: Accuracy vs Parameter Count', 
                fontsize=16, fontweight='bold')
    
    # Use log scale for x-axis with extended range
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='center right', bbox_to_anchor=(0.98, 0.3))  # Move legend
    
    # Extended axis limits for better spacing
    ax.set_xlim(8000, 1200000)  # More space on both sides
    ax.set_ylim(85.5, 94.5)      # More vertical space
    
    # Add efficiency zones with transparency
    ax.axhspan(91, 94.5, alpha=0.08, color='green')
    ax.axvspan(8000, 80000, alpha=0.08, color='blue')
    
    # Add quadrant lines
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(x=100000, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    # Add quadrant labels with better positioning
    ax.text(0.05, 0.95, 'High Efficiency\nLow Accuracy', transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.6))
    
    ax.text(0.95, 0.95, 'Low Efficiency\nHigh Accuracy', transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', horizontalalignment='right', alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.6))
    
    ax.text(0.05, 0.05, 'Low Efficiency\nLow Accuracy', transform=ax.transAxes, 
           fontsize=10, verticalalignment='bottom', alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.6))
    
    ax.text(0.6, 0.05, 'High Efficiency\nHigh Accuracy\n(Sweet Spot)', transform=ax.transAxes, 
           fontsize=10, verticalalignment='bottom', alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.6))
    
    # Adjust layout with extra margins
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.12)
    
    # Save the figure
    output_path = Path('figures')
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / 'efficiency_analysis_fixed.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    
    print(f"✅ Fixed efficiency analysis saved to {output_path / 'efficiency_analysis_fixed.png'}")
    plt.show()

def create_clean_paper_version():
    """Create the cleanest version for the paper."""
    
    # Data
    models = ['MLP Small', 'MLP Large', 'GRU', 'LSTM', 'TCN', 'Transformer']
    accuracies = [88.22, 91.82, 91.06, 92.74, 88.27, 87.71]
    parameters = [17163, 50699, 448011, 596235, 62731, 802955]
    
    # Create clean figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Professional colors
    colors = ['#2E86C1', '#28B463', '#E74C3C', '#8E44AD', '#F39C12', '#566573']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    
    # Create scatter plot with larger markers
    for i, (model, acc, params, color, marker) in enumerate(zip(models, accuracies, parameters, colors, markers)):
        ax.scatter(params, acc, s=120, c=color, marker=marker, 
                  alpha=0.9, edgecolors='black', linewidth=1.2, 
                  label=model, zorder=5)
    
    # Clean annotations without arrows
    label_positions = [
        (17163, 88.22, 'MLP Small', 'bottom'),
        (50699, 91.82, 'MLP Large', 'top'),
        (448011, 91.06, 'GRU', 'bottom'),
        (596235, 92.74, 'LSTM', 'top'),
        (62731, 88.27, 'TCN', 'top'),
        (802955, 87.71, 'Transformer', 'bottom')
    ]
    
    for params, acc, label, position in label_positions:
        if position == 'top':
            xytext = (0, 8)
            va = 'bottom'
        else:
            xytext = (0, -8)
            va = 'top'
            
        ax.annotate(label, (params, acc), 
                   xytext=xytext, textcoords='offset points',
                   fontsize=11, fontweight='bold', ha='center', va=va,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor='none'))
    
    # Professional formatting
    ax.set_xlabel('Number of Parameters', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Efficiency Analysis: Accuracy vs Parameter Count', 
                fontsize=15, fontweight='bold', pad=20)
    
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Clean legend
    ax.legend(fontsize=10, loc='center right', framealpha=0.9)
    
    # Proper axis limits
    ax.set_xlim(10000, 1000000)
    ax.set_ylim(87, 93.5)
    
    # Add key insights as text
    ax.text(0.02, 0.98, 'Key Insights:\n• MLP Large: Best efficiency\n• LSTM: Highest accuracy\n• Transformer: Poor efficiency', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    
    # Save clean version
    output_path = Path('figures')
    fig.savefig(output_path / 'efficiency_analysis_clean_final.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Clean final efficiency analysis saved to {output_path / 'efficiency_analysis_clean_final.png'}")
    plt.show()

if __name__ == "__main__":
    print("🔧 Creating fixed efficiency analysis...")
    
    # Create both fixed versions
    create_fixed_efficiency_analysis()
    print("\n" + "="*50)
    create_clean_paper_version() 