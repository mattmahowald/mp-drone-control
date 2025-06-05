#!/usr/bin/env python3
"""Create per-class analysis with proper spacing and layout."""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def create_fixed_per_class_analysis():
    """Create per-class analysis with improved spacing."""
    
    # Load the LSTM metrics
    with open('evaluation_results/lstm_comprehensive_metrics.json', 'r') as f:
        data = json.load(f)
    
    per_class = data['per_class_metrics']
    
    # Define gesture class names (shorter names for better fit)
    class_names = [
        'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',
        'Rot Left', 'Rot Right', 'Takeoff', 'Land', 'Stop'
    ]
    
    # Extract metrics
    classes = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    support_counts = []
    
    for i in range(11):
        if str(i) in per_class:
            classes.append(class_names[i])
            precision_scores.append(per_class[str(i)]['precision'])
            recall_scores.append(per_class[str(i)]['recall'])
            f1_scores.append(per_class[str(i)]['f1-score'])
            support_counts.append(int(per_class[str(i)]['support']))
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(18, 12))  # Larger figure
    
    # Use GridSpec for better control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Main performance chart (larger, top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax1.bar(x - width, precision_scores, width, label='Precision', 
                   color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x, recall_scores, width, label='Recall', 
                   color='#ff7f0e', alpha=0.8)
    bars3 = ax1.bar(x + width, f1_scores, width, label='F1-Score', 
                   color='#2ca02c', alpha=0.8)
    
    ax1.set_xlabel('Gesture Classes', fontsize=12)
    ax1.set_ylabel('Performance Score', fontsize=12)
    ax1.set_title('Per-Class Performance Metrics (LSTM)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)
    
    # 2. F1-Score ranking (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    sorted_indices = np.argsort(f1_scores)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_classes)))
    bars2 = ax2.barh(range(len(sorted_classes)), sorted_f1, color=colors, alpha=0.8)
    
    ax2.set_yticks(range(len(sorted_classes)))
    ax2.set_yticklabels(sorted_classes, fontsize=10)
    ax2.set_xlabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Scores Ranked by Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0.8, 1.02)  # Zoom in on the relevant range
    
    # Add F1 score labels
    for i, (bar, score) in enumerate(zip(bars2, sorted_f1)):
        ax2.text(score + 0.003, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9)
    
    # 3. Class distribution (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    bars3 = ax3.bar(classes, support_counts, color=colors, alpha=0.8)
    ax3.set_xlabel('Gesture Classes', fontsize=12)
    ax3.set_ylabel('Test Samples', fontsize=12)
    ax3.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars3, support_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                str(count), ha='center', va='bottom', fontsize=10)
    
    # 4. Precision vs Recall (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Create scatter with better visibility
    scatter = ax4.scatter(recall_scores, precision_scores, 
                         c=f1_scores, s=[count*8 for count in support_counts], 
                         cmap='plasma', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add class labels with better positioning
    for i, (r, p, name) in enumerate(zip(recall_scores, precision_scores, classes)):
        ax4.annotate(name, (r, p), xytext=(3, 3), textcoords='offset points', 
                    fontsize=8, alpha=0.9)
    
    ax4.set_xlabel('Recall', fontsize=12)
    ax4.set_ylabel('Precision', fontsize=12)
    ax4.set_title('Precision vs Recall\n(bubble size ∝ test samples)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.78, 1.02)
    ax4.set_ylim(0.78, 1.02)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('F1-Score', fontsize=10)
    
    # Add diagonal line
    ax4.plot([0.78, 1.02], [0.78, 1.02], 'r--', alpha=0.5, linewidth=1)
    
    # Add overall performance summary at the bottom
    overall = data['overall_metrics']
    fig.text(0.5, 0.02, 
            f"Overall LSTM Performance: Accuracy: {overall['accuracy']:.3f} | "
            f"Macro F1: {overall['macro_f1']:.3f} | "
            f"Balanced Accuracy: {overall['balanced_accuracy']:.3f} | "
            f"Parameters: {overall['model_parameters']:,}",
            ha='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Save the fixed figure
    output_path = Path('evaluation_results')
    fig.savefig(output_path / 'lstm_per_class_analysis_fixed.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    
    print(f"✅ Fixed LSTM per-class analysis saved to {output_path / 'lstm_per_class_analysis_fixed.png'}")
    plt.show()

def create_clean_single_chart():
    """Create a single clean chart for the paper."""
    
    # Load data
    with open('evaluation_results/lstm_comprehensive_metrics.json', 'r') as f:
        data = json.load(f)
    
    per_class = data['per_class_metrics']
    
    # Shorter class names
    class_names = [
        'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',
        'Rot Left', 'Rot Right', 'Takeoff', 'Land', 'Stop'
    ]
    
    # Extract metrics
    classes = []
    f1_scores = []
    
    for i in range(11):
        if str(i) in per_class:
            classes.append(class_names[i])
            f1_scores.append(per_class[str(i)]['f1-score'])
    
    # Create single clean chart
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Use gradient colors based on performance
    colors = plt.cm.RdYlGn(np.array(f1_scores))
    
    bars = ax.bar(classes, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Gesture Classes', fontsize=14)
    ax.set_ylabel('F1-Score', fontsize=14)
    ax.set_title('LSTM Per-Class F1-Score Performance', fontsize=16, fontweight='bold')
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.8, 1.02)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add average line
    avg_f1 = np.mean(f1_scores)
    ax.axhline(y=avg_f1, color='red', linestyle='--', alpha=0.8, linewidth=2,
               label=f'Average F1: {avg_f1:.3f}')
    ax.axhline(y=0.9, color='orange', linestyle=':', alpha=0.8, linewidth=2,
               label='90% Threshold')
    
    ax.legend(fontsize=12)
    
    # Better spacing
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save clean version
    output_path = Path('evaluation_results')
    fig.savefig(output_path / 'lstm_per_class_clean_fixed.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Clean LSTM chart saved to {output_path / 'lstm_per_class_clean_fixed.png'}")
    plt.show()

if __name__ == "__main__":
    print("🔧 Creating fixed LSTM per-class analysis...")
    
    # Create both fixed versions
    create_fixed_per_class_analysis()
    print("\n" + "="*50)
    create_clean_single_chart() 