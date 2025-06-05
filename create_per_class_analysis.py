#!/usr/bin/env python3
"""Create per-class performance analysis for LSTM model."""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def create_lstm_per_class_analysis():
    """Create comprehensive per-class analysis visualization."""
    
    # Load the LSTM metrics
    with open('evaluation_results/lstm_comprehensive_metrics.json', 'r') as f:
        data = json.load(f)
    
    # Extract per-class metrics (excluding summary rows)
    per_class = data['per_class_metrics']
    
    # Define gesture class names (update these to match your actual classes)
    class_names = [
        'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',
        'Rotate Left', 'Rotate Right', 'Takeoff', 'Land', 'Stop/Hover'
    ]
    
    # Extract metrics for each class (classes 0-10)
    classes = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    support_counts = []
    
    for i in range(11):  # Classes 0-10
        if str(i) in per_class:
            classes.append(class_names[i] if i < len(class_names) else f'Class {i}')
            precision_scores.append(per_class[str(i)]['precision'])
            recall_scores.append(per_class[str(i)]['recall'])
            f1_scores.append(per_class[str(i)]['f1-score'])
            support_counts.append(int(per_class[str(i)]['support']))
    
    # Create the visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for better visualization
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    # 1. Precision, Recall, F1-Score comparison
    x = np.arange(len(classes))
    width = 0.25
    
    ax1.bar(x - width, precision_scores, width, label='Precision', alpha=0.8, color='skyblue')
    ax1.bar(x, recall_scores, width, label='Recall', alpha=0.8, color='lightcoral')
    ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Gesture Classes', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Per-Class Performance Metrics (LSTM)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
        ax1.text(i-width, p+0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i, r+0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i+width, f+0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 2. F1-Score by class (sorted)
    sorted_indices = np.argsort(f1_scores)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    bars2 = ax2.barh(range(len(sorted_classes)), sorted_f1, color=sorted_colors, alpha=0.8)
    ax2.set_yticks(range(len(sorted_classes)))
    ax2.set_yticklabels(sorted_classes)
    ax2.set_xlabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Scores Ranked by Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 1.05)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars2, sorted_f1)):
        ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=10)
    
    # 3. Sample support (class balance)
    bars3 = ax3.bar(classes, support_counts, color=colors, alpha=0.8)
    ax3.set_xlabel('Gesture Classes', fontsize=12)
    ax3.set_ylabel('Number of Test Samples', fontsize=12)
    ax3.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(classes, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars3, support_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=10)
    
    # 4. Precision vs Recall scatter plot
    scatter = ax4.scatter(recall_scores, precision_scores, c=f1_scores, 
                         s=[count*10 for count in support_counts], 
                         cmap='viridis', alpha=0.7, edgecolors='black')
    
    # Add class labels to points
    for i, (r, p, name) in enumerate(zip(recall_scores, precision_scores, classes)):
        ax4.annotate(name, (r, p), xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, alpha=0.8)
    
    ax4.set_xlabel('Recall', fontsize=12)
    ax4.set_ylabel('Precision', fontsize=12)
    ax4.set_title('Precision vs Recall (bubble size = test samples)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.7, 1.05)
    ax4.set_ylim(0.7, 1.05)
    
    # Add colorbar for F1-scores
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('F1-Score', fontsize=12)
    
    # Add diagonal line (precision = recall)
    ax4.plot([0.7, 1.05], [0.7, 1.05], 'r--', alpha=0.5, linewidth=1)
    
    # Overall layout
    plt.tight_layout()
    
    # Add overall performance text
    overall = data['overall_metrics']
    fig.text(0.02, 0.02, 
            f"Overall Performance: Accuracy: {overall['accuracy']:.3f} | "
            f"Macro F1: {overall['macro_f1']:.3f} | "
            f"Balanced Accuracy: {overall['balanced_accuracy']:.3f}",
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Save the figure
    output_path = Path('evaluation_results')
    fig.savefig(output_path / 'lstm_per_class_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ LSTM per-class analysis saved to {output_path / 'lstm_per_class_analysis.png'}")
    plt.show()

# Alternative simplified version for paper
def create_simple_per_class_chart():
    """Create a cleaner version specifically for the paper."""
    
    # Load the LSTM metrics
    with open('evaluation_results/lstm_comprehensive_metrics.json', 'r') as f:
        data = json.load(f)
    
    per_class = data['per_class_metrics']
    
    # Gesture class names
    class_names = [
        'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',
        'Rotate Left', 'Rotate Right', 'Takeoff', 'Land', 'Stop/Hover'
    ]
    
    # Extract metrics
    classes = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for i in range(11):
        if str(i) in per_class:
            classes.append(class_names[i] if i < len(class_names) else f'Class {i}')
            precision_scores.append(per_class[str(i)]['precision'])
            recall_scores.append(per_class[str(i)]['recall'])
            f1_scores.append(per_class[str(i)]['f1-score'])
    
    # Create clean figure for paper
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Use professional colors
    bars1 = ax.bar(x - width, precision_scores, width, label='Precision', 
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, recall_scores, width, label='Recall', 
                   color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', 
                   color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Gesture Classes', fontsize=14)
    ax.set_ylabel('Performance Score', fontsize=14)
    ax.set_title('LSTM Per-Class Performance Analysis', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # Add performance indicators
    avg_f1 = np.mean(f1_scores)
    ax.axhline(y=avg_f1, color='red', linestyle='--', alpha=0.7, 
               label=f'Average F1: {avg_f1:.3f}')
    ax.axhline(y=0.9, color='green', linestyle=':', alpha=0.7, 
               label='90% Threshold')
    
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save clean version
    output_path = Path('evaluation_results')
    fig.savefig(output_path / 'lstm_per_class_clean.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Clean LSTM per-class chart saved to {output_path / 'lstm_per_class_clean.png'}")
    plt.show()

if __name__ == "__main__":
    print("🎯 Creating LSTM per-class analysis visualizations...")
    
    # Create both versions
    create_lstm_per_class_analysis()
    print("\n" + "="*50)
    create_simple_per_class_chart() 