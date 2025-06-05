#!/usr/bin/env python3
"""Fix the final spacing issue with the bottom summary text."""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def create_final_fixed_analysis():
    """Create per-class analysis with perfect spacing."""
    
    # Load the LSTM metrics
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
    
    # Create figure with extra bottom space
    fig = plt.figure(figsize=(18, 13))  # Increased height
    
    # Use GridSpec with more bottom margin
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25, 
                  top=0.93, bottom=0.15)  # More bottom space
    
    # 1. Main performance chart
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
    
    # 2. F1-Score ranking
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
    ax2.set_xlim(0.8, 1.02)
    
    # Add F1 score labels
    for i, (bar, score) in enumerate(zip(bars2, sorted_f1)):
        ax2.text(score + 0.003, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9)
    
    # 3. Class distribution
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
    
    # 4. Precision vs Recall
    ax4 = fig.add_subplot(gs[1, 1])
    
    scatter = ax4.scatter(recall_scores, precision_scores, 
                         c=f1_scores, s=[count*8 for count in support_counts], 
                         cmap='plasma', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add class labels
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
    
    # Add overall performance summary with proper spacing
    overall = data['overall_metrics']
    fig.text(0.5, 0.05,  # Moved higher up
            f"Overall LSTM Performance: Accuracy: {overall['accuracy']:.3f} | "
            f"Macro F1: {overall['macro_f1']:.3f} | "
            f"Balanced Accuracy: {overall['balanced_accuracy']:.3f} | "
            f"Parameters: {overall['model_parameters']:,}",
            ha='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Save the final version
    output_path = Path('evaluation_results')
    fig.savefig(output_path / 'lstm_per_class_analysis_final.png', 
                dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    
    print(f"✅ Final LSTM per-class analysis saved to {output_path / 'lstm_per_class_analysis_final.png'}")
    plt.show()

def create_paper_ready_version():
    """Create a clean, paper-ready version."""
    
    # Load data
    with open('evaluation_results/lstm_comprehensive_metrics.json', 'r') as f:
        data = json.load(f)
    
    per_class = data['per_class_metrics']
    
    # Class names
    class_names = [
        'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',
        'Rot Left', 'Rot Right', 'Takeoff', 'Land', 'Stop'
    ]
    
    # Extract metrics
    classes = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for i in range(11):
        if str(i) in per_class:
            classes.append(class_names[i])
            precision_scores.append(per_class[str(i)]['precision'])
            recall_scores.append(per_class[str(i)]['recall'])
            f1_scores.append(per_class[str(i)]['f1-score'])
    
    # Create clean two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left panel: Performance metrics
    x = np.arange(len(classes))
    width = 0.25
    
    ax1.bar(x - width, precision_scores, width, label='Precision', 
           color='#1f77b4', alpha=0.8)
    ax1.bar(x, recall_scores, width, label='Recall', 
           color='#ff7f0e', alpha=0.8)
    ax1.bar(x + width, f1_scores, width, label='F1-Score', 
           color='#2ca02c', alpha=0.8)
    
    ax1.set_xlabel('Gesture Classes', fontsize=12)
    ax1.set_ylabel('Performance Score', fontsize=12)
    ax1.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)
    
    # Right panel: F1 ranking
    sorted_indices = np.argsort(f1_scores)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    
    colors = plt.cm.viridis(np.array(sorted_f1))
    bars = ax2.barh(range(len(sorted_classes)), sorted_f1, color=colors, alpha=0.8)
    
    ax2.set_yticks(range(len(sorted_classes)))
    ax2.set_yticklabels(sorted_classes, fontsize=11)
    ax2.set_xlabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Scores Ranked by Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0.8, 1.02)
    
    # Add F1 labels
    for bar, score in zip(bars, sorted_f1):
        ax2.text(score + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Extra space at bottom
    
    # Save paper version
    output_path = Path('evaluation_results')
    fig.savefig(output_path / 'lstm_per_class_paper_ready.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ Paper-ready version saved to {output_path / 'lstm_per_class_paper_ready.png'}")
    plt.show()

if __name__ == "__main__":
    print("🔧 Creating final fixed LSTM per-class analysis...")
    
    # Create both versions
    create_final_fixed_analysis()
    print("\n" + "="*50)
    create_paper_ready_version() 