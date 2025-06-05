#!/usr/bin/env python3
"""Fixed gesture figure with proper spacing to avoid text overlap."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def create_gesture_figure_fixed_spacing():
    """Create figure with proper spacing to avoid text overlap."""
    
    # Your 7 gestures
    gesture_info = [
        ("Takeoff", "Open palm spread\nfingers upward"),
        ("Up", "Thumbs up gesture\nto initiate flight"),  
        ("Down", "Index finger pointing\ndownward motion"),
        ("Right", "Index finger pointing\nto the right"),
        ("Left", "Index finger pointing\nto the left"),
        ("Forward", "Index finger pointing\nstraight ahead"),
        ("Stop", "Closed fist to halt\nall movement")
    ]
    
    # Create figure with more vertical space
    fig = plt.figure(figsize=(16, 12))  # Increased height
    
    # Top row: 4 gestures with more vertical spacing
    for i in range(4):
        ax = plt.subplot(2, 4, i+1)
        
        image_path = f"gesture_{i+1}.png"
        
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Missing:\n{image_path}', 
                   ha='center', va='center', fontsize=12,
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        # Add labels with adjusted positioning
        command, description = gesture_info[i]
        ax.set_title(command, fontsize=14, fontweight='bold', color='navy', pad=15)
        
        # Move description text higher up to avoid overlap
        ax.text(0.5, -0.25, description, ha='center', va='top', 
               transform=ax.transAxes, fontsize=10, style='italic')
        ax.axis('off')
    
    # Bottom row: 3 gestures (centered) with more space from top row
    for i in range(3):
        ax = plt.subplot(2, 4, i+6)  # positions 6, 7, 8
        
        image_path = f"gesture_{i+5}.png"  # gestures 5, 6, 7
        
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Missing:\n{image_path}', 
                   ha='center', va='center', fontsize=12,
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        # Add labels with proper spacing
        command, description = gesture_info[i+4]
        ax.set_title(command, fontsize=14, fontweight='bold', color='navy', pad=15)
        
        # Position description text properly
        ax.text(0.5, -0.20, description, ha='center', va='top', 
               transform=ax.transAxes, fontsize=10, style='italic')
        ax.axis('off')
    
    # Hide the empty subplot (position 5)
    ax_empty = plt.subplot(2, 4, 5)
    ax_empty.axis('off')
    
    # Add main title with proper spacing
    fig.suptitle('Hand Gesture Vocabulary for Intuitive Drone Control', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Move subtitle higher to avoid overlap
    fig.text(0.5, 0.05, 
            'Static gestures captured via standard webcam input for real-time classification and drone command generation',
            ha='center', fontsize=12, style='italic', color='gray')
    
    # Adjust layout with more spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12, hspace=0.6, wspace=0.2)  # Increased hspace
    
    # Save the figure
    output_path = Path('figures')
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / 'gesture_examples_fixed.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"🎯 Fixed gesture figure saved to {output_path / 'gesture_examples_fixed.png'}")
    plt.show()

# Alternative version with even more spacing
def create_gesture_figure_extra_space():
    """Create figure with extra spacing for better text separation."""
    
    gesture_info = [
        ("Takeoff", "Open palm spread\nfingers upward"),
        ("Up", "Thumbs up gesture\nto initiate flight"),  
        ("Down", "Index finger pointing\ndownward motion"),
        ("Right", "Index finger pointing\nto the right"),
        ("Left", "Index finger pointing\nto the left"),
        ("Forward", "Index finger pointing\nstraight ahead"),
        ("Stop", "Closed fist to halt\nall movement")
    ]
    
    # Create figure with generous spacing
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    # Flatten axes for easier access
    axes = axes.flatten()
    
    # Top row: gestures 1-4
    for i in range(4):
        ax = axes[i]
        image_path = f"gesture_{i+1}.png"
        
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Image {i+1}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        command, description = gesture_info[i]
        ax.set_title(command, fontsize=14, fontweight='bold', color='navy', pad=20)
        ax.text(0.5, -0.15, description, ha='center', va='top', 
               transform=ax.transAxes, fontsize=10, style='italic')
        ax.axis('off')
    
    # Empty slot (position 4 in second row)
    axes[4].axis('off')
    
    # Bottom row: gestures 5-7 (positions 5, 6, 7)
    for i in range(3):
        ax = axes[i+5]
        image_path = f"gesture_{i+5}.png"
        
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Image {i+5}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        command, description = gesture_info[i+4]
        ax.set_title(command, fontsize=14, fontweight='bold', color='navy', pad=20)
        ax.text(0.5, -0.15, description, ha='center', va='top', 
               transform=ax.transAxes, fontsize=10, style='italic')
        ax.axis('off')
    
    # Add title and subtitle
    fig.suptitle('Hand Gesture Vocabulary for Intuitive Drone Control', 
                fontsize=18, fontweight='bold', y=0.95)
    fig.text(0.5, 0.02, 
            'Static gestures captured via standard webcam input for real-time classification and drone command generation',
            ha='center', fontsize=12, style='italic', color='gray')
    
    # Generous spacing
    plt.subplots_adjust(top=0.85, bottom=0.15, hspace=0.8, wspace=0.1)
    
    # Save the figure
    output_path = Path('figures')
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / 'gesture_examples_clean.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"🎯 Clean gesture figure saved to {output_path / 'gesture_examples_clean.png'}")
    plt.show()

if __name__ == "__main__":
    print("🖼️  Creating gesture figure with fixed spacing...")
    
    # Try the extra space version first
    create_gesture_figure_extra_space()
    
    print("\nAlso creating the alternative version...")
    create_gesture_figure_fixed_spacing() 