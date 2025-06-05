#!/usr/bin/env python3
"""Create combined gesture figure for the paper."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def create_gesture_figure():
    """Create a combined figure of all gesture examples."""
    
    # Define gesture names based on the images (you can adjust these labels)
    gesture_labels = [
        "Takeoff\n(Thumbs Up)",
        "Up\n(Palm Down, Spread)",
        "Down\n(Point Down)", 
        "Right\n(Point Right)",
        "Left\n(Point Left)",
        "Forward\n(Point Forward)",
        "Stop/Hover\n(Closed Fist)",
    ]
    
    # Image file paths (adjust these to match your actual file names)
    image_files = [
        "gesture_1.jpg",  # Thumbs up
        "gesture_2.jpg",  # Up
        "gesture_3.jpg",  # Down
        "gesture_4.jpg",  # Right
        "gesture_5.jpg",  # Left
        "gesture_6.jpg",  # Forward
        "gesture_7.jpg",  # Stop
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Hand Gesture Commands for Drone Control', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Load and display each image
    for i, (img_file, label) in enumerate(zip(image_files, gesture_labels)):
        try:
            # Load image
            img = mpimg.imread(img_file)
            
            # Display image
            axes[i].imshow(img)
            axes[i].set_title(label, fontsize=12, fontweight='bold', pad=10)
            axes[i].axis('off')  # Remove axes
            
            # Add border around image
            for spine in axes[i].spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(1)
                spine.set_visible(True)
                
        except FileNotFoundError:
            # If image not found, create placeholder
            axes[i].text(0.5, 0.5, f'Image {i+1}\n{label}', 
                        ha='center', va='center', fontsize=10,
                        transform=axes[i].transAxes)
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
            axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.1)
    
    # Save the figure
    output_path = Path('figures')
    output_path.mkdir(exist_ok=True)
    
    fig.savefig(output_path / 'gesture_examples.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Gesture figure saved to {output_path / 'gesture_examples.png'}")
    plt.show()

def create_gesture_figure_from_attachments():
    """
    If you have the images as separate files, save them first then run this.
    This assumes you've saved the images as gesture_1.jpg through gesture_8.jpg
    """
    
    # More descriptive labels based on your drone control system
    gesture_info = [
        ("Takeoff", "Thumbs up gesture\nto initiate flight"),
        ("Ascend", "Open palm facing down\nfingers spread upward"),
        ("Descend", "Index finger pointing\ndownward motion"),
        ("Move Right", "Index finger pointing\nto the right"),
        ("Move Left", "Index finger pointing\nto the left"), 
        ("Move Forward", "Index finger pointing\nstraight ahead"),
        ("Stop/Hover", "Closed fist to halt\nall movement"),
        ("Move Backward", "Index finger pointing\nupward/backward")
    ]
    
    # Create a more detailed figure
    fig = plt.figure(figsize=(18, 10))
    
    # Create a grid layout
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.2)
    
    for i, ((command, description), img_num) in enumerate(zip(gesture_info, range(1, 9))):
        row = i // 4
        col = i % 4
        
        ax = fig.add_subplot(gs[row, col])
        
        try:
            # Try to load the actual image
            img_path = f"gesture_{img_num}.jpg"
            img = mpimg.imread(img_path)
            ax.imshow(img)
        except FileNotFoundError:
            # Create a placeholder if image not found
            ax.text(0.5, 0.5, f'Gesture {img_num}', 
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Style the subplot
        ax.set_title(command, fontsize=14, fontweight='bold', color='navy')
        ax.text(0.5, -0.15, description, ha='center', va='top', 
               transform=ax.transAxes, fontsize=10, 
               style='italic', color='darkslategray')
        ax.axis('off')
        
        # Add subtle border
        rect = plt.Rectangle((0, 0), 1, 1, fill=False, 
                           edgecolor='lightgray', linewidth=2,
                           transform=ax.transAxes)
        ax.add_patch(rect)
    
    # Add main title and subtitle
    fig.suptitle('Hand Gesture Vocabulary for Intuitive Drone Control', 
                fontsize=18, fontweight='bold', y=0.95)
    fig.text(0.5, 0.02, 
            'Static gestures captured via standard webcam input for real-time classification and drone command generation',
            ha='center', fontsize=12, style='italic', color='gray')
    
    # Save with high quality
    output_path = Path('figures')
    output_path.mkdir(exist_ok=True)
    
    fig.savefig(output_path / 'gesture_examples.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Professional gesture figure saved to {output_path / 'gesture_examples.png'}")
    plt.show()

if __name__ == "__main__":
    print("Creating gesture figure...")
    print("Make sure to save your attached images as gesture_1.jpg through gesture_8.jpg")
    create_gesture_figure_from_attachments() 