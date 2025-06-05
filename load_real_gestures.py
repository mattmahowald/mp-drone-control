#!/usr/bin/env python3
"""Load your actual gesture images into the figure."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np
from PIL import Image

def create_gesture_figure_with_real_images():
    """Create figure with your actual gesture images."""
    
    # Your 7 gestures (adjust labels to match what you actually have)
    gesture_info = [
        ("Takeoff", "Thumbs up gesture\nto initiate flight"),
        ("Up", "Open palm spread\nfingers upward"),  
        ("Down", "Index finger pointing\ndownward motion"),
        ("Right", "Index finger pointing\nto the right"),
        ("Left", "Index finger pointing\nto the left"),
        ("Forward", "Index finger pointing\nstraight ahead"),
        ("Stop/Hover", "Closed fist to halt\nall movement")
    ]
    
    # Create figure with 2 rows: 4 on top, 3 on bottom
    fig = plt.figure(figsize=(16, 10))
    
    # Top row: 4 gestures
    for i in range(4):
        ax = plt.subplot(2, 4, i+1)
        
        # Load your actual image files
        # You'll need to replace these with your actual file paths
        image_path = f"gesture_{i+1}.jpg"  # or .png, whatever format you have
        
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
        except FileNotFoundError:
            # If image not found, create placeholder
            ax.text(0.5, 0.5, f'Add your\ngesture {i+1}\nimage here', 
                   ha='center', va='center', fontsize=12,
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Add labels
        command, description = gesture_info[i]
        ax.set_title(command, fontsize=14, fontweight='bold', color='navy')
        ax.text(0.5, -0.15, description, ha='center', va='top', 
               transform=ax.transAxes, fontsize=10, style='italic')
        ax.axis('off')
    
    # Bottom row: 3 gestures (centered)
    for i in range(3):
        ax = plt.subplot(2, 4, i+6)  # positions 6, 7, 8 (skipping 5)
        
        image_path = f"gesture_{i+5}.jpg"  # gestures 5, 6, 7
        
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'Add your\ngesture {i+5}\nimage here', 
                   ha='center', va='center', fontsize=12,
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Add labels
        command, description = gesture_info[i+4]
        ax.set_title(command, fontsize=14, fontweight='bold', color='navy')
        ax.text(0.5, -0.15, description, ha='center', va='top', 
               transform=ax.transAxes, fontsize=10, style='italic')
        ax.axis('off')
    
    # Hide the empty subplot (position 5)
    plt.subplot(2, 4, 5)
    plt.axis('off')
    
    # Add main title
    fig.suptitle('Hand Gesture Vocabulary for Intuitive Drone Control', 
                fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save the figure
    output_path = Path('figures')
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / 'gesture_examples_real.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Real gesture figure saved to {output_path / 'gesture_examples_real.png'}")
    plt.show()

if __name__ == "__main__":
    create_gesture_figure_with_real_images() 