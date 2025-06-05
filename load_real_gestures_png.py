#!/usr/bin/env python3
"""Load your actual gesture images (.png format) into the figure."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def create_gesture_figure_with_png_images():
    """Create figure with your actual .png gesture images."""
    
    # Your 7 gestures
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
        
        # Look for .png files
        image_path = f"gesture_{i+1}.png"
        
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
            print(f"✓ Loaded {image_path}")
        except FileNotFoundError:
            print(f"❌ Could not find {image_path}")
            ax.text(0.5, 0.5, f'Missing:\n{image_path}', 
                   ha='center', va='center', fontsize=12,
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        # Add labels
        command, description = gesture_info[i]
        ax.set_title(command, fontsize=14, fontweight='bold', color='navy')
        ax.text(0.5, -0.15, description, ha='center', va='top', 
               transform=ax.transAxes, fontsize=10, style='italic')
        ax.axis('off')
    
    # Bottom row: 3 gestures (centered)
    for i in range(3):
        ax = plt.subplot(2, 4, i+6)  # positions 6, 7, 8 (skipping 5)
        
        image_path = f"gesture_{i+5}.png"  # gestures 5, 6, 7
        
        try:
            img = mpimg.imread(image_path)
            ax.imshow(img)
            print(f"✓ Loaded {image_path}")
        except FileNotFoundError:
            print(f"❌ Could not find {image_path}")
            ax.text(0.5, 0.5, f'Missing:\n{image_path}', 
                   ha='center', va='center', fontsize=12,
                   transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
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
    fig.text(0.5, 0.02, 
            'Static gestures captured via standard webcam input for real-time classification and drone command generation',
            ha='center', fontsize=12, style='italic', color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save the figure
    output_path = Path('figures')
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / 'gesture_examples.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"\n🎯 Final gesture figure saved to {output_path / 'gesture_examples.png'}")
    plt.show()

# Also create a version that checks what files you actually have
def check_available_images():
    """Check what image files are available."""
    import os
    
    print("📁 Checking for image files...")
    png_files = [f for f in os.listdir(".") if f.lower().endswith('.png')]
    jpg_files = [f for f in os.listdir(".") if f.lower().endswith(('.jpg', '.jpeg'))]
    
    print(f"Found {len(png_files)} PNG files:")
    for f in sorted(png_files):
        print(f"  ✓ {f}")
        
    print(f"Found {len(jpg_files)} JPG files:")
    for f in sorted(jpg_files):
        print(f"  ✓ {f}")
    
    if not png_files and not jpg_files:
        print("❌ No image files found in current directory!")
        print("Make sure your gesture images are in the same folder as this script.")

if __name__ == "__main__":
    print("🖼️  Creating gesture figure with PNG images...")
    check_available_images()
    print("\n" + "="*50)
    create_gesture_figure_with_png_images() 