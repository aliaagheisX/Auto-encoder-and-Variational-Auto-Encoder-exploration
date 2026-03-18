import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x) # C, H, W
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255 # H, W, C 
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im



def make_grid(images, width=256, height=256):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (width * len(images), height))
    for i, im in enumerate(images):
        output_im.paste(im.resize((width, height)), (i * width, 0))
    return output_im



def get_pil_image(x):
    """Given image matrix x, convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    im = x.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255 # H, W, C 
    im = Image.fromarray(np.array(im).astype(np.uint8))
    return im



def show_images_with_labels(images, labels):
    """Plots a list of images side-by-side with labels underneath."""
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        # We can use set_xlabel to put the text directly under the image
        ax.set_xlabel(label, fontsize=14, fontweight='bold')
        ax.set_xticks([]) # Remove axis ticks for a cleaner look
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.show()



def plot_details_from_point(images, labels, point, patch_size=60, ylabel=""):
    """
    Creates a grid zooming in on a specific (x, y) point across multiple images.
    Columns = Models.
    """
    num_images = len(images)
    
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    
    half = patch_size // 2
    cx, cy = point

    if num_images == 1:
        axes = [axes]  # Make iterable

    for col_idx, (img, label) in enumerate(zip(images, labels)):
        ax = axes[col_idx]
        ax.imshow(img)
        
        # Zoom in by setting axis limits around the center point
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_title(label, fontsize=14)
        
        # Add coordinate label to the first column
        if col_idx == 0:
            ax.set_ylabel(ylabel, fontsize=12, rotation=0, labelpad=30, ha='center', va='center')

    plt.tight_layout()
    plt.show()