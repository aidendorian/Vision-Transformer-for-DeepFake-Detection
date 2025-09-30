import dataloaders
import vision_transformer
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os

torch.manual_seed(42)

class MIMVisualizer:
    def __init__(self, save_dir="visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def normalize_for_display(self, tensor):
        """Normalize any tensor to [0,1] range for visualization"""
        tensor = tensor - tensor.min()
        tensor = tensor / (tensor.max() + 1e-8)
        return tensor
    
    def visualize_patch_reconstruction(self, original_img, reconstructed_patches, mask, step, loss_value, patch_size=16):
        """
        Visualize original vs reconstructed patches without norm_pix assumption
        """
        B, C, H, W = original_img.shape
        P = patch_size
        N = (H // P) * (W // P)
        
        original_normalized = self.normalize_for_display(original_img[0])
        
        pred_patches = reconstructed_patches[0].view(2*N, -1)
        
        spatial_patches = pred_patches[:N]
        spatial_patches = spatial_patches.view(N, C, P, P)
        
        spatial_patches = self.normalize_for_display(spatial_patches)
        
        patches_per_row = W // P
        reconstructed_img = self.patches_to_image(spatial_patches, patches_per_row)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0,0].imshow(original_normalized.permute(1, 2, 0).cpu().numpy())
        axes[0,0].set_title('Original Image (Normalized)')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(reconstructed_img.permute(1, 2, 0).cpu().numpy())
        axes[0,1].set_title('Reconstructed Image (Normalized)')
        axes[0,1].axis('off')
        
        diff = torch.abs(original_normalized - reconstructed_img)
        axes[0,2].imshow(diff.permute(1, 2, 0).cpu().numpy(), cmap='hot')
        axes[0,2].set_title('Difference Map')
        axes[0,2].axis('off')
        
        mask_img = mask[0].view(patches_per_row, patches_per_row*2).cpu().numpy()
        axes[1,0].imshow(mask_img, cmap='gray')
        axes[1,0].set_title(f'Mask Pattern\n(White=Masked, Ratio: {mask[0].mean().item():.2f})')
        axes[1,0].axis('off')
        
        self.plot_patch_statistics(original_img[0], reconstructed_patches[0], axes[1,1])
        
        axes[1,2].text(0.5, 0.5, f'Step: {step}\nLoss: {loss_value:.4f}', 
                      ha='center', va='center', transform=axes[1,2].transAxes, fontsize=12)
        axes[1,2].set_title('Training Info')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/recon_step_{step:06d}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved - Original range: [{original_img[0].min():.3f}, {original_img[0].max():.3f}]")
        print(f"Reconstructed range: [{reconstructed_patches[0].min():.3f}, {reconstructed_patches[0].max():.3f}]")
    
    def patches_to_image(self, patches, patches_per_row):
        """Convert patches back to full image"""
        P = patches.shape[-1]
        patches = patches.view(patches_per_row, patches_per_row, 3, P, P)
        patches = patches.permute(2, 0, 3, 1, 4)
        image = patches.contiguous().view(3, patches_per_row * P, patches_per_row * P)
        return image
    
    def plot_patch_statistics(self, original_img, reconstructed_patches, ax):
        """Plot histogram of patch values"""
        orig_vals = original_img.flatten().cpu().numpy()
        recon_vals = reconstructed_patches.flatten().cpu().numpy()
        
        ax.hist(orig_vals, bins=50, alpha=0.7, label='Original', color='blue', density=True)
        ax.hist(recon_vals, bins=50, alpha=0.7, label='Reconstructed', color='red', density=True)
        ax.set_xlabel('Pixel Values')
        ax.set_ylabel('Density')
        ax.set_title('Value Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_loss_curve(self, losses, steps, step):
        """Plot loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, 'b-', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('MIM Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.save_dir}/loss_curve_step_{step:06d}.png', dpi=100, bbox_inches='tight')
        plt.close()