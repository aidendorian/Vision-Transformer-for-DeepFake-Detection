import torch
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)

class MIMVisualizer:
    def __init__(self, save_dir="visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def normalize_for_display(self, tensor):
        """Safe normalization with clamping"""
        tensor = tensor.clone().detach().float()
        min_val = tensor.min()
        max_val = tensor.max()
        
        if (max_val - min_val) < 1e-8:
            return torch.zeros_like(tensor)
        
        tensor = (tensor - min_val) / (max_val - min_val)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor
    
    def visualize_patch_reconstruction(self, original_img, reconstructed_patches, mask, step, loss_value, patch_size=16):
        """
        Visualize original vs reconstructed patches with proper dual-path handling
        """
        B, C, H, W = original_img.shape
        P = patch_size
        N = (H // P) * (W // P)
        
        pred_patches = reconstructed_patches[0].view(2*N, -1)  # [2*N, C*P*P]
        
        spatial_patches_pred = pred_patches#[:N]    # First N: spatial
        #freq_patches_pred = pred_patches[N:]       # Next N: frequency
        
        # Print diagnostic ranges
        spatial_range = [spatial_patches_pred.min().item(), spatial_patches_pred.max().item()]
        #freq_range = [freq_patches_pred.min().item(), freq_patches_pred.max().item()]
        #full_range = [reconstructed_patches[0].min().item(), reconstructed_patches[0].max().item()]
        
        with open('recon_range.txt', 'a') as f:
            #f.write(f'Step {step}\nFull range: {full_range}\nSpatial-only: {spatial_range}\nFrequency-only: {freq_range}')
            f.write(f'Step {step}\nSpatial-only: {spatial_range}')
            f.write('\n\n')
        
        original_normalized = self.normalize_for_display(original_img[0])
        
        spatial_patches_pred = spatial_patches_pred.view(N, C, P, P)
        spatial_patches_normalized = self.normalize_for_display(spatial_patches_pred)
        
        patches_per_row = W // P
        reconstructed_img = self.patches_to_image(spatial_patches_normalized, patches_per_row)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0,0].imshow(original_normalized.permute(1, 2, 0).cpu().numpy())
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(reconstructed_img.permute(1, 2, 0).cpu().numpy())
        axes[0,1].set_title('Reconstructed (Spatial)')
        axes[0,1].axis('off')
        
        diff = torch.abs(original_normalized - reconstructed_img)
        axes[0,2].imshow(diff.permute(1, 2, 0).cpu().numpy(), cmap='hot')
        axes[0,2].set_title('Difference Map')
        axes[0,2].axis('off')
        
        mask_img = mask[0].view(patches_per_row, patches_per_row*2).cpu().numpy()
        axes[1,0].imshow(mask_img, cmap='gray')
        axes[1,0].set_title(f'Mask (White=Masked)\nRatio: {mask[0].mean().item():.2f}')
        axes[1,0].axis('off')
        
        self.plot_patch_statistics(original_img[0], spatial_patches_pred, axes[1,1])
        
        info_text = f'Step: {step}\nLoss: {loss_value:.4f}\n'
        info_text += f'Spatial: [{spatial_range[0]:.3f}, {spatial_range[1]:.3f}]\n'
        # info_text += f'Freq: [{freq_range[0]:.3f}, {freq_range[1]:.3f}]'
        
        axes[1,2].text(0.5, 0.5, info_text, 
                      ha='center', va='center', transform=axes[1,2].transAxes, fontsize=10)
        axes[1,2].set_title('Training Info + Ranges')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/recon_step_{step:06d}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # print(f"Visualization saved - Original: [{original_img[0].min():.3f}, {original_img[0].max():.3f}]")
    
    def patches_to_image(self, patches, patches_per_row):
        """Convert patches back to full image with safety checks"""
        if len(patches.shape) == 4:
            N, C, P, _ = patches.shape
            patches = patches.view(patches_per_row, patches_per_row, C, P, P)
            patches = patches.permute(2, 0, 3, 1, 4)
            image = patches.contiguous().view(C, patches_per_row * P, patches_per_row * P)
        else:
            raise ValueError(f"Unexpected patches shape: {patches.shape}")
        return image
    
    def plot_patch_statistics(self, original_img, reconstructed_patches, ax):
        """Plot histogram of patch values (spatial only for fair comparison)"""
        orig_vals = original_img.flatten().cpu().numpy()
        recon_vals = reconstructed_patches.flatten().cpu().numpy()
        
        ax.hist(orig_vals, bins=50, alpha=0.7, label='Original', color='blue', density=True)
        ax.hist(recon_vals, bins=50, alpha=0.7, label='Reconstructed', color='red', density=True)
        ax.set_xlabel('Pixel Values')
        ax.set_ylabel('Density')
        ax.set_title('Spatial Value Distribution')
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