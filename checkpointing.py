import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, f"{path}/checkpoint_epoch_{epoch}.pth")
    print(f"Checkpoint saved: {path}/checkpoint_epoch_{epoch}.pth")

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from epoch {epoch + 1}, previous loss: {loss:.6f}")
    
    return epoch, loss