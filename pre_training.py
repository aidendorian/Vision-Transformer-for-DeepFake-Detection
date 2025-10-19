import dataloaders
import vision_transformer
import visualization
# import With_Freq
import checkpointing
import torch
import time
from torch.amp import autocast, GradScaler
from tqdm import tqdm

scaler = GradScaler()

torch.manual_seed(42)

class Config:
    def __init__(self):
        self.batch_size = 150
        self.epoch = 4
        self.num_workers = 4
        self.device = "cuda"
        self.persistent_workers = True
        self.prefetch_factor = 2
        self.pin_memory = True
        
        self.img_size = 224
        self.in_channels = 3
        self.patch_size = 16
        self.num_transformer_layers = 12
        self.embedding_dim = 896
        self.mlp_size = 3072
        self.num_heads = 14
        self.attn_dropout = 0
        self.mlp_dropout = 0.1
        self.embedding_dropout = 0.1
        self.phase = "pretraining"
        self.projection_head_mode = "linear"
        self.mask_ratio = 0.75
        self.num_classes = 2
        self.norm_pix = False
        self.resume_training = False
        self.checkpoint_path = 'checkpoints/checkpoint_spatial_pretraining_epoch_1.pt'

config = Config()

VisionTransformer = vision_transformer.ViT(
    batch_size=config.batch_size,
    img_size=config.img_size,
    in_channels=config.in_channels,
    patch_size=config.patch_size,
    num_transformer_layers=config.num_transformer_layers,
    embedding_dim=config.embedding_dim,
    mlp_size=config.mlp_size,
    num_heads=config.num_heads,
    attn_dropout=config.attn_dropout,
    mlp_dropout=config.mlp_dropout,
    embedding_dropout=config.embedding_dropout,
    phase=config.phase,
    projection_head_mode=config.projection_head_mode,
    mask_ratio=config.mask_ratio,
    num_classes=config.num_classes,
    norm_pix=config.norm_pix
)

pretrain_data = dataloaders.get_dataloader(
    phase=config.phase,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
    prefetch_factor=config.prefetch_factor,
    persistent_workers=config.persistent_workers
)

optimizer = torch.optim.AdamW(VisionTransformer.parameters(), lr=0.0001)
visualizer = visualization.MIMVisualizer()

VisionTransformer.to(config.device)
VisionTransformer.train()

all_losses = []
all_steps = []
current_step = 0
loss_value = 0.0

start_epoch = 0

if config.resume_training:
    start_epoch, _, VisionTransformer, optimizer, scaler = checkpointing.load_checkpoint(VisionTransformer,
                                                                                         optimizer,
                                                                                         scaler,
                                                                                         config.checkpoint_path)
else:
    print('Starting Training from Scratch')

for epoch in range(start_epoch, config.epoch):
    start_time = time.time()
    epoch_loss = 0.
    steps = 0
    
    for data in tqdm(pretrain_data, desc=f"Epoch {epoch+1}/{config.epoch}"):
        data = data.to(config.device)
        optimizer.zero_grad()
        
        with autocast(device_type=config.device):
            reconstructed_patches, loss = VisionTransformer(data)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_value = loss.item()
        
        current_step += 1
        all_losses.append(loss_value)
        all_steps.append(current_step)
        epoch_loss += loss_value
        steps += 1
        
        if steps % 250 == 0:         
            with torch.no_grad():
                sample_data = data[:1]
                
                B, C, H, W = sample_data.shape
                P = config.patch_size
                N = (H // P) * (W // P)
                
                mask = torch.ones(1, 2*N, device=data.device)
                mask_ratio = config.mask_ratio
                len_keep = int(2*N * (1 - mask_ratio))
                mask[:, :len_keep] = 0
                
                reconstructed_patches_vis = reconstructed_patches[:1].float()
                
                visualizer.visualize_patch_reconstruction(
                    sample_data, reconstructed_patches_vis, mask, current_step, loss_value, config.patch_size
                )
                
                visualizer.plot_loss_curve(all_losses, all_steps, current_step)
    
    epoch_loss /= steps
    end_time = time.time()
    
    print(f"Epoch {epoch+1}/{config.epoch} | Avg Loss: {epoch_loss:.6f} | Time: {end_time-start_time:.2f}s")
    
    checkpointing.save_checkpoint(VisionTransformer,
                                  optimizer,
                                  scaler,
                                  epoch+1,
                                  'checkpoints',
                                  f'checkpoint_spatial_pretraining_epoch_{epoch+1}.pth',
                                  epoch_loss,
                                  0)

torch.save(VisionTransformer.state_dict(), f"models/Spatial_Only_More_than_Base_PreTrained_{config.epoch}.pth")