import dataloaders
import vision_transformer
import checkpointing
import torch
import time
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import Freq_Spatial

scaler = GradScaler()
torch.manual_seed(42)

class Config:
    def __init__(self):
        self.batch_size = 32
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
        self.embedding_dropout = 0.2
        self.phase = "finetuning"
        self.projection_head_mode = "linear"
        self.mask_ratio = 0.75
        self.num_classes = 2
        self.norm_pix = False
        self.unfreezed_transformer_layers = 4
        self.resume_training = True
        self.checkpoint_path = "checkpoints/checkpoint_finetuning_epoch_3_last_4.pth"
        self.pretrained_model_state_dict = 'More_than_Base_PreTrained_4.pth'

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

train_data, val_data, test_data = dataloaders.get_dataloader(phase=config.phase,
                                                             batch_size=config.batch_size,
                                                             num_workers=config.num_workers,
                                                             pin_memory=config.pin_memory,
                                                             prefetch_factor=config.prefetch_factor,
                                                             persistent_workers=config.persistent_workers
                                                             )

optimizer = torch.optim.AdamW(VisionTransformer.parameters(), lr=0.00001)
loss_fn = torch.nn.CrossEntropyLoss()

VisionTransformer.to(config.device)
pretrained_state_dict =  torch.load(f"models/{config.pretrained_model_state_dict}")
current_state_dict = VisionTransformer.state_dict()
filtered_state_dict = {}

for k, v in pretrained_state_dict.items():
    if k in current_state_dict and current_state_dict[k].shape == v.shape:
        filtered_state_dict[k] = v
        
class_embedding = torch.rand(1, 1, config.embedding_dim, requires_grad=True, device=config.device)
torch.nn.init.normal_(class_embedding, std=0.02)
filtered_state_dict['patch_and_embed.position_embed'] = torch.cat((pretrained_state_dict['patch_and_embed.position_embed'], class_embedding), dim=1)
VisionTransformer.load_state_dict(filtered_state_dict)

for name, param in VisionTransformer.named_parameters():
    if name == 'patch_and_embed.position_embed':
        param.requires_grad = False
        
for i, block in enumerate(VisionTransformer.transformer_encoder):
    if i < config.num_transformer_layers - config.unfreezed_transformer_layers:
        for param in block.parameters():
            param.requires_grad = False
    else:
        for param in block.parameters():
            param.requires_grad = True

all_losses = []
all_steps = []
current_step = 0
epoch_saved = 0
start_epoch = 0

if config.resume_training:
    print("Loading checkpoint...")
    start_epoch, _, VisionTransformer, optimizer, scaler = checkpointing.load_checkpoint(
        VisionTransformer, optimizer, scaler, config.checkpoint_path
    )
    print(f"Resuming from epoch {start_epoch + 1}")
    print(f"GradScaler state: {scaler.state_dict()}")
else:
    print("Starting training from scratch")
    
for epoch in range(start_epoch, config.epoch):
    start_time = time.time()
    epoch_loss = 0.
    epoch_acc = 0.
    steps = 0
    
    VisionTransformer.train()
    for train_img, train_labels in tqdm(train_data, desc=f"Epoch {epoch+1}/{config.epoch}"):
        train_img, train_labels = train_img.to(config.device), train_labels.to(config.device)
        optimizer.zero_grad()
        
        with autocast(device_type=config.device):
            train_preds = VisionTransformer(train_img)
            loss = loss_fn(train_preds, train_labels)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_value = loss.item()
        
        _, preds = torch.max(train_preds, dim=1)
        train_acc = (preds == train_labels).float().mean().item()
        
        current_step += 1
        all_losses.append(loss_value)
        all_steps.append(current_step)
        epoch_loss += loss_value
        epoch_acc += train_acc
        steps += 1
    
    epoch_loss /= steps
    epoch_acc /= steps

    VisionTransformer.eval()
    val_loss = 0.0
    val_steps = 0
    
    with torch.no_grad():
        for val_img, val_labels in tqdm(val_data, desc='Validating'):
            val_img, val_labels = val_img.to(config.device), val_labels.to(config.device)
            
            with autocast(device_type=config.device):
                val_preds = VisionTransformer(val_img)
                val_loss += loss_fn(val_preds, val_labels).item()
                val_steps += 1
    
    val_loss /= val_steps
    
    end_time = time.time()
    
    print(f"Epoch {epoch+1}/{config.epoch} | Train Loss: {epoch_loss:.6f} | Train Acc: {epoch_acc:.6f} | Val Loss: {val_loss:.6f} | Time: {end_time-start_time:.2f}s")
    
    checkpointing.save_checkpoint(
        VisionTransformer, 
        optimizer, 
        scaler,
        epoch + 1,
        "checkpoints", 
        f'checkpoint_finetuning_epoch_{epoch+1}_last_{config.unfreezed_transformer_layers}.pth',
        loss=val_loss,
        accuracy=epoch_acc
    )

torch.save(VisionTransformer.state_dict(), f"models/More_than_Base_Finetuned_{config.epoch}_Last_{config.unfreezed_transformer_layers}.pt")

VisionTransformer.eval()
test_loss = 0.
test_steps = 0
test_acc = 0.

with torch.no_grad():
    for test_img, test_labels in tqdm(test_data, desc='Testing'):
        test_img, test_labels = test_img.to(config.device), test_labels.to(config.device)
        
        with autocast(device_type=config.device):
            test_preds = VisionTransformer(test_img)
            test_loss += loss_fn(test_preds, test_labels).item()
            
            _, preds = torch.max(test_preds, dim=1)
            test_acc += (preds == test_labels).float().mean().item()
            test_steps += 1
                
test_loss /= test_steps
test_acc /= test_steps

print(f'Test Loss: {test_loss} | Test Accuracy: {test_acc}')