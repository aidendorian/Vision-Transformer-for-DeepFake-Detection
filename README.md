# Vision Transformer for DeepFake Detection

A PyTorch implementation of Vision Transformer (ViT) for detecting deepfake images using masked image modeling (MIM) pretraining followed by supervised fine-tuning with Grad-CAM visualization support.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Overview

This project implements a comprehensive two-stage approach to deepfake detection:

1. **Self-Supervised Pretraining**: Masked Image Modeling (MIM) where the model learns robust visual representations by reconstructing randomly masked image patches (75% masking ratio)
2. **Supervised Fine-tuning**: Transfer learning on labeled real/fake images with selective layer unfreezing for optimal performance
3. **Interpretable Predictions**: Grad-CAM visualization to highlight manipulated regions in deepfake images

## Model Accuracy
Model achieves **85.7%** accuracy in DFDC's Test split and a ROC-AUC of **0.9363**.

## Key Features

- **Masked Autoencoding**: Self-supervised pretraining inspired by MAE (Masked Autoencoders)
- **Flexible Architecture**: Configurable transformer depth, embedding dimensions, and attention heads
- **Mixed Precision Training**: FP16 automatic mixed precision for faster training and reduced memory usage
- **Grad-CAM Visualization**: Interpretable model predictions showing attention on suspicious regions
- **Checkpoint Management**: Resume training from saved checkpoints with optimizer and scaler states
- **Comprehensive Evaluation**: Classification reports, confusion matrices, and ROC-AUC metrics

## Datasets Used
- **Pretraining**: ```torchvision.datasets.CelebA()``` CelebA's train split, ~160K images.
- **Finetuning**: DFDC's Train split ~140K images, Validation split ~40K images and Test split ~11K images.   

## Architecture

The Vision Transformer architecture consists of:

- **Patch Embedding Layer**: Converts 224×224 images into sequences of 16×16 patch embeddings
- **Transformer Encoder**: 12-14 layers with multi-head self-attention (12-14 heads) and feedforward MLP blocks
- **Pretraining Head**: Reconstructs masked patches using linear or convolutional projection
- **Classification Head**: Layer normalization + linear classifier for binary real/fake classification
- **Grad-CAM Integration**: Attention visualization on final transformer layer

### Model Specifications

| Component | Default Configuration |
|-----------|----------------------|
| Image Size | 224×224 pixels |
| Patch Size | 16×16 pixels |
| Embedding Dimension | 768 (Base) / 896 (Large) |
| Transformer Layers | 12-14 layers |
| Attention Heads | 12-14 heads |
| MLP Hidden Size | 3072 |
| Mask Ratio (Pretraining) | 75% |
| Total Parameters | ~86M (Base) / ~120M (Large) |

## Project Structure

```
Vision-Transformer-for-DeepFake-Detection/
├── checkpoints/
├── data/
│   ├── pretraining/                
│   └── finetuning/
│       ├── Train/
│       │   ├── FAKE/              
│       │   └── REAL/
│       ├── Validation/
│       │   ├── FAKE/
│       │   └── REAL/
│       └── Test/
│           ├── FAKE/
│           └── REAL/
├── models/                         
├── results/            
├── visualizations/
├── checkpointing.py                
├── dataloaders.py                  
├── grad_cam.py                     
├── pre_training.py                 
├── fine_tuning.py                  
├── infer.py                        
├── vision_transformer.py           
├── visualization.py
```

## Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=9.5.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## Usage

### 1. Self-Supervised Pretraining

Train the Vision Transformer using masked image modeling:

```bash
python pre_training.py
```

**Key Configuration** (edit `Config` class in `pre_training.py`):

```python
class Config:
    batch_size = 120                    
    epoch = 4
    embedding_dim = 896                 
    num_transformer_layers = 12         
    num_heads = 14                      
    mask_ratio = 0.75                   
    projection_head_mode = "linear"     
    norm_pix = False                    
```

**Output**:
- Checkpoints: `checkpoints/checkpoint_spatial_pretraining_epoch_*.pt`
- Final weights: `models/Spatial_Only_More_than_Base_PreTrained_4.pth`
- Visualizations: Patch reconstructions and loss curves in `visualizations/`

**Training Features**:
- Real-time visualization of masked patch reconstruction (every 250 steps)
- Loss curve plotting
- Automatic checkpoint saving after each epoch
- Resume training from checkpoint support

### 2. Supervised Fine-tuning

Load pretrained weights and fine-tune on labeled data:

```bash
python fine_tuning.py
```

**Key Configuration**:

```python
class Config:
    batch_size = 64
    epoch = 15                          
    embedding_dropout = 0.2             
    unfreezed_transformer_layers = 4    
    pretrained_model_state_dict = 'Spatial_Only_More_than_Base_PreTrained_4.pth'
    resume_training = True              
    checkpoint_path = "checkpoints/checkpoint_finetuning_epoch_15_last_4.pth"
```

**Transfer Learning Strategy**:
1. Load pretrained weights
2. Add learnable class token to position embeddings
3. Freeze early transformer layers (feature extraction)
4. Unfreeze last N layers for task-specific adaptation
5. Train classification head from scratch

**Output**:
- Checkpoints: `checkpoints/checkpoint_finetuning_epoch_*_last_4.pth`
- Final model: `models/More_than_Base_Finetuned_15_Last_4.pt`
- Evaluation metrics: Classification report, confusion matrix, ROC-AUC

### 3. Inference with Grad-CAM Visualization

Predict and visualize manipulated regions:

```bash
python infer.py
```

**Usage Example**:

```python
# Edit img_path in infer.py
img_path = 'example_images/suspicious_image.jpg'

# Run inference
prediction, confidence, probabilities, cam, output_path = predict_and_save(model, img_path)
```

**Output Visualizations** (saved to `results/`):
- `original_*.png`: Original input image
- `heatmap_overlay_*.png`: Heatmap overlaid on image (red = suspicious)
- `heatmap_only_*.png`: Attention heatmap with colorbar
- `combined_*.png`: Side-by-side comparison of all three

**Grad-CAM Features**:
- Highlights regions the model focuses on for classification
- Helps identify manipulated facial features, edges, or artifacts
- Provides interpretability for model predictions

## Model Components

### PatchAndEmbed

Converts input images into patch embeddings:

```python
self.patches_spatial = torch.nn.Conv2d(
    in_channels=3,
    out_channels=embedding_dim,
    kernel_size=patch_size,
    stride=patch_size,
    padding=0
)

self.position_embed = torch.nn.Parameter(
    torch.randn(1, num_patches, embedding_dim)
)

self.class_embed = torch.nn.Parameter(
    torch.randn(1, 1, embedding_dim)
)
```

### TransformerEncoder

Standard transformer block with residual connections:

```python
def forward(self, x):
    x = self.msa_block(x) + x      
    x = self.mlp_block(x) + x
    return x
```

### MIMHead (Pretraining)

Reconstructs masked patches:

**Linear Mode**:
```python
self.proj = torch.nn.Linear(embedding_dim, patch_size**2 * 3)
```

**Conv Mode** (better for spatial coherence):
```python
self.decoder = torch.nn.Sequential(
    Upsample -> Conv2d -> GELU -> Upsample -> Conv2d
)
```

### Masking Strategy

```python
def random_masking(self, x):
    ids_shuffle = torch.argsort(torch.rand(B, N))

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones([B, N])
    mask[:, :len_keep] = 0
    
    return x_masked, mask, ids_restore
```

### Loss Functions

**Pretraining** (MSE on masked patches):
```python
def compute_loss(self, pred, target, mask):
    if self.norm_pix:
        target_normalized = (target - mean) / std
        loss = (pred - target_normalized) ** 2
    else:
        loss = (pred - target) ** 2
    
    loss = (loss.mean(dim=-1) * mask).sum() / mask.sum()
    return loss
```

**Fine-tuning** (Cross-entropy):
```python
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(predictions, labels)
```

## Training Features

### Mixed Precision Training

Automatic mixed precision for faster training:

```python
scaler = GradScaler()

with autocast(device_type='cuda'):
    outputs = model(images)
    loss = loss_fn(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Checkpoint Management

```python
# to save
checkpointing.save_checkpoint(
    model, optimizer, scaler,
    epoch=5,
    save_dir='checkpoints',
    filename='checkpoint_epoch_5.pth',
    loss=0.123,
    accuracy=0.95
)

# to load
start_epoch, _, model, optimizer, scaler = checkpointing.load_checkpoint(
    model, optimizer, scaler,
    checkpoint_path='checkpoints/checkpoint_epoch_5.pth'
)
```

**Metrics**:
```
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.83      0.85      5492
           1       0.84      0.88      0.86      5413

    accuracy                           0.86     10905
   macro avg       0.86      0.86      0.86     10905
weighted avg       0.86      0.86      0.86     10905

ROC-AUC: 0.9363
Confusion Matrix:
[[4563  929]
 [ 641 4772]]

Test Loss: 0.33330712863924905 | Test Accuracy: 0.8569773390279178
```

## Hyperparameters

### Pretraining

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 120 | Images per batch |
| Epochs | 4 | Training epochs |
| Embedding Dim | 896 | Token dimension |
| Transformer Layers | 12 | Encoder depth |
| Attention Heads | 14 | Multi-head count |
| MLP Size | 3072 | Feedforward hidden dim |
| Mask Ratio | 0.75 | Percentage masked |
| Learning Rate | 0.0001 | AdamW LR |
| Dropout | 0.1 | MLP dropout |

### Fine-tuning

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 64 | Images per batch |
| Epochs | 15 | Training epochs |
| Learning Rate | 0.00001 | Lower LR for fine-tuning |
| Embedding Dropout | 0.1 | Regularization |
| Unfrozen Layers | 4 | Last N layers trainable |
| Optimizer | AdamW | Weight decay optimizer |

## Acknowledgments

- Vision Transformer architecture based on [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Masked Autoencoder pretraining inspired by [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

***if this helped you, give it a ⭐***
---
