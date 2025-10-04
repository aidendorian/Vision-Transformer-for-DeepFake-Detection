# Vision Transformer for DeepFake Detection

A PyTorch implementation of Vision Transformer (ViT) for detecting deepfake images using masked image modeling (MIM) for pretraining followed by supervised fine-tuning.

## Overview

This project implements a two-stage approach to deepfake detection:
1. **Pretraining**: Self-supervised learning using Masked Image Modeling (MIM) where the model learns to reconstruct randomly masked image patches
2. **Fine-tuning**: Supervised classification to distinguish between real and fake images

## Architecture

The Vision Transformer architecture includes:
- **Patch Embedding**: Converts images into sequence of patch embeddings (16×16 patches)
- **Transformer Encoder**: 12 layers with multi-head self-attention (12 heads) and MLP blocks
- **Pretraining Head**: Reconstructs masked patches using linear or convolutional projection
- **Classification Head**: Binary classifier for real vs fake detection

### Key Features
- Masked autoencoding with 75% masking ratio
- Position embeddings for spatial information
- Class token for classification during fine-tuning
- Mixed precision training support
- Comprehensive visualization tools

## Project Structure

```
└── 📁Deepfakes_ViT
    ├── 📁checkpoints          # Model checkpoints
    ├── 📁data
    │   ├── 📁finetuning
    │   │   ├── 📁fake        # Fake images for training
    │   │   └── 📁real        # Real images for training
    │   └── 📁pretraining     # Unlabeled images for 
    ├── 📁models              # Saved model weights
    ├── 📁visualizations      # Training visualizations and 
    ├── checkpointing.py      # Checkpoint saving/loading 
    ├── dataloaders.py        # Dataset and dataloader 
    ├── fine_tuning.py        # Fine-tuning script
    ├── pre_training.py       # Pretraining script
    ├── vision_transformer.py # ViT model architecture
    └── visualization.py      # Visualization utilities
```

## Requirements

```bash
torch>=2.0.0
torchvision
PIL
matplotlib
numpy
```

## Dataset Preparation

### Pretraining Data
Place unlabeled images in `data/pretraining/`. These images are used for self-supervised learning through masked image modeling.

### Fine-tuning Data
Organize labeled images as follows:
```
data/finetuning/
├── real/  # Real/authentic images (label: 0)
└── fake/  # Deepfake/synthetic images (label: 1)
```

## Usage

### 1. Pretraining

Train the model using masked image modeling:

```python
python pre_training.py
```

**Configuration** (in `pre_training.py`):
- `batch_size`: 192
- `epoch`: 20
- `img_size`: 224
- `patch_size`: 16
- `mask_ratio`: 0.75 (75% of patches masked)
- `embedding_dim`: 768
- `num_transformer_layers`: 12
- `num_heads`: 12

The script will:
- Generate patch reconstruction visualizations every 250 steps
- Plot loss curves during training
- Save checkpoints and final model weights to `models/PreTrained_20_False.pth`

### 2. Fine-tuning

After pretraining, fine-tune the model for deepfake detection:

```python
python fine_tuning.py
```

Load pretrained weights and train the classifier head on labeled data.

## Model Components

### PatchAndEmbed
Converts input images into patch embeddings using convolutional layers. Supports both pretraining (no class token) and fine-tuning (with class token) modes.

### TransformerEncoder
Standard transformer encoder block with:
- Multi-head self-attention with layer normalization
- MLP block with GELU activation and dropout
- Residual connections

### MIMHead
Projection head for reconstructing masked patches:
- **Linear mode**: Direct linear projection to patch pixels
- **Conv mode**: Upsampling decoder with convolutional layers

### Masking Strategy
- Random masking of 75% of patches during pretraining
- Mask tokens are learnable parameters
- Shuffle and restore mechanism maintains spatial correspondence

## Training Features

- **Mixed Precision Training**: Uses `torch.amp` for faster training and reduced memory usage
- **Gradient Scaling**: Prevents underflow in fp16 training
- **Data Augmentation**: Random crops, horizontal flips, color jittering
- **Checkpointing**: Regular model and optimizer state saves
- **Visualization**: Real-time monitoring of reconstructions and loss curves

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 224×224 | Input image resolution |
| Patch Size | 16×16 | Size of each image patch |
| Embedding Dim | 768 | Token embedding dimension |
| MLP Size | 3072 | Hidden dimension in MLP blocks |
| Num Heads | 12 | Number of attention heads |
| Num Layers | 12 | Number of transformer layers |
| Mask Ratio | 0.75 | Percentage of patches to mask |
| Batch Size | 192 | Training batch size |
| Learning Rate | 0.0001 | AdamW optimizer learning rate |

## Outputs

### Checkpoints
- Periodic checkpoints saved in `checkpoints/`
- Final pretrained model: `models/`

### Visualizations
- Patch reconstruction comparisons (original vs reconstructed)
- Training loss curves
- Saved in `visualizations/` directory

## Implementation Details

### Patch Extraction
Images are divided into non-overlapping 16×16 patches, flattened and linearly embedded into 768-dimensional vectors.

### Position Encoding
Learnable position embeddings are added to patch embeddings to retain spatial information.

### Loss Computation
Mean squared error (MSE) between predicted and actual patch values, computed only on masked patches.

### Phase Modes
- **Pretraining**: Uses masked autoencoding objective, no class token
- **Fine-tuning**: Adds class token, uses classification head