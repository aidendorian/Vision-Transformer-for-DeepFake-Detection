import torch
        
class PatchAndEmbed(torch.nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768,
                 batch_size:int=16,
                 img_size=224,
                 phase:str="pretraining"):
        super().__init__()
        
        self.phase = phase
        num_patches = (img_size // patch_size) ** 2
        
        if phase == "pretraining":
            num_tokens = num_patches * 2
        elif phase == "fine-tuning":
            num_tokens = num_patches * 2 + 1
        
        self.batch_size = batch_size
        self.patches_spatial = torch.nn.Conv2d(in_channels=in_channels,
                                               out_channels=embedding_dim,
                                               kernel_size=patch_size,
                                               stride=patch_size,
                                               padding=0)
        
        self.patches_freq = torch.nn.Conv2d(in_channels=in_channels,
                                            out_channels=embedding_dim,
                                            kernel_size=patch_size,
                                            stride=patch_size,
                                            padding=0)
        
        self.class_embed = torch.nn.Parameter(torch.randn(1, 1, embedding_dim),
                                              requires_grad=True)
        
        self.position_embed = torch.nn.Parameter(torch.randn(1, num_tokens, embedding_dim),
                                                 requires_grad=True)
        
        self.flatten = torch.nn.Flatten(start_dim=2,
                                        end_dim=3)
        
    def forward(self, x):
        x_spatial = self.patches_spatial(x)
        
        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq)
        x_freq = torch.abs(x_freq)
        x_freq = x_freq / (x_freq.max() + 1e-8)
        x_freq = self.patches_freq(x_freq)
        
        x_spatial = self.flatten(x_spatial)
        x_freq = self.flatten(x_freq)
        
        x_spatial = x_spatial.permute(0, 2, 1)
        x_freq = x_freq.permute(0, 2, 1)
        
        x_spatial_freq = torch.cat((x_spatial, x_freq), dim=1)
        
        if self.phase == "fine-tuning":
            class_token = self.class_embed.expand(self.batch_size, -1, -1)
            x_tokens = torch.cat((x_spatial_freq, class_token), dim=1)
            position_embed = self.position_embed
            x_tokens = x_tokens + position_embed
            
        elif self.phase == "pretraining":
            position_embed = self.position_embed
            x_tokens = x_spatial_freq + position_embed
        
        else:
            raise ValueError("phase can only be 'fine_tuning' or 'pretraining'")        
        
        return x_tokens
    
class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0):
        super().__init__()
        
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=embedding_dim,
                                                          num_heads=num_heads,
                                                          dropout=attn_dropout,
                                                          batch_first=True)
        
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,
                                          key=x,
                                          value=x,
                                          need_weights=False)
        return attn_output
    
class MLPBlock(torch.nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 dropout:float=0.1):
        super().__init__()
        
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=embedding_dim,
                            out_features=mlp_size),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=mlp_size,
                            out_features=embedding_dim),
            torch.nn.Dropout(p=dropout)
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
class MIMHead(torch.nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 patch_size:int=16,
                 in_channels:int=3,
                 mode:str="linear"):
        super().__init__()
        
        self.mode = mode
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.target_dim = in_channels * patch_size**2
        
        if mode == "linear":
            self.proj = torch.nn.Linear(in_features=embedding_dim, out_features=self.target_dim)
            
        elif mode == "conv":
            hidden = embedding_dim//2
            self.token_to_feat = torch.nn.Linear(in_features=embedding_dim, out_features=hidden*(patch_size//4)**2)
            self.decoder = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                torch.nn.Conv2d(in_channels=hidden, out_channels=hidden//2, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                torch.nn.Conv2d(in_channels=hidden//2, out_channels=in_channels, kernel_size=3, padding=1)
            )
            
        else:
            raise ValueError("mode can only be either 'linear' or 'conv'")
        
    def forward(self, x):
        B, N, D = x.shape
        
        if self.mode == "linear":
            return self.proj(x)
        
        elif self.mode == "conv":
            feat = self.token_to_feat(x)
            s = self.patch_size//4
            hidden = feat.shape[-1]//(s*s)
            feat = feat.view(B*N, hidden, s, s)
            out = self.decoder(feat)
            return out.view(B, N, -1)

class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0,
                 num_heads:int=12,
                 ):
        super().__init__()
        
        self.msa_block = MultiheadSelfAttention(embedding_dim=embedding_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout)
        
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x
    
class ViT(torch.nn.Module):
    def __init__(self,
                 batch_size:int=16,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_transformer_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 phase:str="pretraining",
                 projection_head_mode:str="linear",
                 mask_ratio:float=0.75,
                 num_classes:int=2):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.phase = phase
        
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, embedding_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        self.patch_and_embed = PatchAndEmbed(in_channels=in_channels,
                                             patch_size=patch_size,
                                             embedding_dim=embedding_dim,
                                             batch_size=batch_size,
                                             img_size=img_size,
                                             phase=phase)
        
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        
        self.transformer_encoder = torch.nn.Sequential(
            *[TransformerEncoder(embedding_dim=embedding_dim,
                                 mlp_size=mlp_size,
                                 mlp_dropout=mlp_dropout,
                                 attn_dropout=attn_dropout,
                                 num_heads=num_heads) for _ in range(num_transformer_layers)]
        )
        
        self.projection_head = MIMHead(embedding_dim=embedding_dim,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       mode=projection_head_mode)
        
        self.classifier_head = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )
        
    def extract_patches(self, x):
        spatial = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq)
        x_freq = torch.abs(x_freq)
        x_freq = x_freq / (x_freq.max() + 1e-8)
        freq = torch.nn.functional.unfold(x_freq, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        return torch.cat([spatial, freq], dim=1)
    
    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N* (1-self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    
    def restore_mask(self, x_encoded, mask_token, ids_restore):
        B, N, D = ids_restore.shape[0], ids_restore.shape[1], x_encoded.shape[2]
        len_keep = x_encoded.shape[1]
        
        mask_tokens = mask_token.expand(B, N-len_keep, -1)
        x_combined = torch.cat([x_encoded, mask_tokens], dim=1)
        x_full = torch.gather(x_combined, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        return x_full
    
    def compute_loss(self, pred, target, mask):
        loss = (pred-target)**2
        loss = loss.mean(dim=-1)
        loss = (loss*mask).sum()/mask.sum()
        return loss
        
    def forward(self, x):
        patches = self.extract_patches(x)
        x = self.patch_and_embed(x)
        x = self.embedding_dropout(x)
        
        if self.phase == "pretraining":
            x_masked, mask, ids_restore = self.random_masking(x)
            x_encoded = self.transformer_encoder(x_masked)
            x_full = self.restore_mask(x_encoded, self.mask_token, ids_restore)
            pred = self.projection_head(x_full)
            loss = self.compute_loss(pred, patches, mask)
            return pred, loss
        
        elif self.phase == "fine-tuning":
            x = self.transformer_encoder(x)
            class_token = x[:, -1]
            logits = self.classifier_head(class_token)
            return logits
        
        raise ValueError("wrong value for phase")