import torch
        
class PatchAndEmbed(torch.nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768,
                 batch_size:int=16,
                 img_size=224):
        
        super().__init__()
        
        num_patches = (img_size // patch_size) ** 2
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
        
        self.class_embed = torch.nn.Parameter(torch.zeros(1, 1, embedding_dim),
                                              requires_grad=True)
        
        self.position_embed = torch.nn.Parameter(torch.zeros(1, num_tokens, embedding_dim),
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
        
        class_token = self.class_embed.expand(self.batch_size, -1, -1)
        x_tokens = torch.cat((x_spatial_freq, class_token), dim=1)
        
        position_embed = self.position_embed
        x_tokens = x_tokens + position_embed
        
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
        attn_output = self.multihead_attn(query=x,
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
                torch.nn.Conv2d(out_channels=hidden//2, out_channels=in_channels, kernel_size=3, padding=1)
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
                 num_classes:int=2):
        super().__init__()
        
        self.patch_and_embed = PatchAndEmbed(in_channels=in_channels,
                                             patch_size=patch_size,
                                             embedding_dim=embedding_dim,
                                             batch_size=batch_size,
                                             img_size=img_size)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        
        self.transformer_encoder = torch.nn.Sequential(
            *[TransformerEncoder(embedding_dim=embedding_dim,
                                 mlp_size=mlp_size,
                                 mlp_dropout=mlp_dropout,
                                 attn_dropout=attn_dropout,
                                 num_heads=num_heads) for _ in range(num_transformer_layers)]
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=embedding_dim),
            torch.nn.Linear(in_features=embedding_dim,
                            out_features=num_classes)
        )
        
        self.projection_head = MIMHead(embedding_dim=embedding_dim,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       mode="linear")
        
    def forward(self, x):
        x = self.patch_and_embed(x)
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.projection_head(x)
        return x