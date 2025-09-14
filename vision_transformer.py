import torch

class PatchAndEmbed(torch.nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        
        super().__init__()
        
        num_patches = (224 // patch_size) ** 2
        num_tokens = num_patches * 2 + 1
        
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
        
        class_token = self.class_embed
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
            torch.nn.GELU(p=dropout),
            torch.nn.Linear(in_features=mlp_size,
                            out_features=embedding_dim),
            torch.nn.Dropout(p=dropout)
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
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