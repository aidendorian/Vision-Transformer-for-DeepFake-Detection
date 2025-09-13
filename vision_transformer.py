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
    
