import torch

class PatchAndProcess(torch.nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768): # 768, assuming image would be of size 224 x 224.
        super().__init__()
        
        self.patches = torch.nn.Conv2d(in_channels=in_channels,
                                       out_channels=embedding_dim,
                                       kernel_size=patch_size,
                                       stride=patch_size,
                                       padding=0)
        
        self.flatten = torch.nn.Flatten(start_dim=2,
                                        end_dim=3)
        
    def forward(self, x):
        x_patched = self.patches(x)
        x_ft = torch.fft.fft2(x_patched)
        x_patched_ft = torch.cat((x_patched, x_ft), dim=-1)
        x_flat = self.flatten(x_patched_ft)
        return x_flat.permute(0, 2, 1)