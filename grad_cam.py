import torch
import torch.nn.functional as F
import math

class ViTGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()        
        output = self.model(input_tensor.unsqueeze(0))
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            print("WARNING: No gradients or activations captured!")
            return None
        
        B, N, D = self.activations.shape
        gradients = self.gradients.mean(dim=1, keepdim=True)
        
        weights = gradients.mean(dim=1, keepdim=True)
        cam = torch.matmul(self.activations, weights.transpose(1, 2))
        cam = F.relu(cam.squeeze(-1))
        
        num_patches = N - 1
        H_patches = W_patches = int(math.sqrt(num_patches))
        
        cam_patches = cam[:, 1:1+num_patches]
        
        cam_spatial = cam_patches.reshape(B, H_patches, W_patches)
        
        cam_upsampled = F.interpolate(cam_spatial.unsqueeze(1), 
                                    size=input_tensor.shape[1:], 
                                    mode='bilinear', 
                                    align_corners=False)
        
        cam_final = cam_upsampled.squeeze()
        cam_final = cam_final - cam_final.min()
        cam_final = cam_final / (cam_final.max() + 1e-8)
        
        return cam_final.cpu().numpy()