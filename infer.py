import grad_cam
import torch
import torchvision
import PIL
import matplotlib.pyplot as plt
import numpy as np
import os
import vision_transformer
from datetime import datetime
from torch.amp import autocast
import checkpointing

torch.manual_seed(42)

def preprocess_img(img_path, img_size=224):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    img = PIL.Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    
    return input_tensor, img

def save_manipulated_regions(model, image_tensor, label, original_image, output_dir="results"):
    """Save visualization as images instead of showing"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    target_layer = model.transformer_encoder[-1].mlp_block.mlp[0]
    gradcam = grad_cam.ViTGradCAM(model, target_layer)
    cam = gradcam.generate_cam(image_tensor.squeeze(0), target_class=label)
    
    def denormalize(tensor):
        tensor = tensor * 0.5 + 0.5
        return tensor.cpu().permute(1, 2, 0).numpy()
    
    class_names = {0: 'FAKE', 1: 'REAL'}
    prediction_text = class_names[label]
    
    plt.figure(figsize=(8, 6))
    if isinstance(original_image, PIL.Image.Image):
        plt.imshow(original_image)
    else:
        plt.imshow(denormalize(image_tensor.squeeze(0)))
    plt.title(f'Original Image\nPredicted: {prediction_text}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/original_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    if isinstance(original_image, PIL.Image.Image):
        display_img = np.array(original_image.resize((224, 224)))
        plt.imshow(display_img)
    else:
        plt.imshow(denormalize(image_tensor.squeeze(0)))
    plt.imshow(cam, alpha=0.5, cmap='jet')
    plt.title(f'Manipulation Heatmap\nPredicted: {prediction_text}\n(Red = Suspicious Regions)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_overlay_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cam, cmap='jet')
    plt.title('Heatmap Only')
    plt.axis('off')
    plt.colorbar(label='Attention Intensity')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_only_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if isinstance(original_image, PIL.Image.Image):
        axes[0].imshow(original_image)
    else:
        axes[0].imshow(denormalize(image_tensor.squeeze(0)))
    axes[0].set_title(f'Original\n{prediction_text}')
    axes[0].axis('off')
    
    if isinstance(original_image, PIL.Image.Image):
        display_img = np.array(original_image.resize((224, 224)))
        axes[1].imshow(display_img)
    else:
        axes[1].imshow(denormalize(image_tensor.squeeze(0)))
    axes[1].imshow(cam, alpha=0.5, cmap='jet')
    axes[1].set_title('Manipulation Heatmap')
    axes[1].axis('off')
    
    im = axes[2].imshow(cam, cmap='jet')
    axes[2].set_title('Attention Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to '{output_dir}/' directory:")
    print(f"  - original_{timestamp}.png")
    print(f"  - heatmap_overlay_{timestamp}.png") 
    print(f"  - heatmap_only_{timestamp}.png")
    print(f"  - combined_{timestamp}.png")
    
    return cam, f'{output_dir}/combined_{timestamp}.png'

def predict_and_save(model, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    input_tensor, original_image = preprocess_img(image_path)
    input_tensor = input_tensor.to(device)
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        with autocast(device_type='cuda'):
            outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    class_names = {0: 'FAKE', 1: 'REAL'}
    prediction = class_names[predicted_class]
    
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.3f}")
    print(f"FAKE probability: {probabilities[0, 0]:.3f}")
    print(f"REAL probability: {probabilities[0, 1]:.3f}")
    
    cam, output_path = save_manipulated_regions(model, input_tensor, predicted_class, original_image)
    
    return prediction, confidence, probabilities, cam, output_path

img_path = 'example_images/real.jpg'

model = vision_transformer.ViT(
    embedding_dim=896,
    num_heads=14,
    phase='finetuning',
    num_classes=2
)

state_dict = torch.load('models/More_than_Base_Finetuned_24_Last_4.pt')
model.load_state_dict(state_dict)

model.eval()

prediction, confidence, probabilities, cam, output_path = predict_and_save(model, img_path)