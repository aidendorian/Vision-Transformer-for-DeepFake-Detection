import torch
import os
from PIL import Image
import torchvision

class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, phase="pretraining", transform=None):
        super().__init__()
        
        self.phase = phase
        self.transform = transform
        
        if phase == "pretraining":
            self.samples = [os.path.join("data/pretraining", f)
                            for f in os.listdir("data/pretraining")]

        elif phase == "finetuning":
            real_dir = os.path.join("data/finetuning", "real")
            fake_dir = os.path.join("data/finetuning", "fake")
            self.samples = []
            
            self.samples += [(os.path.join(real_dir, f), 0) 
                             for f in os.listdir(real_dir)]

            self.samples += [(os.path.join(fake_dir, f), 1) 
                             for f in os.listdir(fake_dir)]
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if self.phase == "finetuning":
            img_path, label = self.samples[index]
            image = Image.open(img_path).convert("RGB")
        
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        elif self.phase == "pretraining":
            img_path = self.samples[index]
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image
            
    
pretrain_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    torchvision.transforms.ToTensor()
])

finetune_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms. ToTensor()
])

pretrain_dataset = DeepfakeDataset(transform=pretrain_transform,
                                   phase="pretraining")

finetune_dataset = DeepfakeDataset(transform=finetune_transform,
                                   phase="finetuning")

def get_dataloader(phase="pretraining",
                   batch_size=16,
                   num_workers=4):
    if phase == "pretraining":
        return torch.utils.data.DataLoader(pretrain_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers)
    elif phase == "finetuning":
        return torch.utils.data.DataLoader(finetune_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers)