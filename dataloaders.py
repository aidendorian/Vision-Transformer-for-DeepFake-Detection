import torch
import os
from PIL import Image
import torchvision

class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
                
        self.transform = transform
        self.samples = [os.path.join("data/pretraining", f)
                        for f in os.listdir("data/pretraining")]
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
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
    torchvision.transforms.ToTensor()
])

pretrain_dataset = DeepfakeDataset(transform=pretrain_transform)

def get_dataloader(phase="pretraining",
                   batch_size=16,
                   num_workers=4,
                   pin_memory=True,
                   prefetch_factor=2,
                   persistent_workers=True):
    
    if phase == "pretraining":
        
        return torch.utils.data.DataLoader(pretrain_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           persistent_workers=persistent_workers,
                                           prefetch_factor=prefetch_factor)
    elif phase == "finetuning":

        train_dataset = torchvision.datasets.ImageFolder("data/finetuning/Train",
                                                         transform=finetune_transform)
        
        val_dataset = torchvision.datasets.ImageFolder("data/finetuning/Validation",
                                                       transform=finetune_transform)
        
        test_dataset = torchvision.datasets.ImageFolder("data/finetuning/Test",
                                                        transform=finetune_transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory,
                                                    persistent_workers=persistent_workers,
                                                    prefetch_factor=prefetch_factor,
                                                    shuffle=True)
        
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory,
                                                 persistent_workers=persistent_workers,
                                                 prefetch_factor=prefetch_factor,
                                                 shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  persistent_workers=persistent_workers,
                                                  prefetch_factor=prefetch_factor,
                                                  shuffle=True)
        
        return train_loader, val_loader, test_loader