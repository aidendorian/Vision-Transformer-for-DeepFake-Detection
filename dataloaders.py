import torch
import torchvision
            
pretrain_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

finetune_transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def get_dataloader(phase="pretraining",
                   batch_size=16,
                   num_workers=4,
                   pin_memory=True,
                   prefetch_factor=2,
                   persistent_workers=True):
    
    if phase == "pretraining":
        
        pretrain_dataset = torchvision.datasets.CelebA(root='/home/oslyris/Deepfakes_ViT/data/pretraining',
                                                       split='train',
                                                       transform=pretrain_transform,
                                                       download=False)
        
        return torch.utils.data.DataLoader(pretrain_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory,
                                           persistent_workers=persistent_workers,
                                           prefetch_factor=prefetch_factor,
                                           shuffle=True)
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