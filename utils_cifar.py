# utils_cifar.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_transforms():
    # CIFAR-10 均值和方差
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # 训练集变换：DeiT 需要 224x224 输入
    transform_train = transforms.Compose([
        transforms.Resize(224), 
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 验证集变换
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return transform_train, transform_val

def get_cifar10_datasets(root='./data_cifar'):
    """返回 Dataset 对象，用于需要直接访问 dataset 的场景 (如 SCFP 计算)"""
    transform_train, transform_val = get_cifar10_transforms()
    
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_val)
    
    return train_dataset, val_dataset

def get_cifar10_loaders(batch_size, root='./data_cifar', num_workers=4):
    """返回 DataLoader"""
    train_dataset, val_dataset = get_cifar10_datasets(root)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader