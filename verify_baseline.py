import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
from tqdm import tqdm
import sys

# --- 配置 ---
NUM_CLASSES = 100
EPOCHS = 20
BASE_LR = 1e-4  # 基础学习率，会根据GPU数量自动缩放
BATCH_SIZE_PER_GPU = 128

# 数据集路径 (请确保与您之前的配置一致)
TRAIN_PATH = "/root/autodl-tmp/imagenet100"
VAL_PATH = "/root/autodl-tmp/imagenet100_val"

def setup_ddp():
    """初始化DDP环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        return rank, local_rank, world_size
    else:
        print("未检测到DDP环境变量，将在单卡模式下运行...")
        # 默认单卡配置
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def validate(model, loader, device_id, world_size):
    """分布式验证函数"""
    model.eval()
    correct = torch.tensor(0.0, device=device_id)
    total = torch.tensor(0.0, device=device_id)
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            correct += (preds == labels).sum()
            total += labels.size(0)
    
    # 在多卡模式下汇总结果
    if world_size > 1:
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        
    return (100.0 * correct / total).item()

def main():
    # 1. 环境初始化
    rank, local_rank, world_size = setup_ddp()
    is_master = (rank == 0)
    
    if is_master:
        print(f"--- 开始 Baseline 训练验证 ---")
        print(f"World Size: {world_size}")
        print(f"Dataset: {TRAIN_PATH}")
        print(f"Epochs: {EPOCHS}")

    # 2. 数据准备
    # 使用标准的 ImageNet 预处理
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(TRAIN_PATH):
        if is_master: print(f"错误: 训练集路径不存在: {TRAIN_PATH}")
        return

    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform_train)
    val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform_val)

    # DDP Sampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE_PER_GPU, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE_PER_GPU, 
        shuffle=False, 
        sampler=val_sampler, 
        num_workers=4, 
        pin_memory=True
    )

    # 3. 模型构建 (使用标准 timm 接口，确保无剪枝副作用)
    # 使用 pretrained=True 加载 ImageNet-1k 权重，然后修改 Head 为 100 类
    model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
    model.to(local_rank)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 4. 优化器与调度器
    # 根据 GPU 数量自动缩放学习率
    actual_lr = BASE_LR * world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if is_master:
        print(f"初始化完成，开始训练... (LR: {actual_lr})")

    # 5. 训练循环
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_loss = torch.tensor(0.0, device=local_rank)
        
        # 仅在主进程显示进度条
        if is_master:
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
            iterator = pbar
        else:
            iterator = train_loader
            
        for images, labels in iterator:
            images = images.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if is_master and isinstance(iterator, tqdm):
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # 验证阶段
        val_acc = validate(model, val_loader, local_rank, world_size)
        
        if is_master:
            print(f"Epoch {epoch+1} 完成. Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = "baseline_best_100class.pth"
                # 保存时注意去掉 DDP 的 'module.' 前缀
                state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
                torch.save(state_dict, save_path)
                print(f"*** 发现新最佳模型，已保存到 {save_path} ***")

    if is_master:
        print(f"训练结束。最佳验证集准确率: {best_acc:.2f}%")

    cleanup_ddp()

if __name__ == "__main__":
    main()