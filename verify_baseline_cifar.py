# verify_baseline_cifar.py
import torch
import torch.nn as nn
import timm
import argparse
import os
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils_cifar import get_cifar10_loaders

# --- 配置 ---
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
args = parser.parse_args()

NUM_CLASSES = 10
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 加载全量模型 (ImageNet Pretrained) ---
print(f"正在加载全量 DeiT-Small (ImageNet Pretrained)...")
# 这里的模型没有经过任何修改，是标准的 timm 模型
model = timm.create_model('deit_small_patch16_224', pretrained=True)
# 重置分类头
model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
model.to(DEVICE)

# --- 2. 准备数据 ---
# 注意：这里使用的是你目前的 utils_cifar，没有 Mixup/Cutmix
train_loader, val_loader = get_cifar10_loaders(BATCH_SIZE)


# 这些参数是 DeiT/ViT 在 CIFAR 上常用的推荐值
mixup_fn = Mixup(
    mixup_alpha=0.8,       # mixup 参数
    cutmix_alpha=1.0,      # cutmix 参数
    cutmix_minmax=None,
    prob=1.0,              # 启用概率
    switch_prob=0.5,       # mixup 和 cutmix 切换概率
    mode='batch',          #在这个batch中应用
    label_smoothing=0.1,   # 标签平滑
    num_classes=NUM_CLASSES
)

# --- 3. 优化器 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
if mixup_fn is not None:
    criterion = SoftTargetCrossEntropy()
else:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# --- 4. 验证函数 ---
@torch.no_grad()
def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return 100. * correct / total

# --- 5. 训练循环 ---
print(f"--- 开始训练全量 Baseline (LR: {args.lr}) ---")
best_acc = 0.0

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    
    # 验证
    acc = validate(model, val_loader)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "baseline_cifar10_deit_small.pth")
    
    print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

print(f"\n结论：")
print(f"当前训练配方下的全量模型上限 (Baseline) 为: {best_acc:.2f}%")