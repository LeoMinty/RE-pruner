# verify_baseline_cifar.py
import torch
import torch.nn as nn
import timm
import argparse
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils_cifar import get_cifar10_loaders

# --- 配置 ---
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='Training epochs') # 验证50轮足够看趋势
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

# --- 3. 优化器 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

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
if best_acc < 95.0:
    print(">> 警告：Baseline 远低于 98%。")
    print(">> 原因：缺少 Mixup/Cutmix 数据增强。")
    print(">> 建议：比较剪枝模型精度时，请以这个 Baseline ({:.2f}%) 为准，而不是论文的 98%。".format(best_acc))
else:
    print(">> Baseline 正常。剪枝模型精度低可能是因为剪枝策略问题。")