import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# 关键：从你修改过的本地文件导入模型
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention # 导入用于类型检查

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 100 
BATCH_SIZE = 64
FINETUNE_EPOCHS = 50 # 微调的Epochs数量 (论文中DeiT是80, 您可以先设50)
FINETUNE_LR = 1e-5 # 微调时使用非常小的学习率

# !!! 必需：指向您第二阶段(main_phase2.py)输出的模型文件 !!!
# (请确保这是您正式训练后得到的文件)
PHASE2_MODEL_PATH = "re_pruner_phase2_pruned_formal_r_logit_100class_r0.5.pth" 

# !!! 必需：您的100类ImageNet子集路径 !!!
IMAGENET_SUBSET_TRAIN_PATH = "/path/to/your/100_class_imagenet/train"
IMAGENET_SUBSET_VAL_PATH = "/path/to/your/100_class_imagenet/val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 准备数据集 (训练集 和 验证集) ---
print("正在加载数据集 (用于微调)...")
if not os.path.exists(IMAGENET_SUBSET_TRAIN_PATH) or not os.path.exists(IMAGENET_SUBSET_VAL_PATH):
    print(f"错误: 路径 '{IMAGENET_SUBSET_TRAIN_PATH}' 或 '{IMAGENET_SUBSET_VAL_PATH}' 不存在。请修改 finetune.py 中的路径。")
    exit()

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = datasets.ImageFolder(IMAGENET_SUBSET_TRAIN_PATH, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_dataset = datasets.ImageFolder(IMAGENET_SUBSET_VAL_PATH, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"训练集: {len(train_dataset)} 图像, 验证集: {len(val_dataset)} 图像。")

# --- 3. 加载第二阶段剪枝模型 ---
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
print(f"正在从 {PHASE2_MODEL_PATH} 加载剪枝模型...")
if not os.path.exists(PHASE2_MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {PHASE2_MODEL_PATH} 不存在。请检查第二阶段是否已成功运行。")
model.load_state_dict(torch.load(PHASE2_MODEL_PATH, map_location=device), strict=False)
model.to(device)
print("加载成功！")

# --- 关键：激活剪枝模式 ---
# 这将确保 forward pass 使用 differentiable_pruning_operation (topk)
# 从而只使用 "is_important" 的掩码元素
for module in model.modules():
    if isinstance(module, MaskedAttention):
        module.is_pruning_phase = True

# --- 4. 设置优化器和损失函数 ---
# 优化所有 *未被冻结* 的参数 (r_logit 和 theta 仍会被优化，但无害)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FINETUNE_LR)
criterion = nn.CrossEntropyLoss()

# --- 5. 验证函数 (计算Top-1和Top-5准确率) ---
@torch.no_grad() # 确保不计算梯度
def validate(model, loader, criterion, device):
    model.eval() # 切换到评估模式
    total_loss = 0
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    pbar = tqdm(loader, desc="验证中")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # --- 关键：不再传入 y_labels ---
        outputs = model(images, y_labels=None)
        
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        # --- Top-k 计算 ---
        _, pred = outputs.topk(5, 1, True, True) # 获取Top 5预测
        pred = pred.t() # 转置
        correct = pred.eq(labels.view(1, -1).expand_as(pred)) # 比较

        correct_1 += correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_5 += correct[:5].reshape(-1).float().sum(0, keepdim=True)
        total += labels.size(0)
        
        pbar.set_postfix({
            "Loss": loss.item(), 
            "Top-1": (100 * correct_1.item() / total),
            "Top-5": (100 * correct_5.item() / total)
        })

    avg_loss = total_loss / len(loader)
    top1_acc = 100 * correct_1.item() / total
    top5_acc = 100 * correct_5.item() / total
    return avg_loss, top1_acc, top5_acc

# --- 6. 微调训练循环 ---
print("--- 开始第三阶段 (Finetune) ---")
best_acc1 = 0.0
best_acc5 = 0.0

for epoch in range(FINETUNE_EPOCHS):
    model.train() # 切换到训练模式
    
    # 确保剪枝模式始终开启
    for module in model.modules():
        if isinstance(module, MaskedAttention):
            module.is_pruning_phase = True
            
    pbar = tqdm(train_loader, desc=f"微调 Epoch {epoch+1}/{FINETUNE_EPOCHS}")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # --- 关键：不再传入 y_labels ---
        outputs = model(images, y_labels=None)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            pbar.set_postfix({"Loss": loss.item()})

    # --- 每个 Epoch 结束后进行验证 ---
    val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1} 验证完成: Avg Loss: {val_loss:.4f}, Top-1 Acc: {val_acc1:.2f}%, Top-5 Acc: {val_acc5:.2f}%")
    
    if val_acc1 > best_acc1:
        best_acc1 = val_acc1
        best_acc5 = val_acc5 # 同时保存
        # 保存性能最好的模型
        output_filename = f"re_pruner_finetuned_best_{NUM_CLASSES}class.pth"
        torch.save(model.state_dict(), output_filename)
        print(f"*** 新的最佳Top-1准确率！模型已保存到 {output_filename} ***")

print("微调完成！")
print(f"--- 最终最佳 Top-1 准确率: {best_acc1:.2f}% ---")
print(f"--- 最终最佳 Top-5 准确率: {best_acc5:.2f}% ---")