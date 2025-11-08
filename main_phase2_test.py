# main_phase2.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.data import Subset
import numpy as np
import os
# 关键：从你修改过的本地文件导入模型
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention # 导入用于类型检查

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 100
BATCH_SIZE = 64
EPOCHS = 10 # 减少epochs用于测试
ALPHA_TARGET = 0.5 # 目标总剪枝率

# 模型状态文件路径
MODEL_STATE_PATH = "re_pruner_phase1_masks_100class.pth"

# --- 2. 准备数据集 (与第一阶段相同) ---
IMAGENET_SUBSET_TRAIN_PATH = "/root/autodl-tmp/imagenet100"
IMAGENET_SUBSET_VAL_PATH = "/root/autodl-tmp/imagenet100_val" # 验证集路径


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载训练集
train_dataset = datasets.ImageFolder(IMAGENET_SUBSET_TRAIN_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# 首先，加载完整的训练集
full_train_dataset = datasets.ImageFolder(IMAGENET_SUBSET_TRAIN_PATH, transform=transform)

# --- 新增：创建数据集的子集 ---
subset_percentage = 0.1 # 使用10%的数据进行快速调试
num_train = len(full_train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices) # 打乱索引
split = int(np.floor(subset_percentage * num_train))
subset_indices = indices[:split]

# 使用Subset创建子集
train_subset = Subset(full_train_dataset, subset_indices)
print(f"使用 {len(train_subset)} / {num_train} 个样本进行快速调试...")

# 在DataLoader中使用子集
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 加载模型并切换到剪枝模式 ---
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
print(f"正在从 {MODEL_STATE_PATH} 加载模型状态...")
if not os.path.exists(MODEL_STATE_PATH):
    raise FileNotFoundError(f"模型文件 {MODEL_STATE_PATH} 不存在。请检查第一阶段是否已成功运行。")
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=device), strict=False)
model.to(device)
print("加载成功！")

print("正在强制重新初始化剪枝参数 (theta)...")
with torch.no_grad(): 
    for module in model.modules():
        if isinstance(module, MaskedAttention):
            # 将 theta 初始化为 0.0
            # 初始时，约一半分数 > 0, 一半 < 0, R 约为 0.5
            # 如果掩码分数大多为正，theta=0.0 会导致 R 接近 0
            # 我们可以根据第一阶段掩码的均值来设置初始theta，但 0.0 是一个合理的起点
            module.theta.data = torch.tensor([0.0], device=device) 
print("剪枝参数初始化完毕。")

# 关键：激活所有MaskedAttention模块的剪枝模式
num_prunable_elements = 0
for module in model.modules():
    if isinstance(module, MaskedAttention):
        module.is_pruning_phase = True
        num_prunable_elements += module.explainability_mask.numel()
print(f"总可剪枝参数元素 (来自掩码): {num_prunable_elements}")

# --- 4. 设置损失函数和优化器 ---
ce_loss_fn = nn.CrossEntropyLoss()


def calculate_pruning_loss_simple(model, alpha_target, total_prunable_elements):
    """计算一个简单的、稳定的二次惩罚剪枝损失"""
    n = 10.0
    current_pruned_elements = torch.tensor(0.0, device=device)
    total_elements = 0.0
    for module in model.modules():
        if isinstance(module, MaskedAttention):
            # 注意：这里我们使用 .data 来获取掩码分数，
            # 因为我们不希望在计算损失时跟踪掩码本身的梯度
            # 掩码在第一阶段已经训练完毕
            mask_scores = module.explainability_mask.data
            theta = module.theta # theta 是可学习的
            
            # (1 - 门控因子) ≈ 剪枝概率
            # 门控因子 = 0.5 * (tanh(n * (M - theta)) + 1)
            # 剪枝概率 ≈ 1.0 - 0.5 * (tanh(n * (M - theta)) + 1)
            #           = 0.5 * (1.0 - tanh(n * (M - theta)))
            # 这是可微分的
            pruned_probability = 0.5 * (1.0 - torch.tanh(n * (mask_scores - theta)))
            
            # 累加预期被剪掉的元素数量
            current_pruned_elements += pruned_probability.sum()
            total_elements += mask_scores.numel()

    if total_elements == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # R 是 (预期剪枝元素 / 总元素)
    current_R_avg = current_pruned_elements / total_elements 
    
    # 损失是 R 与目标的L2距离
    loss_r = (current_R_avg - alpha_target)**2
    return loss_r, current_R_avg

# --- 关键：为不同参数组设置不同的优化器和学习率 ---
# a. 冻结第一阶段学到的掩码分数
pruning_params = []
model_weights = []
for name, param in model.named_parameters():
    if "explainability_mask" in name:
        param.requires_grad = False
    elif "theta" in name:
        pruning_params.append(param)
    else:
        model_weights.append(param)

        
optimizer_weights = torch.optim.AdamW(model_weights, lr=5e-4)
optimizer_pruning = torch.optim.AdamW(pruning_params, lr=0.02)

print(f"模型权重参数组大小: {len(model_weights)}")
print(f"剪枝参数组大小: {len(pruning_params)}")
if not pruning_params:
    print("警告：未找到名为 'theta' 的剪枝参数。请检查 'vision_transformer_modified.py'。")

# --- 5. 第二阶段训练循环 ---
model.train()
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 梯度清零
        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()
        
        # 前向传播
        outputs = model(images, y_labels=labels)
        
        # 计算损失
        loss_ce = ce_loss_fn(outputs, labels)
        loss_r, current_R_val = calculate_pruning_loss_simple(model, ALPHA_TARGET, num_prunable_elements)
        
        # 引入一个超参数 lambda_prune 来放大剪枝损失的权重
        lambda_prune = 50.0 # 可以从1.0, 10.0, 100.0开始尝试
        total_loss = loss_ce + lambda_prune * loss_r
        # total_loss = loss_ce + loss_r
        
        # 反向传播
        total_loss.backward()
        
        # 更新参数
        optimizer_weights.step()
        optimizer_pruning.step()
        
        if i % 50 == 0:
            
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Total Loss: {total_loss.item():.4f}, CE Loss: {loss_ce.item():.4f}, Pruning Loss: {loss_r.item():.4f}, Current R: {current_R_val.item():.4f}")


print("第二阶段训练完成!")
# 保存最终的剪枝模型
output_filename = f"re_pruner_phase2_pruned_test_{NUM_CLASSES}class.pth"
print(f"正在将模型状态保存到: {output_filename} ...")
torch.save(model.state_dict(), output_filename)
print("保存成功！")