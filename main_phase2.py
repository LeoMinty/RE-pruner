# main_phase2.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 关键：从你修改过的本地文件导入模型
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention # 导入用于类型检查

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 100
BATCH_SIZE = 64
EPOCHS = 80 # 论文中DeiT的剪枝训练轮数
ALPHA_TARGET = 0.2 # 目标总剪枝率

# 模型状态文件路径
MODEL_STATE_PATH = "re_pruner_phase1_masks_100class.pth"

# --- 2. 准备数据集 (与第一阶段相同) ---
IMAGENET_SUBSET_TRAIN_PATH = "/root/autodl-tmp/imagenet100"
IMAGENET_SUBSET_VAL_PATH = "/root/autodl-tmp/imagenet100_val" # 验证集路径

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224), # 使用RandomCrop进行训练
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

# 加载训练集
train_dataset = datasets.ImageFolder(IMAGENET_SUBSET_TRAIN_PATH, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# (如果需要) 加载验证集
val_dataset = datasets.ImageFolder(IMAGENET_SUBSET_VAL_PATH, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"数据集加载完毕，共 {len(train_dataset)} 张训练图像。")
# --- 3. 加载模型并切换到剪枝模式 ---
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
print(f"正在从 {MODEL_STATE_PATH} 加载模型状态...")
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=device), strict=False)
model.to(device)
print("加载成功！")

print("正在强制重新初始化剪枝参数 (theta)...")
with torch.no_grad(): 
    for module in model.modules():
        if isinstance(module, MaskedAttention):
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

# 新增：用于L_R损失的可学习参数
beta = nn.Parameter(torch.tensor(0.0, device=device))
gamma = nn.Parameter(torch.tensor(0.0, device=device))

def calculate_pruning_loss(model, alpha_target, total_prunable_elements):
    """计算剪枝率正则化损失 L_R (Eq. 10, 11)"""
    n = 10.0
    current_pruned_elements = torch.tensor(0.0, device=device)
    total_elements = 0.0
    for module in model.modules():
        if isinstance(module, MaskedAttention):
            # 掩码分数在第一阶段已固定，不跟踪其梯度
            mask_scores = module.explainability_mask.data
            theta = module.theta # theta 是可学习的
            
            # (1 - 门控因子) ≈ 剪枝概率
            pruned_probability = 0.5 * (1.0 - torch.tanh(n * (mask_scores - theta)))
            
            current_pruned_elements += pruned_probability.sum()
            total_elements += mask_scores.numel()

    if total_elements == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    current_R_avg = current_pruned_elements / total_elements
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

# 添加beta和gamma到模型权重组
model_weights.append(beta)
model_weights.append(gamma)
        
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
        loss_r = calculate_pruning_loss(model, ALPHA_TARGET, num_prunable_elements, beta, gamma)
        
        # 引入一个超参数 lambda_prune 来放大剪枝损失的权重
        lambda_prune = 10.0 # 可以从1.0, 10.0, 100.0开始尝试
        total_loss = loss_ce + lambda_prune * loss_r
        # total_loss = loss_ce + loss_r
        
        # 反向传播
        total_loss.backward()
        
        # 更新参数
        optimizer_weights.step()
        optimizer_pruning.step()
        
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Total Loss: {total_loss.item():.4f}, CE Loss: {loss_ce.item():.4f}, Pruning Loss: {loss_r.item():.4f}")

            print(f"--> beta: {beta.item():.6f}, gamma: {gamma.item():.6f}")

print("第二阶段训练完成!")
# 保存最终的剪枝模型
output_filename = f"re_pruner_phase2_pruned_formal_theta_{NUM_CLASSES}class_r{ALPHA_TARGET}.pth"
print(f"正在将模型状态保存到: {output_filename} ...")
torch.save(model.state_dict(), output_filename)
print("保存成功！")