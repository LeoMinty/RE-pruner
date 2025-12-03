# main_phase2_cifar.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention, MaskedMlp 
from utils_cifar import get_cifar10_loaders # <---

# --- 增加参数解析 ---
parser = argparse.ArgumentParser(description='RE-Pruner Phase 2 for CIFAR-10')
parser.add_argument('--pruning_rate', type=float, default=0.4, help='Target pruning rate')
args = parser.parse_args()

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 10 # <---
BATCH_SIZE = 128
EPOCHS = 20        
ALPHA_TARGET = args.pruning_rate # <--- 动态剪枝率

MODEL_STATE_PATH = "re_pruner_phase1_masks_cifar10.pth" # <--- Phase 1 产物

# --- 2. 准备数据集 ---
train_loader, val_loader = get_cifar10_loaders(BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 加载模型并切换到剪枝模式 ---
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
print(f"正在从 {MODEL_STATE_PATH} 加载模型状态...")
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=device), strict=False)
model.to(device)

print("正在初始化剪枝阈值 (theta)...")
with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, (MaskedAttention, MaskedMlp)): 
            module.is_pruning_phase = True
            if isinstance(module, MaskedAttention):
                scores = module.get_head_importance()
            else:
                scores = module.get_neuron_importance()
            mean_score = scores.mean()
            module.theta.data.fill_(mean_score.item())

# 激活剪枝模式
for module in model.modules():
    if isinstance(module, MaskedAttention):
        module.is_pruning_phase = True

# --- 4. 设置损失函数和优化器 (完全保持原逻辑) ---
ce_loss_fn = nn.CrossEntropyLoss()
beta = nn.Parameter(torch.tensor(1.0, device=device))    
gamma = nn.Parameter(torch.tensor(0.01, device=device)) 

def calculate_pruning_loss(model, alpha_target, beta, gamma):
    total_params_proxy = 0.0
    kept_params_proxy = torch.tensor(0.0, device=device)
    WEIGHT_HEAD = 128.0
    WEIGHT_NEURON = 1.0
    loss_theta_boundary = torch.tensor(0.0, device=device)
    MIN_RETENTION_RATIO = 0.25 
    
    for module in model.modules():
        if isinstance(module, MaskedAttention):
            scores = module.get_head_importance()
            soft_mask = torch.sigmoid((scores - module.theta) * module.temperature)
            kept_params_proxy += soft_mask.sum() * WEIGHT_HEAD
            total_params_proxy += scores.numel() * WEIGHT_HEAD

        elif isinstance(module, MaskedMlp):
            scores = module.get_neuron_importance()
            soft_mask = torch.sigmoid((scores - module.theta) * module.temperature)
            kept_params_proxy += soft_mask.sum() * WEIGHT_NEURON
            total_params_proxy += scores.numel() * WEIGHT_NEURON
            
            sorted_scores, _ = torch.sort(scores, descending=True)
            num_neurons = scores.numel()
            safe_idx = int(num_neurons * MIN_RETENTION_RATIO)
            safe_idx = min(safe_idx, num_neurons - 1)
            safe_threshold = sorted_scores[safe_idx]
            
            if module.theta > safe_threshold:
                penalty = (module.theta - safe_threshold) ** 2
                loss_theta_boundary = loss_theta_boundary + penalty.sum()

    if total_params_proxy == 0: 
        return torch.tensor(0.0, device=device), 0.0

    current_retention_rate = kept_params_proxy / total_params_proxy
    current_pruning_rate = 1.0 - current_retention_rate
    diff = abs(alpha_target - current_pruning_rate)
    loss_r = beta * (diff ** 2) + gamma * diff
    total_pruning_loss = loss_r + 10000.0 * loss_theta_boundary
    
    return total_pruning_loss, current_pruning_rate

pruning_params = []
model_weights = []
for name, param in model.named_parameters():
    if "explainability_mask" in name:
        param.requires_grad = False
    elif "theta" in name:
        pruning_params.append(param)
    else:
        model_weights.append(param)
model_weights.append(beta)
model_weights.append(gamma)
        
optimizer_weights = torch.optim.AdamW(model_weights, lr=1e-5)
optimizer_pruning = torch.optim.AdamW(pruning_params, lr=0.005)

# --- 5. 第二阶段训练循环 ---
model.train()
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer_weights.zero_grad()
        optimizer_pruning.zero_grad()

        with torch.no_grad():
            beta.data.clamp_(min=0)
            gamma.data.clamp_(min=0)
        
        outputs = model(images, y_labels=None)
        loss_ce = ce_loss_fn(outputs, labels)
        loss_r, current_R_val = calculate_pruning_loss(model, ALPHA_TARGET, beta, gamma)
        
        lambda_prune = 80.0
        total_loss = loss_ce + lambda_prune * loss_r
        total_loss.backward()
        
        optimizer_weights.step()
        optimizer_pruning.step()

        with torch.no_grad():
            lr_lagrange = 0.01
            if beta.grad is not None:
                beta.data.add_(lr_lagrange * beta.grad)
                beta.grad.zero_()
            if gamma.grad is not None:
                gamma.data.add_(lr_lagrange * gamma.grad)
                gamma.grad.zero_()
            beta.data.clamp_(min=0)
            gamma.data.clamp_(min=0)
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}], Total: {total_loss.item():.4f}, Current R: {current_R_val.item():.4f}")

print("第二阶段训练完成!")
# 文件名带上 pruning_rate
output_filename = f"re_pruner_phase2_cifar10_r{ALPHA_TARGET}.pth" 
print(f"正在将模型状态保存到: {output_filename} ...")
torch.save(model.state_dict(), output_filename)