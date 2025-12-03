# main_phase1_cifar.py
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import os

# 导入本地文件
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention, MaskedMlp
from utils_cifar import get_cifar10_loaders # <--- 使用 CIFAR 工具

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 10  # <--- 修改为 10
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 20 # CIFAR-10 可以适当减少轮数，或者保持 50
LAMBDA_SP = 0.01 
LAMBDA_SM = 0.01 

# --- 2. 准备数据集 (CIFAR-10) ---
print(f"正在加载 CIFAR-10 数据集...")
train_loader, val_loader = get_cifar10_loaders(BATCH_SIZE)
print(f"数据集加载完毕。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 加载SCFP可靠性得分 ---
print("加载SCFP可靠性得分...")
SCFP_SCORES_FILE = f'scfp_head_scores_cifar10_deit_small_patch16_224.pt' # <--- 修改文件名
if not os.path.exists(SCFP_SCORES_FILE):
    raise FileNotFoundError(f"SCFP得分文件 {SCFP_SCORES_FILE} 不存在。")

delta_f_scores = torch.load(SCFP_SCORES_FILE, map_location='cpu')
epsilon = 1e-8 
print(f"成功加载 SCFP 得分字典。")

# 预计算全局分数均值 (用于归一化) - 保持原逻辑
print("正在计算 SCFP 全局均值以进行归一化...")
all_head_scores = [v for k, v in delta_f_scores.items() if 'attn.head' in k]
all_neuron_scores = [v for k, v in delta_f_scores.items() if 'mlp.neuron' in k]

if len(all_head_scores) > 0:
    HEAD_SCORE_MEAN = torch.tensor(all_head_scores).abs().mean().item()
else:
    print("警告: 未找到 Head 分数，使用默认均值 1.0")
    HEAD_SCORE_MEAN = 1.0

if len(all_neuron_scores) > 0:
    NEURON_SCORE_MEAN = torch.tensor(all_neuron_scores).abs().mean().item()
else:
    print("警告: 未找到 Neuron 分数，使用默认均值 1.0")
    NEURON_SCORE_MEAN = 1.0

print(f"全局 Head 分数均值 (Abs): {HEAD_SCORE_MEAN:.6f}")
print(f"全局 Neuron 分数均值 (Abs): {NEURON_SCORE_MEAN:.6f}")

# --- 4. 加载模型并载入预训练权重 (完全保持原逻辑) ---
print("正在加载模型...")
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)

print("下载/加载标准预训练权重...")
base_model = timm.create_model('deit_small_patch16_224', pretrained=True)
base_state_dict = base_model.state_dict()

print("调整权重键名以匹配 MaskedAttention 和 MaskedMlp...")
new_state_dict = {}
for k, v in base_state_dict.items():
    if '.attn.' in k:
        new_k = k.replace('.attn.', '.attn.attn.')
    elif '.mlp.' in k:
        new_k = k.replace('.mlp.', '.mlp.mlp.')
    else:
        new_k = k
        
    if 'head' in k:
        # 跳过分类头，因为 CIFAR 是 10 类，ImageNet 是 1000 类
        continue
            
    new_state_dict[new_k] = v

missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
#  验证加载是否成功
print("\n--- 权重加载报告 ---")
print(f"未找到的键 (Missing keys): {len(missing_keys)}")
# 这里的 missing_keys 应该只包含 'explainability_mask', 'r_logit', 'theta' 等新参数
# 以及分类头(如果跳过了)
relevant_missing = [k for k in missing_keys if 'explainability_mask' not in k and 'r_logit' not in k and 'theta' not in k and 'head' not in k]
if len(relevant_missing) > 0:
    print(f"警告！以下关键权重未加载 (可能导致性能低下): \n{relevant_missing[:5]} ...")
else:
    print("成功：所有基础 Transformer 权重 (MHA, MLP) 均已正确加载！")
model.to(device)
print("模型准备就绪。")

# --- 5. 设置损失函数和优化器 (完全保持原逻辑) ---
def calculate_total_loss_re_pruner(
    model, outputs, labels, ce_loss_fn, lambda_sp, lambda_sm, 
    scfp_scores, device, epsilon,
    head_score_mean, neuron_score_mean
):
    loss_ce = ce_loss_fn(outputs, labels)
    loss_adaptive_sparse = torch.tensor(0.0, device=device)
    loss_smooth = torch.tensor(0.0, device=device)
    
    total_prunable_elements = 0

    for block_idx, block in enumerate(model.blocks):
        
        # --- A. 处理 Attention ---
        if isinstance(block.attn, MaskedAttention):
            mask_tensor = block.attn.explainability_mask 
            if mask_tensor.shape[0] > 1:
                loss_smooth += torch.norm(mask_tensor[1:] - mask_tensor[:-1], p=1)

            num_heads = mask_tensor.shape[1]
            head_keys = [f'blocks.{block_idx}.attn.head.{h}' for h in range(num_heads)]
            raw_scores = [scfp_scores.get(k, head_score_mean) for k in head_keys]
            raw_scores_tensor = torch.tensor(raw_scores, device=device)
            
            norm_scores = raw_scores_tensor / (head_score_mean + epsilon)
            penalty_vec = 1.0 / (torch.abs(norm_scores) + 0.1)
            
            for h in range(num_heads):
                loss_adaptive_sparse += penalty_vec[h] * torch.norm(mask_tensor[:, h, :], p=2)
            
            total_prunable_elements += num_heads

        # --- B. 处理 MLP ---
        if hasattr(block, 'mlp') and isinstance(block.mlp, MaskedMlp):
            mask_tensor_mlp = block.mlp.explainability_mask 
            if mask_tensor_mlp.shape[0] > 1:
                loss_smooth += torch.norm(mask_tensor_mlp[1:] - mask_tensor_mlp[:-1], p=1)
            
            hidden_dim = mask_tensor_mlp.shape[1]
            mlp_keys = [f'blocks.{block_idx}.mlp.neuron.{n}' for n in range(hidden_dim)]
            raw_scores_mlp = [scfp_scores.get(k, neuron_score_mean) for k in mlp_keys] 
            raw_scores_tensor_mlp = torch.tensor(raw_scores_mlp, device=device)
            
            norm_scores_mlp = raw_scores_tensor_mlp / (neuron_score_mean + epsilon)
            penalty_vec_mlp = 1.0 / (torch.abs(norm_scores_mlp) + 0.1)
            
            mask_col_norms = torch.norm(mask_tensor_mlp, p=2, dim=0)
            loss_adaptive_sparse += torch.sum(penalty_vec_mlp * mask_col_norms)
            
            total_prunable_elements += hidden_dim

    if total_prunable_elements > 0:
        loss_adaptive_sparse = loss_adaptive_sparse / total_prunable_elements
        loss_smooth = loss_smooth / total_prunable_elements

    total_loss = loss_ce + lambda_sp * loss_adaptive_sparse + lambda_sm * loss_smooth
    
    return total_loss, loss_ce, loss_adaptive_sparse, loss_smooth

# 冻结权重 (保持原逻辑)
for name, param in model.named_parameters():
    if "explainability_mask" in name:
        param.requires_grad = True 
    elif "head" in name:
        param.requires_grad = True 
    else:
        param.requires_grad = False

# 确认只有掩码(以及分类头)是可训练的
print("以下参数将被训练:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LEARNING_RATE, 
    momentum=0.9
)
ce_loss_fn = nn.CrossEntropyLoss()

# --- 6. 训练循环 ---
print("--- 开始第一阶段 (RE-Pruner 掩码学习) CIFAR-10 ---")
model.train()
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, y_labels=labels)
        
        loss, loss_c, loss_as, loss_s = calculate_total_loss_re_pruner(
            model, outputs, labels, ce_loss_fn, LAMBDA_SP, LAMBDA_SM,
            delta_f_scores, device, epsilon, HEAD_SCORE_MEAN, NEURON_SCORE_MEAN         
        )
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Total Loss: {loss.item():.4f}")
            print(f"  -> CE Loss: {loss_c.item():.4f}, Adaptive Sparse Loss: {loss_as.item():.4f}, Smooth Loss: {loss_s.item():.4f}")
output_filename = f"re_pruner_phase1_masks_cifar10.pth" # <--- 修改文件名
print(f"正在将模型状态保存到: {output_filename} ...")
torch.save(model.state_dict(), output_filename)
print("保存成功！")