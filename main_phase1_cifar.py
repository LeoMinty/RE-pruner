# main_phase1_cifar.py
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
import os
from utils_cifar import get_cifar10_loaders

# 导入修改后的模型结构
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention, MaskedMlp

# --- 配置 ---
NUM_CLASSES = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 10 # CIFAR 收敛快，Mask 训练不需要像 ImageNet 那么多轮
LAMBDA_SP = 0.01
LAMBDA_SM = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCFP_FILE = 'scfp_head_scores_cifar10_deit_small_patch16_224.pt'

# --- 加载数据 ---
train_loader, val_loader = get_cifar10_loaders(BATCH_SIZE)

# --- 加载 SCFP 得分 ---
if not os.path.exists(SCFP_FILE):
    raise FileNotFoundError(f"请先运行 compute_scfp_scores_cifar.py 生成 {SCFP_FILE}")
    
delta_f_scores = torch.load(SCFP_FILE, map_location='cpu')
# 计算均值
all_scores = [v for v in delta_f_scores.values()]
SCORE_MEAN = torch.tensor(all_scores).abs().mean().item() if all_scores else 1.0
print(f"Global Score Mean: {SCORE_MEAN}")

# --- 模型初始化 ---
# 1. 创建 10 类模型
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)

# 2. 加载 ImageNet 预训练权重
print("Loading ImageNet weights...")
base_model = timm.create_model('deit_small_patch16_224', pretrained=True)
base_dict = base_model.state_dict()
new_dict = {}
for k, v in base_dict.items():
    if 'head' in k: continue # 跳过分类头
    if '.attn.' in k: new_k = k.replace('.attn.', '.attn.attn.')
    elif '.mlp.' in k: new_k = k.replace('.mlp.', '.mlp.mlp.')
    else: new_k = k
    new_dict[new_k] = v
    
model.load_state_dict(new_dict, strict=False)
model.to(DEVICE)

# --- 损失函数 ---
def calculate_loss(model, outputs, labels, criterion, lambda_sp, lambda_sm):
    loss_ce = criterion(outputs, labels)
    loss_sparse = torch.tensor(0.0, device=DEVICE)
    loss_smooth = torch.tensor(0.0, device=DEVICE)
    count = 0
    
    for module in model.modules():
        if isinstance(module, (MaskedAttention, MaskedMlp)):
            mask = module.explainability_mask
            # Smoothness
            if mask.shape[0] > 1:
                loss_smooth += torch.norm(mask[1:] - mask[:-1], p=1)
            
            # Sparsity with Penalty
            if isinstance(module, MaskedAttention):
                keys = [f'blocks.{count//2}.attn.head.{h}' for h in range(mask.shape[1])]
            else:
                keys = [f'blocks.{count//2}.mlp.neuron.{n}' for n in range(mask.shape[1])]
            
            scores = torch.tensor([delta_f_scores.get(k, SCORE_MEAN) for k in keys], device=DEVICE)
            penalty = 1.0 / (torch.abs(scores / SCORE_MEAN) + 0.1)
            
            if isinstance(module, MaskedAttention):
                for h in range(mask.shape[1]):
                    loss_sparse += penalty[h] * torch.norm(mask[:, h, :], p=2)
            else:
                loss_sparse += torch.sum(penalty * torch.norm(mask, p=2, dim=0))
            
            if isinstance(module, MaskedMlp): count += 2 # Block done
            
    return loss_ce + lambda_sp * loss_sparse + lambda_sm * loss_smooth

# --- 训练 ---
optimizer = torch.optim.SGD(
    [p for n, p in model.named_parameters() if 'explainability_mask' in n or 'head' in n],
    lr=LEARNING_RATE, momentum=0.9
)
ce_loss = nn.CrossEntropyLoss()

print("Start Phase 1 Training...")
model.train()
for epoch in range(EPOCHS):
    for i, (imgs, lbls) in enumerate(train_loader):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs, y_labels=lbls)
        loss = calculate_loss(model, outputs, lbls, ce_loss, LAMBDA_SP, LAMBDA_SM)
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), f"re_pruner_phase1_masks_cifar10.pth")
print("Phase 1 Done.")