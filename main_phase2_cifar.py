# main_phase2_cifar.py
import torch
import torch.nn as nn
import argparse
from utils_cifar import get_cifar10_loaders
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention, MaskedMlp

# --- 参数解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--pruning_rate', type=float, default=0.4, help='Target Pruning Rate (0.0 - 1.0)')
args = parser.parse_args()

# --- 配置 ---
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 10 # 快速搜索阈值
ALPHA_TARGET = args.pruning_rate
PHASE1_PATH = "re_pruner_phase1_masks_cifar10.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, _ = get_cifar10_loaders(BATCH_SIZE)

# --- 模型 ---
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(PHASE1_PATH, map_location=DEVICE), strict=False)
model.to(DEVICE)

# 初始化 Theta
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, (MaskedAttention, MaskedMlp)):
            m.is_pruning_phase = True
            scores = m.get_head_importance() if isinstance(m, MaskedAttention) else m.get_neuron_importance()
            m.theta.data.fill_(scores.mean().item())

# --- 优化器 ---
beta = nn.Parameter(torch.tensor(1.0, device=DEVICE))
gamma = nn.Parameter(torch.tensor(0.01, device=DEVICE))

params_pruning = [p for n, p in model.named_parameters() if 'theta' in n]
params_weights = [p for n, p in model.named_parameters() if 'theta' not in n and 'explainability_mask' not in n]

opt_pruning = torch.optim.AdamW(params_pruning, lr=0.01)
opt_weights = torch.optim.AdamW(params_weights + [beta, gamma], lr=1e-5) # beta/gamma 放在这里更新

criterion = nn.CrossEntropyLoss()

def calc_pruning_loss(model, alpha_target, beta, gamma):
    kept, total = 0.0, 0.0
    for m in model.modules():
        if isinstance(m, (MaskedAttention, MaskedMlp)):
            scores = m.get_head_importance() if isinstance(m, MaskedAttention) else m.get_neuron_importance()
            soft_mask = torch.sigmoid((scores - m.theta) * m.temperature)
            kept += soft_mask.sum()
            total += scores.numel()
    
    current_pruning = 1.0 - (kept / (total + 1e-6))
    diff = abs(alpha_target - current_pruning)
    loss_r = beta * (diff ** 2) + gamma * diff
    return loss_r, current_pruning

# --- 训练 ---
print(f"Start Phase 2 for Target Rate: {ALPHA_TARGET}")
model.train()
for epoch in range(EPOCHS):
    for i, (imgs, lbls) in enumerate(train_loader):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        
        opt_weights.zero_grad()
        opt_pruning.zero_grad()
        
        outputs = model(imgs) # 此时使用 class-agnostic 掩码
        loss_ce = criterion(outputs, lbls)
        loss_r, curr_r = calc_pruning_loss(model, ALPHA_TARGET, beta, gamma)
        
        total_loss = loss_ce + 10.0 * loss_r # 权重可调
        total_loss.backward()
        
        opt_weights.step()
        opt_pruning.step()
        
        # Beta/Gamma 梯度上升 (Lagrange Multipliers)
        with torch.no_grad():
            beta.data += 0.01 * beta.grad
            gamma.data += 0.01 * gamma.grad
            beta.data.clamp_(min=0)
            gamma.data.clamp_(min=0)
            beta.grad.zero_()
            gamma.grad.zero_()
            
        if i % 50 == 0:
            print(f"Epoch {epoch} Step {i}: Loss {total_loss.item():.4f}, Current Pruning: {curr_r.item():.4f}")

output_file = f"re_pruner_phase2_cifar10_r{ALPHA_TARGET}.pth"
torch.save(model.state_dict(), output_file)
print(f"Saved to {output_file}")