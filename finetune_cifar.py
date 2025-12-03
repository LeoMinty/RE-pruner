# finetune_cifar.py
import torch
import torch.nn as nn
import argparse
from utils_cifar import get_cifar10_loaders
from prune_model import PrunedVisionTransformer # 需确保 prune_model.py 能被导入

parser = argparse.ArgumentParser()
parser.add_argument('--pruning_rate', type=float, required=True)
args = parser.parse_args()

RATE = args.pruning_rate
PHYSICAL_PATH = f"re_pruner_physically_pruned_cifar10_r{RATE}.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20 # Finetune
LR = 5e-5

# 1. 必须先读取模型结构配置 (或从保存的文件名/meta信息中读取)
# 为简化，我们这里重新加载一遍 state_dict 来推断结构 (有点笨拙但有效)
state_dict = torch.load(PHYSICAL_PATH, map_location='cpu')

# 推断每层结构
head_counts = []
neuron_counts = []
for i in range(12):
    # 通过查找 qkv.weight 的形状来推断 head 数量
    # shape: [3 * heads * 64, 384]
    qkv_weight = state_dict[f'blocks.{i}.attn.qkv.weight']
    heads = qkv_weight.shape[0] // 3 // 64
    head_counts.append(heads)
    
    # 通过 fc1.weight 形状推断 neurons
    # shape: [neurons, 384]
    fc1_weight = state_dict[f'blocks.{i}.mlp.fc1.weight']
    neurons = fc1_weight.shape[0]
    neuron_counts.append(neurons)

model = PrunedVisionTransformer(
    head_counts_per_block=head_counts,
    neuron_counts_per_block=neuron_counts,
    patch_size=16, embed_dim=384, depth=12,
    num_classes=10, qkv_bias=True, proj_bias=True
)
model.load_state_dict(state_dict)
model.to(DEVICE)

train_loader, val_loader = get_cifar10_loaders(64)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"Finetuning Rate {RATE}...")
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        
    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outputs = model(imgs)
            _, pred = outputs.max(1)
            total += lbls.size(0)
            correct += pred.eq(lbls).sum().item()
            
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"re_pruner_cifar10_r{RATE}_best.pth")
    print(f"Epoch {epoch}: Acc {acc:.2f}% (Best: {best_acc:.2f}%)")

print(f"FINAL RESULT for RATE {RATE}: {best_acc}")