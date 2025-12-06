# finetune_cifar.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from functools import partial
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils_cifar import get_cifar10_loaders # <---

# 导入 *原始* 的 ViT 基类
from timm.models.vision_transformer import VisionTransformer, Block as TimmBlock, Attention as TimmAttention
from timm.layers import DropPath, Mlp, to_2tuple

# --- 参数解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--pruning_rate', type=float, default=0.4)
args = parser.parse_args()
RATE = args.pruning_rate

# --- 1. 配置 ---
NUM_CLASSES = 10 # <---
BATCH_SIZE = 128
FINETUNE_EPOCHS = 30
FINETUNE_LR = 5e-4
WEIGHT_DECAY = 1e-4    # 保持 AdamW 的默认权重衰减

# 注意：这里需要加载 PHASE2 的模型来计算结构，同时也加载物理剪枝后的权重
# 文件名根据 rate 变化
PHASE2_MODEL_PATH = f"re_pruner_phase2_cifar10_r{RATE}.pth" 
PRUNED_MODEL_PATH = f"re_pruner_physically_pruned_cifar10_r{RATE}.pth"
# PRUNED_MODEL_PATH = f"re_pruner_finetuned_best_cifar10_r{RATE}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 准备数据集 ---
train_loader, val_loader = get_cifar10_loaders(BATCH_SIZE)

# --- 3. 重建结构 (保持原逻辑) ---
print("--- 正在重建物理剪枝模型结构 ---")
if not os.path.exists(PHASE2_MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {PHASE2_MODEL_PATH} 不存在。")
state_dict_phase2 = torch.load(PHASE2_MODEL_PATH, map_location=device)

new_head_counts = []
new_neuron_counts = []
BASE_NUM_HEADS = 6
BASE_EMBED_DIM = 384
HEAD_DIM = 64

for i in range(12): 
    theta_attn = state_dict_phase2.get(f'blocks.{i}.attn.theta')
    mask_attn = state_dict_phase2[f'blocks.{i}.attn.explainability_mask']
    if theta_attn is None: 
        print(f"Error: Block {i} missing theta")
        exit()
    importance_attn = mask_attn.mean(dim=0).abs().sum(dim=-1)
    kept_heads = torch.nonzero(importance_attn > theta_attn.item()).squeeze(1)
    n_heads = kept_heads.numel()
    if n_heads == 0: n_heads = 1
    new_head_counts.append(n_heads)
    
    theta_mlp = state_dict_phase2.get(f'blocks.{i}.mlp.theta')
    mask_mlp = state_dict_phase2[f'blocks.{i}.mlp.explainability_mask']
    importance_mlp = mask_mlp.mean(dim=0).abs()
    kept_neurons = torch.nonzero(importance_mlp > theta_mlp.item()).squeeze(1)
    n_neurons = kept_neurons.numel()
    if n_neurons == 0: n_neurons = 1
    new_neuron_counts.append(n_neurons)

print(f"学习到的保留头数量: {new_head_counts}")
print(f"学习到的保留神经元数量: {new_neuron_counts}")
# 定义类 (为了不依赖外部文件，这里再次定义，或者 import prune_model_cifar)
# 为了安全起见，这里再贴一遍类定义，确保脚本独立运行不出错
# (如果您已经将类定义放在一个公共文件中，可以 import，但为了复现您的 finetune.py，我保留完整定义)
BASE_EMBED_DIM = 384
HEAD_DIM = BASE_EMBED_DIM // BASE_NUM_HEADS
class PrunedAttention(TimmAttention):
    def __init__(self, dim, num_heads, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        super(TimmAttention, self).__init__() 
        self.num_heads = num_heads
        self.head_dim = HEAD_DIM
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, (num_heads * self.head_dim) * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_heads * self.head_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PrunedBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, proj_bias=True,
                 proj_drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_ratio=None):
        super(TimmBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrunedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, bias=proj_bias, drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class PrunedVisionTransformer(VisionTransformer):
    def __init__(self, head_counts_per_block, neuron_counts_per_block, **kwargs):
        depth = len(head_counts_per_block)
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        qkv_bias = kwargs.get('qkv_bias', True)
        proj_bias = kwargs.get('proj_bias', True) 
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
        proj_drop_rate = kwargs.get('drop_rate', 0.) 
        norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
        act_layer = kwargs.get('act_layer', nn.GELU)

        super_kwargs = kwargs.copy()
        super_kwargs['depth'] = depth
        super_kwargs['num_heads'] = 6
        super().__init__(**super_kwargs)
        del self.blocks
        self.blocks = nn.ModuleList([
            PrunedBlock( 
                dim=kwargs['embed_dim'], 
                num_heads=head_counts_per_block[i], 
                mlp_hidden_dim=neuron_counts_per_block[i],
                qkv_bias=qkv_bias, proj_bias=proj_bias,
                proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer,
                mlp_ratio=None
            ) for i in range(depth)
        ])
        self.apply(self._init_weights)
    
    def forward_features(self, x, attn_mask=None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

# c. 实例化模型
pruned_model = PrunedVisionTransformer(
    head_counts_per_block=new_head_counts,
    neuron_counts_per_block=new_neuron_counts, 
    patch_size=16, embed_dim=BASE_EMBED_DIM, depth=12,
    num_classes=NUM_CLASSES, qkv_bias=True, proj_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    act_layer=nn.GELU, drop_rate=0.0, attn_drop_rate=0.0
)

# d. 加载权重
print(f"正在从 {PRUNED_MODEL_PATH} 加载 *物理* 剪枝模型权重...")
if not os.path.exists(PRUNED_MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {PRUNED_MODEL_PATH} 不存在。")
pruned_model.load_state_dict(torch.load(PRUNED_MODEL_PATH, map_location=device))
pruned_model.to(device)
print("加载成功！")
# --- 4. 优化器 ---
# 使用论文推荐的 AdamW
optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)

# 新增: 余弦退火调度器
# T_max 设为总 Epoch 数，eta_min 设为极小值 (如 1e-6)
scheduler = CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)

criterion = nn.CrossEntropyLoss()



# --- 5. 验证函数 ---
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    pbar = tqdm(loader, desc="验证中")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images) # <-- 正常前向传播
        
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        _, pred = outputs.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

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

# --- 6. 训练循环 ---
print(f"--- 开始微调 (Rate: {RATE}) ---")
best_acc1 = 0.0
best_acc5 = 0.0

for epoch in range(FINETUNE_EPOCHS):
    pruned_model.train() 
    pbar = tqdm(train_loader, desc=f"微调 Epoch {epoch+1}/{FINETUNE_EPOCHS}")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = pruned_model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            pbar.set_postfix({"Loss": loss.item()})
    #更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
        
        

    val_loss, val_acc1, val_acc5 = validate(pruned_model, val_loader, criterion, device)
    print(f"Epoch {epoch+1} 验证完成: Avg Loss: {val_loss:.4f}, Top-1 Acc: {val_acc1:.2f}%, Top-5 Acc: {val_acc5:.2f}%")
    
    if val_acc1 > best_acc1:
        best_acc1 = val_acc1
        best_acc5 = val_acc5
        # 文件名带 Rate
        output_filename = f"re_pruner_finetuned_best_cifar10_r{RATE}.pth"
        torch.save(pruned_model.state_dict(), output_filename)
        print(f"*** 新的最佳Top-1准确率！模型已保存到 {output_filename} ***")

print("微调完成！")
print(f"--- 最终最佳 Top-1 准确率: {best_acc1:.2f}% ---")
print(f"--- 最终最佳 Top-5 准确率: {best_acc5:.2f}% ---")