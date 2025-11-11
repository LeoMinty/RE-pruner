# finetune.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from functools import partial

# 导入 *原始* 的 ViT 基类
from timm.models.vision_transformer import VisionTransformer, Block as TimmBlock, Attention as TimmAttention
from timm.layers import DropPath, Mlp, to_2tuple

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 100 
BATCH_SIZE = 64 # 减小一点批量大小以便在VRAM中容纳
FINETUNE_EPOCHS = 50
FINETUNE_LR = 1e-5

PRUNED_MODEL_PATH = "re_pruner_PHYSICALLY_pruned.pth"
PHASE2_MODEL_PATH = "re_pruner_phase2_pruned_formal_r_logit_100class_r0.5.pth" 

IMAGENET_SUBSET_TRAIN_PATH = "/root/autodl-tmp/imagenet100"
IMAGENET_SUBSET_VAL_PATH = "/root/autodl-tmp/imagenet100_val" # 验证集路径

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 准备数据集 (训练集 和 验证集) ---
print("正在加载数据集 (用于微调)...")
if not os.path.exists(IMAGENET_SUBSET_TRAIN_PATH) or not os.path.exists(IMAGENET_SUBSET_VAL_PATH):
    print(f"错误: 路径 '{IMAGENET_SUBSET_TRAIN_PATH}' 或 '{IMAGENET_SUBSET_VAL_PATH}' 不存在。请修改 finetune.py 中的路径。")
    exit()

transform_train = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = datasets.ImageFolder(IMAGENET_SUBSET_TRAIN_PATH, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

transform_val = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_dataset = datasets.ImageFolder(IMAGENET_SUBSET_VAL_PATH, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"训练集: {len(train_dataset)} 图像, 验证集: {len(val_dataset)} 图像。")

# --- 3. 重建 *物理* 剪枝的模型结构 ---
print("--- 正在重建物理剪枝模型结构 ---")
# a. 首先，我们需要知道每层到底留了几个头
if not os.path.exists(PHASE2_MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {PHASE2_MODEL_PATH} 不存在。")
state_dict_phase2 = torch.load(PHASE2_MODEL_PATH, map_location=device)
new_head_counts = []
for i in range(12): # 12 blocks for DeiT-Small
    r_logit = state_dict_phase2[f'blocks.{i}.attn.r_logit']
    r = torch.sigmoid(r_logit).item()
    num_heads_to_keep = int(round((1.0 - r) * 6)) # 6 heads for DeiT-Small
    num_heads_to_keep = max(1, num_heads_to_keep)
    new_head_counts.append(num_heads_to_keep)
print(f"学习到的保留头数量: {new_head_counts}")

# b. 定义我们的 Pruned 类 (与 prune_model.py  一致)
BASE_EMBED_DIM = 384
HEAD_DIM = BASE_EMBED_DIM // 6

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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_bias=True,
                 proj_drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TimmBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrunedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, bias=proj_bias, drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class PrunedVisionTransformer(VisionTransformer):
    def __init__(self, head_counts_per_block, **kwargs):
        depth = len(head_counts_per_block)
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        mlp_ratio = kwargs.get('mlp_ratio', 4.)
        qkv_bias = kwargs.get('qkv_bias', True)
        proj_bias = kwargs.get('proj_bias', True) 
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
        proj_drop_rate = kwargs.get('drop_rate', 0.) 
        norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
        act_layer = kwargs.get('act_layer', nn.GELU)

        super_kwargs = kwargs.copy()
        super_kwargs['depth'] = depth
        super_kwargs['num_heads'] = 6 # 占位符
        super().__init__(**super_kwargs)
        del self.blocks
        self.blocks = nn.ModuleList([
            PrunedBlock( 
                dim=kwargs['embed_dim'], num_heads=head_counts_per_block[i], 
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias,
                proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer
            ) for i in range(depth)
        ])
        self.apply(self._init_weights)
    
    def forward_features(self, x, attn_mask=None):
        """重写 forward_features 以正确处理 ModuleList blocks"""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x

# c. 实例化物理上更小的模型
pruned_model = PrunedVisionTransformer(
    head_counts_per_block=new_head_counts,
    patch_size=16, embed_dim=BASE_EMBED_DIM, depth=12,
    num_classes=NUM_CLASSES, qkv_bias=True, proj_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    act_layer=nn.GELU, drop_rate=0.0, attn_drop_rate=0.0
)

# d. 加载 *物理* 剪枝后的权重
print(f"正在从 {PRUNED_MODEL_PATH} 加载 *物理* 剪枝模型权重...")
if not os.path.exists(PRUNED_MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {PRUNED_MODEL_PATH} 不存在。请先运行 prune_model.py。")
pruned_model.load_state_dict(torch.load(PRUNED_MODEL_PATH, map_location=device))
pruned_model.to(device)
print("加载成功！")

# --- 4. 设置优化器和损失函数 ---
optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=FINETUNE_LR) # 优化所有参数
criterion = nn.CrossEntropyLoss()

# --- 5. 验证函数 (计算Top-1和Top-5准确率) ---
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

# --- 6. 微调训练循环 ---
print("--- 开始第三阶段 (Finetune 物理剪枝模型) ---")
best_acc1 = 0.0
best_acc5 = 0.0

for epoch in range(FINETUNE_EPOCHS):
    pruned_model.train() 
    pbar = tqdm(train_loader, desc=f"微调 Epoch {epoch+1}/{FINETUNE_EPOCHS}")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = pruned_model(images) # <-- 正常前向传播
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            pbar.set_postfix({"Loss": loss.item()})

    val_loss, val_acc1, val_acc5 = validate(pruned_model, val_loader, criterion, device)
    print(f"Epoch {epoch+1} 验证完成: Avg Loss: {val_loss:.4f}, Top-1 Acc: {val_acc1:.2f}%, Top-5 Acc: {val_acc5:.2f}%")
    
    if val_acc1 > best_acc1:
        best_acc1 = val_acc1
        best_acc5 = val_acc5 
        output_filename = f"re_pruner_finetuned_best_{NUM_CLASSES}class.pth"
        torch.save(pruned_model.state_dict(), output_filename)
        print(f"*** 新的最佳Top-1准确率！模型已保存到 {output_filename} ***")

print("微调完成！")
print(f"--- 最终最佳 Top-1 准确率: {best_acc1:.2f}% ---")
print(f"--- 最终最佳 Top-5 准确率: {best_acc5:.2f}% ---")