import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from functools import partial

# 导入 *原始* 的 ViT 和 Block
from timm.models.vision_transformer import VisionTransformer, Block 

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 100 
BATCH_SIZE = 64
FINETUNE_EPOCHS = 50
FINETUNE_LR = 1e-5

# !!! 必需：指向您 *prune_model.py* 输出的 *物理* 剪枝模型 !!!
PRUNED_MODEL_PATH = "re_pruner_PHYSICALLY_pruned.pth"
# (这是 prune_model.py 使用的状态字典，用于重建)
PHASE2_MODEL_PATH = "re_pruner_phase2_pruned_formal_r_logit_100class_r0.5.pth" 

# !!! 必需：您的100类ImageNet子集路径 !!!
IMAGENET_SUBSET_TRAIN_PATH = "/path/to/your/100_class_imagenet/train"
IMAGENET_SUBSET_VAL_PATH = "/path/to/your/100_class_imagenet/val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 准备数据集 (训练集 和 验证集) ---
print("正在加载数据集 (用于微调)...")
# ... (与上一条消息中的 finetune.py 相同的数据集加载代码) ...
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
state_dict_phase2 = torch.load(PHASE2_MODEL_PATH, map_location=device)
pruning_config = {}
new_head_counts = []
for i in range(12): # 12 blocks for DeiT-Small
    r_logit = state_dict_phase2[f'blocks.{i}.attn.r_logit']
    r = torch.sigmoid(r_logit).item()
    num_heads_to_keep = int(round((1.0 - r) * 6)) # 6 heads for DeiT-Small
    num_heads_to_keep = max(1, num_heads_to_keep)
    new_head_counts.append(num_heads_to_keep)
print(f"学习到的保留头数量: {new_head_counts}")

# b. 定义我们的 PrunedVisionTransformer 类 (与 prune_model.py 中一致)
class PrunedVisionTransformer(VisionTransformer):
    def __init__(self, head_counts_per_block, **kwargs):
        kwargs['depth'] = len(head_counts_per_block)
        super().__init__(**kwargs)
        del self.blocks
        dpr = [x.item() for x in torch.linspace(0, kwargs.get('drop_path_rate', 0.), kwargs['depth'])]
        self.blocks = nn.ModuleList([
            Block(
                dim=kwargs['embed_dim'], 
                num_heads=head_counts_per_block[i], # <-- 使用自定义的头数量
                mlp_ratio=kwargs.get('mlp_ratio', 4.),
                qkv_bias=kwargs.get('qkv_bias', True),
                drop_path=dpr[i],
                norm_layer=kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
            )
            for i in range(kwargs['depth'])
        ])
        self.apply(self._init_weights)

# c. 实例化物理上更小的模型
pruned_model = PrunedVisionTransformer(
    head_counts_per_block=new_head_counts,
    patch_size=16,
    embed_dim=384, # DeiT-Small
    depth=12,
    num_heads=6, # 占位符
    num_classes=NUM_CLASSES,
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
        
        # --- 正常前向传播 (不再需要 y_labels) ---
        outputs = model(images)
        
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
        
        # --- 正常前向传播 ---
        outputs = pruned_model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            pbar.set_postfix({"Loss": loss.item()})

    # --- 每个 Epoch 结束后进行验证 ---
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