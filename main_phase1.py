# main_phase1.py
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 关键：从你修改过的本地文件导入模型
from deit_modified import deit_small_patch16_224
from vision_transformer_modified import MaskedAttention

# --- 1. 定义超参数和配置 ---
NUM_CLASSES = 100  # ImageNet100
BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 50 # 论文中DeiT的训练轮数
LAMBDA_SP = 1e-6 # 稀疏性损失权重 (需要调试)
LAMBDA_SM = 0.01 # 平滑性损失权重 (需要调试)

# --- 2. 准备数据集 (ImageNet100) ---
IMAGENET_SUBSET_TRAIN_PATH = "/root/autodl-tmp/imagenet100"
IMAGENET_SUBSET_VAL_PATH = "/root/autodl-tmp/imagenet100_val" # 验证集路径

print(f"正在加载 {NUM_CLASSES} 类 ImageNet 子集...")
if not os.path.exists(IMAGENET_SUBSET_TRAIN_PATH):
    print(f"错误: 路径 '{IMAGENET_SUBSET_TRAIN_PATH}' 不存在。请修改 main_phase1.py 中的路径。")
    exit()

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
print(f"数据集加载完毕，共 {len(train_dataset)} 张训练图像。")
# (如果需要) 加载验证集
val_dataset = datasets.ImageFolder(IMAGENET_SUBSET_VAL_PATH, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 加载SCFP可靠性得分 ---
print("加载SCFP可靠性得分...")
SCFP_SCORES_FILE = f'scfp_head_scores_deit_small_patch16_224_100class.pt'
if not os.path.exists(SCFP_SCORES_FILE):
    raise FileNotFoundError(f"SCFP得分文件 {SCFP_SCORES_FILE} 不存在。请先运行 compute_scfp_scores.py。")

delta_f_scores = torch.load(SCFP_SCORES_FILE, map_location='cpu')
epsilon = 1e-8 # 防止除以零
print(f"成功加载 {len(delta_f_scores)} 个头的SCFP得分。")

# --- 4. 加载模型并载入预训练权重 (修复版) ---
print("正在加载模型...")
# 1. 创建模型实例 (pretrained=False, 因为我们要手动加载)
model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)

# 2. 下载标准 DeiT-Small 预训练权重
# 我们创建一个临时的标准模型来获取权重，或者直接下载权重文件
print("下载/加载标准预训练权重...")
base_model = timm.create_model('deit_small_patch16_224', pretrained=True)
base_state_dict = base_model.state_dict()

# 3. 调整权重键名以匹配 RE-Pruner 结构
print("调整权重键名以匹配 MaskedAttention...")
new_state_dict = {}
for k, v in base_state_dict.items():
    # 如果是注意力层的权重，需要增加一个 .attn 中间层
    if '.attn.' in k:
        new_k = k.replace('.attn.', '.attn.attn.')
    else:
        new_k = k
        
    # 处理分类头 (1000类 -> 100类)
    # 如果是 head.weight/bias，且形状不匹配，则跳过(保持随机初始化)或进行处理
    if 'head' in k:
        if v.shape != model.state_dict()[k].shape:
            print(f"跳过分类头权重: {k} (形状不匹配)")
            continue
            
    new_state_dict[new_k] = v

# 4. 加载调整后的权重到模型中
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

# 5. 验证加载是否成功
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


# --- 5. 设置损失函数和优化器 ---
def calculate_total_loss_re_pruner(
    model, 
    outputs, 
    labels, 
    ce_loss_fn, 
    lambda_sp, 
    lambda_sm, 
    scfp_scores, 
    device, 
    epsilon
):
    """
    计算第一阶段的总损失，使用基于SCFP的自适应稀疏正则化 (RE-Pruner)。
    """
    
    # 1. 交叉熵损失
    loss_ce = ce_loss_fn(outputs, labels)
    
    loss_adaptive_sparse = torch.tensor(0.0, device=device)
    loss_smooth = torch.tensor(0.0, device=device)

    # 遍历模型中的所有Block
    for block_idx, block in enumerate(model.blocks):
        # 确保我们正在处理正确的模块
        if isinstance(block.attn, MaskedAttention):
            # 获取该Block中所有头的掩码张量
            # 形状为: [num_classes, num_heads, head_dim]
            mask_tensor = block.attn.explainability_mask 
            
            # 3. 平滑性损失 (L_smooth - Eq. 6) - 论文中的简化实现
            if mask_tensor.shape[0] > 1: # 确保类别数大于1
                diff = mask_tensor[1:, ...] - mask_tensor[:-1, ...]
                loss_smooth += torch.norm(diff, p=1) # L1 norm of differences

            # 2. 自适应稀疏性损失 (L_adaptive_sparse - 融合方案)
            num_heads_in_block = mask_tensor.shape[1]
            for head_idx in range(num_heads_in_block):
                # 构建SCFP得分文件中的key
                scfp_key = f'blocks.{block_idx}.attn.head.{head_idx}'
                
                if scfp_key in scfp_scores:
                    delta_F_i = scfp_scores[scfp_key]
                    
                    # 可靠性越低，惩罚权重越高。
                    delta_F_i_tensor = torch.tensor(delta_F_i, device=device)
                    
                    # 惩罚权重 (基于费舍尔信息正则化)
                    penalty_weight = 1.0 / (torch.abs(delta_F_i_tensor) + epsilon)
                    
                    # 提取这个特定头的掩码 (跨所有类别)
                    # 形状: [num_classes, head_dim]
                    head_mask = mask_tensor[:, head_idx, :]
                    
                    # 计算L2范数并施加惩罚
                    loss_adaptive_sparse += penalty_weight * torch.norm(head_mask, p=2)
                
                else:
                    # 如果由于某种原因找不到分数，打印警告
                    print(f"警告: 找不到 {scfp_key} 的SCFP分数。将不应用稀疏惩罚。")
                    
    # [注意]：此实现仅惩罚了Attention Head。
    # 如果您的 `vision_transformer_modified.py` 也为MLP层添加了掩码，
    # 您需要在这里添加对它们的处理 (例如，使用原始的、未加权的稀疏惩罚)
    for name, param in model.named_parameters():
        if "mlp_mask" in name:
            loss_adaptive_sparse += torch.norm(param, p=2) # 示例：添加原始的L2惩罚

    # 总损失
    total_loss = loss_ce + lambda_sp * loss_adaptive_sparse + lambda_sm * loss_smooth
    
    return total_loss, loss_ce, loss_adaptive_sparse, loss_smooth
# --- RE-Pruner: 结束 ---

# 冻结模型原始权重
# 冻结除 掩码 和 分类头 以外的所有参数
for name, param in model.named_parameters():
    if "explainability_mask" in name:
        param.requires_grad = True  # 训练掩码
    elif "head" in name:
        param.requires_grad = True  # 必须训练分类头，因为它目前是随机的！
    else:
        param.requires_grad = False # 冻结主干权重

# 确认只有掩码(以及分类头)是可训练的
print("以下参数将被训练:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# 优化器只包含掩码参数(以及分类头)
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LEARNING_RATE, 
    momentum=0.9
)
ce_loss_fn = nn.CrossEntropyLoss()

# --- 6. 训练循环 ---
print("--- 开始第一阶段 (RE-Pruner 掩码学习) ---")
model.train()
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 将标签传递给模型
        outputs = model(images, y_labels=labels)
        
        # --- RE-Pruner: 调用新的损失函数 ---
        loss, loss_c, loss_as, loss_s = calculate_total_loss_re_pruner(
            model, 
            outputs, 
            labels, 
            ce_loss_fn, 
            LAMBDA_SP, 
            LAMBDA_SM,
            delta_f_scores, # 传入加载的分数
            device,         
            epsilon         
        )
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Total Loss: {loss.item():.4f}")
            print(f"  -> CE Loss: {loss_c.item():.4f}, Adaptive Sparse Loss: {loss_as.item():.4f}, Smooth Loss: {loss_s.item():.4f}")

print("第一阶段 (RE-Pruner) 训练完成!")


# 保存训练好的掩码权重


# 保存训练好的掩码权重
output_filename = f"re_pruner_phase1_masks_{NUM_CLASSES}class.pth"
print(f"正在将模型状态保存到: {output_filename} ...")
torch.save(model.state_dict(), output_filename)
print("保存成功！")
