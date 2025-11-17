import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# 导入您的模型定义
from deit_modified import deit_small_patch16_224

# --- 配置 ---
NUM_CLASSES = 100
BATCH_SIZE = 128
LR = 0.001

# !!! 修改为您的真实路径 !!!

TRAIN_PATH = "/root/autodl-tmp/imagenet100"
VAL_PATH = "/root/autodl-tmp/imagenet100_val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. 准备数据加载器
    print(f"正在加载数据集...")
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        print(f"错误: 路径不存在，请检查 TRAIN_PATH 和 VAL_PATH")
        return

    # 统一使用验证集的预处理 (Resize -> CenterCrop) 以获得稳定评估
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载训练集和验证集
    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    # 2. 创建模型
    print("正在创建模型...")
    # 我们的目标是验证 backbone 是否好用，不需要 mask 逻辑，所以直接用 forward 即可
    model = deit_small_patch16_224(pretrained=False, num_classes=NUM_CLASSES)
    
    # 3. 加载并适配预训练权重
    print("正在加载标准预训练权重并适配...")
    base_model = timm.create_model('deit_small_patch16_224', pretrained=True)
    base_state_dict = base_model.state_dict()
    
    new_state_dict = {}
    for k, v in base_state_dict.items():
        # 适配 MaskedAttention 的键名
        if '.attn.' in k:
            new_k = k.replace('.attn.', '.attn.attn.')
        else:
            new_k = k
            
        # 跳过形状不匹配的分类头
        if 'head' in k:
            if v.shape != model.state_dict()[k].shape:
                # print(f"  跳过: {k}")
                continue
        
        new_state_dict[new_k] = v
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("权重加载完成。")
    
    # 尝试加载 Phase 1 的训练模型（如果存在）
    weight_path = "re_pruner_phase1_masks_100class.pth"
    if os.path.exists(weight_path):
        print(f"加载 Phase 1 训练后的模型: {weight_path}")
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"未找到 {weight_path}，跳过加载 Phase 1 模型。")
    
    model.to(device)

    # --- 关键步骤：冻结主干，只训练分类头 (Linear Probing) ---
    print("\n--- 开始线性探测验证 (只训练分类头) ---")
    # 1. 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 2. 解冻分类头 (head)
    for param in model.head.parameters():
        param.requires_grad = True
    
    # 验证只有 head 在训练
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"当前可训练参数: {trainable_params}") # 应该只有 head.weight, head.bias

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 4. 快速训练一个 Epoch
    model.train()
    pbar = tqdm(train_loader, desc="Training Head")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, y_labels=None) # 确保不使用 mask 逻辑
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # 5. 验证准确率
    print("\n--- 验证集评估 ---")
    model.eval()
    correct_1 = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, y_labels=None)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_1 += (predicted == labels).sum().item()

    acc1 = 100 * correct_1 / total
    print("\n" + "="*40)
    print(f"验证集 Top-1 准确率: {acc1:.2f}%")
    print("="*40)
    
    if acc1 > 50.0:
        print("✅ 验证成功！预训练权重加载正确，主干网络特征提取能力正常。")
        print("提示：请务必在 main_phase1.py 中解冻 'head' 参数！")
    else:
        print("❌ 验证失败。准确率过低，请检查权重加载逻辑或数据集路径。")

if __name__ == "__main__":
    main()