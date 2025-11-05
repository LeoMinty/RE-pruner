import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os

# --- 1. 配置 ---
# 必须：设置您的ImageNet训练集路径
IMAGENET_TRAIN_PATH = "/root/autodl-tmp/imagenet100" 


NUM_CLASSES = 100

# 模型参数 (deit_small)
MODEL_NAME = 'deit_small_patch16_224'
NUM_BLOCKS = 12
EMBED_DIM = 384
NUM_HEADS = 6
HEAD_DIM = EMBED_DIM // NUM_HEADS

# 训练参数
BATCH_SIZE = 64 # 根据您的GPU显存调整
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = f'scfp_head_scores_{MODEL_NAME}.pt'

# --- 2. 准备数据集 ---

# A. 标准ImageNet变换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# B. 真实数据集
print(f"加载真实数据集从: {IMAGENET_TRAIN_PATH}")
if not os.path.exists(IMAGENET_TRAIN_PATH):
    print(f"错误：ImageNet路径 {IMAGENET_TRAIN_PATH} 不存在。")
    print("请在运行此脚本前修改 'IMAGENET_TRAIN_PATH' 变量。")
    exit()
    
real_dataset = datasets.ImageFolder(IMAGENET_TRAIN_PATH, transform=transform)
real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# C. 伪 (Knockoff) 数据集
class KnockoffDataset(Dataset):
    """一个包装器，保持图像不变，但提供随机打乱的标签。"""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        # 复制并打乱标签
        self.knockoff_labels = np.random.permutation([label for _, label in original_dataset.samples])
        
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 获取原始图像
        image, _ = self.original_dataset[idx]
        # 获取打乱后的标签
        knockoff_label = self.knockoff_labels[idx]
        return image, torch.tensor(knockoff_label)

print("创建伪 (Knockoff) 数据集...")
knockoff_dataset = KnockoffDataset(real_dataset)
knockoff_loader = DataLoader(knockoff_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# --- 3. 计算费舍尔信息 (F) ---

def get_fisher_scores(model, data_loader, device):
    """计算模型中每个注意力头的经验费舍尔信息 (E[grad^2])"""
    
    # 初始化用于累积平方梯度的字典
    fisher_scores = {}
    for i in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            key = f'blocks.{i}.attn.head.{h}'
            fisher_scores[key] = 0.0

    criterion = nn.CrossEntropyLoss().to(device)
    model.train() # 确保开启训练模式以计算梯度

    num_batches = 0
    # 使用tqdm显示进度条
    pbar = tqdm(data_loader, desc="计算费舍尔信息")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 仅计算一次反向传播
        loss.backward()

        # 遍历每个Block的Attention模块
        for block_idx, block in enumerate(model.blocks):
            # 获取qkv和proj的平方梯度
            # 注意：timm库中的Attention模块默认名为 'attn'
            qkv_grad_sq = block.attn.qkv.weight.grad.pow(2)
            proj_grad_sq = block.attn.proj.weight.grad.pow(2)

            # --- 按头 (Head) 分割梯度 ---
            # qkv_grad_sq shape: (3*embed_dim, embed_dim) -> (3, num_heads, head_dim, embed_dim)
            qkv_grad_sq_view = qkv_grad_sq.view(3, NUM_HEADS, HEAD_DIM, EMBED_DIM)
            
            # proj_grad_sq shape: (embed_dim, embed_dim) -> (embed_dim, num_heads, head_dim)
            proj_grad_sq_view = proj_grad_sq.view(EMBED_DIM, NUM_HEADS, HEAD_DIM)

            for head_idx in range(NUM_HEADS):
                key = f'blocks.{block_idx}.attn.head.{head_idx}'
                
                # 累加该头对应的所有参数的平方梯度
                qkv_head_score = qkv_grad_sq_view[:, head_idx, :, :].sum()
                proj_head_score = proj_grad_sq_view[:, head_idx, :].sum()
                
                total_head_score = qkv_head_score + proj_head_score
                
                # 累加到字典中
                fisher_scores[key] += total_head_score.item()
        
        num_batches += 1
        # --- (可选) 为了节省时间，可以只运行一部分数据 ---
        # if num_batches > 1000: 
        #     print("注意：为节省时间，仅使用了部分数据...")
        #     break
        
    # 计算平均值 (期望)
    for key in fisher_scores:
        fisher_scores[key] /= num_batches
        
    return fisher_scores

# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    print(f"正在加载预训练模型: {MODEL_NAME}")
    # 1. 加载标准timm模型，不是你修改过的模型
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.to(DEVICE)
    
    # 2. 计算 F_real
    print("开始计算 F_real (真实数据)...")
    fisher_real_scores = get_fisher_scores(model, real_loader, DEVICE)
    print("F_real 计算完毕。")

    # 3. 计算 F_knockoff
    print("开始计算 F_knockoff (伪数据)...")
    fisher_knockoff_scores = get_fisher_scores(model, knockoff_loader, DEVICE)
    print("F_knockoff 计算完毕。")

    # 4. 计算 Delta F (可靠性得分)
    print("正在计算 Delta F (可靠性得分)...")
    delta_f_scores = {
        key: fisher_real_scores[key] - fisher_knockoff_scores.get(key, 0.0)
        for key in fisher_real_scores
    }

    # 打印一些示例得分
    print("\n--- 示例可靠性得分 (Delta F) ---")
    for i in range(min(5, len(delta_f_scores))):
        key = list(delta_f_scores.keys())[i]
        print(f"{key}: {delta_f_scores[key]}")

    # 5. 保存结果
    print(f"\n正在将得分保存到: {OUTPUT_FILE}")
    torch.save(delta_f_scores, OUTPUT_FILE)
    print("第一步完成！SCFP可靠性得分已成功保存。")