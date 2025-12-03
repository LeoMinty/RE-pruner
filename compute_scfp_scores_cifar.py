# compute_scfp_scores_cifar.py
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
from utils_cifar import get_cifar10_datasets # <--- 引入工具

# --- 1. 配置 ---
NUM_CLASSES = 10 # <--- 修改为 10
# 模型参数 (deit_small)
MODEL_NAME = 'deit_small_patch16_224'
NUM_BLOCKS = 12
EMBED_DIM = 384
NUM_HEADS = 6
HEAD_DIM = EMBED_DIM // NUM_HEADS

# 训练参数
BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = f'scfp_head_scores_cifar10_{MODEL_NAME}.pt' # <--- 修改文件名

# --- 2. 准备数据集 ---
# 使用 utils_cifar 获取数据集
real_dataset, _ = get_cifar10_datasets()
real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# C. 伪 (Knockoff) 数据集
class KnockoffDatasetCIFAR(Dataset):
    """适配 CIFAR-10 的 Knockoff Dataset"""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        # CIFAR-10 使用 .targets
        self.knockoff_labels = np.random.permutation(original_dataset.targets)
        
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, _ = self.original_dataset[idx]
        knockoff_label = self.knockoff_labels[idx]
        return image, torch.tensor(knockoff_label)

print("创建伪 (Knockoff) 数据集...")
knockoff_dataset = KnockoffDatasetCIFAR(real_dataset)
knockoff_loader = DataLoader(knockoff_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# --- 3. 计算费舍尔信息 (F) ---
# (此函数保持原样，未做修改)
def get_fisher_scores(model, data_loader, device):
    """计算模型中每个注意力头的经验费舍尔信息 (E[grad^2])"""
    
    fisher_scores = {}
    for i in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            key = f'blocks.{i}.attn.head.{h}'
            fisher_scores[key] = 0.0
        # MLP Neurons
        # 此时模型已在 device 上，直接获取
        hidden_dim = model.blocks[i].mlp.fc1.out_features
        for n in range(hidden_dim):
            fisher_scores[f'blocks.{i}.mlp.neuron.{n}'] = 0.0
            
    criterion = nn.CrossEntropyLoss().to(device)
    model.train() 

    num_batches = 0
    pbar = tqdm(data_loader, desc="计算费舍尔信息")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()

        for block_idx, block in enumerate(model.blocks):
            qkv_grad_sq = block.attn.qkv.weight.grad.pow(2)
            proj_grad_sq = block.attn.proj.weight.grad.pow(2)

            qkv_grad_sq_view = qkv_grad_sq.view(3, NUM_HEADS, HEAD_DIM, EMBED_DIM)
            proj_grad_sq_view = proj_grad_sq.view(EMBED_DIM, NUM_HEADS, HEAD_DIM)

            for head_idx in range(NUM_HEADS):
                key = f'blocks.{block_idx}.attn.head.{head_idx}'
                qkv_head_score = qkv_grad_sq_view[:, head_idx, :, :].sum()
                proj_head_score = proj_grad_sq_view[:, head_idx, :].sum()
                total_head_score = qkv_head_score + proj_head_score
                fisher_scores[key] += total_head_score.item()
            
            # MLP
            fc1_grad_sq = block.mlp.fc1.weight.grad.pow(2) 
            fc2_grad_sq = block.mlp.fc2.weight.grad.pow(2) 
            fc1_neuron_scores = fc1_grad_sq.sum(dim=1)
            fc2_neuron_scores = fc2_grad_sq.sum(dim=0)
            total_neuron_scores = fc1_neuron_scores + fc2_neuron_scores

            hidden_dim_curr = total_neuron_scores.shape[0]
            for n in range(hidden_dim_curr):
                key = f'blocks.{block_idx}.mlp.neuron.{n}'
                fisher_scores[key] += total_neuron_scores[n].item()

        num_batches += 1
        
    for key in fisher_scores:
        fisher_scores[key] /= num_batches
        
    return fisher_scores

# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    print(f"正在加载预训练模型: {MODEL_NAME}")
    model = timm.create_model(MODEL_NAME, pretrained=True)
    # <--- 关键：重置 Head 以匹配 CIFAR-10 的 10 类，防止 shape mismatch 报错
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
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

    print("\n--- 示例可靠性得分 (Delta F) ---")
    for i in range(min(5, len(delta_f_scores))):
        key = list(delta_f_scores.keys())[i]
        print(f"{key}: {delta_f_scores[key]}")

    print(f"\n正在将得分保存到: {OUTPUT_FILE}")
    torch.save(delta_f_scores, OUTPUT_FILE)
    print("SCFP可靠性得分已成功保存。")