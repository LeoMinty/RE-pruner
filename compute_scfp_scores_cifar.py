# compute_scfp_scores_cifar.py
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
import argparse

# 导入 CIFAR 工具
from utils_cifar import get_cifar10_datasets

# --- 配置 ---
NUM_CLASSES = 10
MODEL_NAME = 'deit_small_patch16_224'
NUM_BLOCKS = 12
EMBED_DIM = 384
NUM_HEADS = 6
HEAD_DIM = EMBED_DIM // NUM_HEADS

BATCH_SIZE = 64 # CIFAR resize后显存占用较大，适当减小 BatchSize
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILE = f'scfp_head_scores_cifar10_{MODEL_NAME}.pt'

# --- 伪 (Knockoff) 数据集 ---
class KnockoffDatasetCifar(Dataset):
    """适配 CIFAR-10 的 Knockoff Dataset"""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        # CIFAR10 dataset 使用 .targets 存储标签
        self.knockoff_labels = np.random.permutation(original_dataset.targets)
        
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, _ = self.original_dataset[idx]
        knockoff_label = self.knockoff_labels[idx]
        return image, torch.tensor(knockoff_label).long()

# --- 计算费舍尔信息 (逻辑保持不变) ---
def get_fisher_scores(model, data_loader, device):
    fisher_scores = {}
    # 初始化
    for i in range(NUM_BLOCKS):
        for h in range(NUM_HEADS):
            fisher_scores[f'blocks.{i}.attn.head.{h}'] = 0.0
        # MLP Neurons
        hidden_dim = model.blocks[i].mlp.fc1.out_features
        for n in range(hidden_dim):
            fisher_scores[f'blocks.{i}.mlp.neuron.{n}'] = 0.0
            
    criterion = nn.CrossEntropyLoss().to(device)
    model.train() 

    # 限制 Batch 数量以加快计算 (可选，全量计算更准)
    MAX_BATCHES = 200 
    num_batches = 0
    
    pbar = tqdm(data_loader, desc="Calculating Fisher Info")
    for images, labels in pbar:
        if num_batches >= MAX_BATCHES: break
        
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        for block_idx, block in enumerate(model.blocks):
            # Attention Heads
            qkv_grad_sq = block.attn.qkv.weight.grad.pow(2)
            proj_grad_sq = block.attn.proj.weight.grad.pow(2)
            qkv_view = qkv_grad_sq.view(3, NUM_HEADS, HEAD_DIM, EMBED_DIM)
            proj_view = proj_grad_sq.view(EMBED_DIM, NUM_HEADS, HEAD_DIM)

            for h in range(NUM_HEADS):
                score = qkv_view[:, h, :, :].sum() + proj_view[:, h, :].sum()
                fisher_scores[f'blocks.{block_idx}.attn.head.{h}'] += score.item()

            # MLP Neurons
            fc1_grad_sq = block.mlp.fc1.weight.grad.pow(2).sum(dim=1)
            fc2_grad_sq = block.mlp.fc2.weight.grad.pow(2).sum(dim=0)
            neuron_scores = fc1_grad_sq + fc2_grad_sq
            
            for n in range(len(neuron_scores)):
                fisher_scores[f'blocks.{block_idx}.mlp.neuron.{n}'] += neuron_scores[n].item()

        num_batches += 1

    for key in fisher_scores:
        fisher_scores[key] /= num_batches
    return fisher_scores

if __name__ == "__main__":
    print(f"Loading Model: {MODEL_NAME}")
    # 注意：这里加载 ImageNet 预训练权重。
    # 理想情况下，计算 Fisher Score 最好使用在 CIFAR-10 上 Finetune 过的模型。
    # 但为了流程简单，我们暂时使用预训练权重，只需重置分类头。
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES) # 重置为10类
    model.to(DEVICE)
    
    # 获取数据集
    real_dataset, _ = get_cifar10_datasets()
    knockoff_dataset = KnockoffDatasetCifar(real_dataset)
    
    real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    knockoff_loader = DataLoader(knockoff_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print("Calculating F_real...")
    fisher_real = get_fisher_scores(model, real_loader, DEVICE)
    
    print("Calculating F_knockoff...")
    fisher_knockoff = get_fisher_scores(model, knockoff_loader, DEVICE)

    print("Calculating Delta F...")
    delta_f = {k: fisher_real[k] - fisher_knockoff.get(k, 0.0) for k in fisher_real}

    torch.save(delta_f, OUTPUT_FILE)
    print(f"Scores saved to {OUTPUT_FILE}")