import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler # 混合精度
import os
import timm
from tqdm import tqdm
from functools import partial

# --- timm 导入 ---
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models.vision_transformer import VisionTransformer, Block as TimmBlock, Attention as TimmAttention
from timm.layers import DropPath, Mlp
RATE = 0.4
# ================= 配置区域 =================
# 注意：在DDP中，BATCH_SIZE 是指"每张卡"的大小
# 如果你有 4 张卡，每张 128，总 Batch Size 就是 512
BATCH_SIZE_PER_GPU = 128 
FINETUNE_EPOCHS = 50       # 建议跑 50 轮以上
BASE_LR = 5e-5             # 基础学习率 (会自动根据卡数缩放)

# 路径
PRUNED_MODEL_PATH = f"re_pruner_PHYSICALLY_pruned_r{RATE}.pth"
PHASE2_MODEL_PATH = f"re_pruner_phase2_pruned_formal_theta_100class_r{RATE}.pth" 
TRAIN_PATH = "/root/autodl-tmp/imagenet100"
VAL_PATH = "/root/autodl-tmp/imagenet100_val"
NUM_CLASSES = 100
# ===========================================

# --- 1. 定义剪枝模型类 (必须与 prune_model.py 一致) ---
class PrunedAttention(TimmAttention):
    def __init__(self, dim, num_heads, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        # 初始化父类，但注意我们后续会覆盖 qkv 和 proj
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.num_heads = num_heads
        # DeiT-Small 固定 head_dim 为 64
        self.head_dim = 64 
        self.scale = self.head_dim ** -0.5

        # [关键] 定义剪枝后的 QKV 和 Proj 层
        # 输出维度 = 3 * num_heads * 64
        self.qkv = nn.Linear(dim, num_heads * self.head_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(num_heads * self.head_dim, dim, bias=proj_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # [关键修复] 显式使用 self.head_dim = 64 进行 reshape
        # 原始 dim 是 384，但剪枝后 qkv 输出维度只有 num_heads * 64 * 3
        # 所以必须用 (3, self.num_heads, self.head_dim) 来拆解
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
    # 必须接收 mlp_hidden_dim
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
        
        # 使用 mlp_hidden_dim
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
        super().__init__(**kwargs)
        depth = len(head_counts_per_block)
        dpr = [x.item() for x in torch.linspace(0, kwargs.get('drop_path_rate', 0.), depth)]
        self.blocks = nn.Sequential(*[
            PrunedBlock(
                dim=kwargs['embed_dim'], 
                num_heads=head_counts_per_block[i], 
                mlp_hidden_dim=neuron_counts_per_block[i],
                qkv_bias=True, 
                proj_bias=True, 
                drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            ) for i in range(depth)
        ])
        self.apply(self._init_weights)

# --- 2. DDP 初始化工具 ---
def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"DDP Init: Rank {rank}, Local {local_rank}, World {world_size}")
        return rank, local_rank, world_size
    else:
        print("未检测到 DDP 环境，请使用 torchrun 启动。")
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- 3. 辅助函数：计算剪枝结构 ---
def get_model_structure(phase2_path, device):
    state_dict = torch.load(phase2_path, map_location=device)
    heads, neurons = [], []
    for i in range(12):
        # Calc Heads
        theta_a = state_dict[f'blocks.{i}.attn.theta'].item()
        mask_a = state_dict[f'blocks.{i}.attn.explainability_mask']
        imp_a = mask_a.mean(0).abs().sum(-1)
        h_kept = torch.nonzero(imp_a > theta_a).numel()
        heads.append(max(h_kept, 1))
        # Calc Neurons
        theta_m = state_dict[f'blocks.{i}.mlp.theta'].item()
        mask_m = state_dict[f'blocks.{i}.mlp.explainability_mask']
        imp_m = mask_m.mean(0).abs()
        n_kept = torch.nonzero(imp_m > theta_m).numel()
        neurons.append(max(n_kept, 1))
    return heads, neurons

# --- 主函数 ---
def main():
    # A. 环境初始化
    torch.backends.cudnn.benchmark = True # [加速] 开启 CuDNN Benchmark
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)

    # B. 数据准备
    transform_train = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform_train)
    val_dataset = datasets.ImageFolder(VAL_PATH, transform=transform_val)

    # DDP Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_PER_GPU, 
        shuffle=False, sampler=train_sampler, 
        num_workers=8, pin_memory=True, persistent_workers=True, # [加速] Dataloader 优化
        drop_last=True # [关键] 解决 Mixup 报错 (Batch size must be even)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_PER_GPU, 
        shuffle=False, sampler=val_sampler, 
        num_workers=4, pin_memory=True
    )

    # C. Mixup 配置
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, 
        mode='batch', label_smoothing=0.1, num_classes=NUM_CLASSES
    )

    # D. 模型构建
    if is_master: print("正在重建并加载模型...")
    h_counts, n_counts = get_model_structure(PHASE2_MODEL_PATH, 'cpu')
    if is_master: print(f"Heads: {h_counts}\nNeurons: {n_counts}")

    model = PrunedVisionTransformer(
        head_counts_per_block=h_counts, neuron_counts_per_block=n_counts,
        patch_size=16, embed_dim=384, depth=12, num_classes=NUM_CLASSES,
        qkv_bias=True, proj_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    # 加载权重
    model.load_state_dict(torch.load(PRUNED_MODEL_PATH, map_location='cpu'))
    model.to(device)

    # 包装 DDP
    # find_unused_parameters=False 是加速的关键，因为微调时所有参数都参与计算
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # E. 优化器 & Loss
    # 线性缩放学习率: Base_LR * Total_Batch / 256 (假设基准)
    total_batch_size = BATCH_SIZE_PER_GPU * world_size
    actual_lr = BASE_LR * (total_batch_size / 128) # 根据你的基准调整
    if is_master: print(f"Total Batch: {total_batch_size}, Actual LR: {actual_lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
    
    criterion_train = SoftTargetCrossEntropy().to(device)
    criterion_val = nn.CrossEntropyLoss().to(device)
    
    # [加速] 混合精度 Scaler
    scaler = torch.amp.GradScaler('cuda')

    # F. 训练循环
    if is_master: print(f"开始 DDP 微调 (Epochs: {FINETUNE_EPOCHS})...")
    best_acc = 0.0

    for epoch in range(FINETUNE_EPOCHS):
        # DDP 必须设置 epoch 以保证 shuffle 正确
        train_sampler.set_epoch(epoch)
        model.train()
        
        # 只在主进程显示进度条
        if is_master:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{FINETUNE_EPOCHS}")
            iterator = pbar
        else:
            iterator = train_loader
            
        for images, labels in iterator:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixup
            images, labels = mixup_fn(images, labels)
            
            optimizer.zero_grad()
            
            # [加速] 混合精度前向
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion_train(outputs, labels)
            
            # [加速] 混合精度反向
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if is_master and isinstance(iterator, tqdm):
                iterator.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # --- 验证 ---
        model.eval()
        correct_tensor = torch.tensor(0.0, device=device)
        total_tensor = torch.tensor(0.0, device=device)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct_tensor += (preds == labels).sum()
                total_tensor += labels.size(0)
        
        # DDP 汇总所有卡的结果
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        acc = (correct_tensor / total_tensor * 100).item()
        
        if is_master:
            print(f"Epoch {epoch+1} Val Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                # 注意保存 model.module
                torch.save(model.module.state_dict(), f"pruned_finetuned_best_ddp_{RATE}.pth")
                print("已保存最佳模型!")

    if is_master: print("微调结束!")
    cleanup_ddp()

if __name__ == "__main__":
    main()