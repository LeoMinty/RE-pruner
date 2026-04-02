import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import timm
from tqdm import tqdm
from functools import partial

# --- 新增的高级训练库 ---
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2 # EMA
from timm.data import create_transform # 自动增强
from timm.models.vision_transformer import VisionTransformer, Block as TimmBlock, Attention as TimmAttention
from timm.layers import Mlp, DropPath

# ================= 配置区域 =================
PRUNED_MODEL_PATH = "re_pruner_physically_pruned_cifar10_r0.33.pth" 
PHASE2_MODEL_PATH = "re_pruner_phase2_cifar10_r0.33.pth"

FINETUNE_EPOCHS = 140 
BATCH_SIZE_PER_GPU = 128
BASE_LR = 5e-4 
WEIGHT_DECAY = 0.05

# [修改1] CIFAR-10 数据下载/存储路径
DATA_PATH = "./data_cifar"
# [修改2] CIFAR-10 类别数
NUM_CLASSES = 10
# ===========================================

# --- 修正后的类定义 (保持不变) ---
class PrunedAttention(TimmAttention):
    def __init__(self, dim, num_heads, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.num_heads = num_heads
        self.head_dim = 64
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, num_heads * self.head_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(num_heads * self.head_dim, dim, bias=proj_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None: attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PrunedBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, proj_bias=True, 
                 proj_drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, 
                         proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path, 
                         act_layer=act_layer, norm_layer=norm_layer)
        self.attn = PrunedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

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
                qkv_bias=True, proj_bias=True, drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            ) for i in range(depth)
        ])
        self.apply(self._init_weights)

def get_model_structure(phase2_path, device):
    state_dict = torch.load(phase2_path, map_location=device)
    heads, neurons = [], []
    for i in range(12):
        theta_a = state_dict[f'blocks.{i}.attn.theta'].item()
        mask_a = state_dict[f'blocks.{i}.attn.explainability_mask']
        imp_a = mask_a.mean(0).abs().sum(-1)
        heads.append(max(torch.nonzero(imp_a > theta_a).numel(), 1))
        theta_m = state_dict[f'blocks.{i}.mlp.theta'].item()
        mask_m = state_dict[f'blocks.{i}.mlp.explainability_mask']
        imp_m = mask_m.mean(0).abs()
        neurons.append(max(torch.nonzero(imp_m > theta_m).numel(), 1))
    return heads, neurons

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def setup_ddp():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size

# --- 主函数 ---
def main():
    torch.backends.cudnn.benchmark = True
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_master = (rank == 0)

    # [修改3] 虽然是 CIFAR，但模型是 ViT-S，必须强制 Resize 到 224
    train_transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1', 
        interpolation='bicubic',
        re_prob=0.25, 
        re_mode='pixel',
        re_count=1,
        mean=(0.485, 0.456, 0.406), # 保持 ImageNet 均值方差配合预训练权重
        std=(0.229, 0.224, 0.225),
    )
    
    val_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # [修改4] 使用 torchvision.datasets.CIFAR10 自动下载和加载数据
    train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=False, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=False, transform=val_transform)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU, 
                              sampler=train_sampler, num_workers=8, pin_memory=True, 
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, 
                            sampler=val_sampler, num_workers=4, pin_memory=True)

    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, 
                     mode='batch', label_smoothing=0.1, num_classes=NUM_CLASSES)

    h_counts, n_counts = get_model_structure(PHASE2_MODEL_PATH, 'cpu')
    model = PrunedVisionTransformer(
        head_counts_per_block=h_counts, neuron_counts_per_block=n_counts,
        patch_size=16, embed_dim=384, depth=12, num_classes=NUM_CLASSES,
        qkv_bias=True, proj_bias=True, 
        drop_path_rate=0.1, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    if is_master: print(f"加载权重: {PRUNED_MODEL_PATH}")
    model.load_state_dict(torch.load(PRUNED_MODEL_PATH, map_location='cpu'))
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    model_ema = ModelEmaV2(model, decay=0.9995, device=None)
    
    if is_master:
        print("Model EMA 已启用(所有进程)")

    total_batch_size = BATCH_SIZE_PER_GPU * world_size
    actual_lr = BASE_LR * (total_batch_size / 512) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    criterion = SoftTargetCrossEntropy().to(device)
    scaler = torch.amp.GradScaler('cuda')

    if is_master: print(f"开始高级微调 (Epochs: {FINETUNE_EPOCHS}, EMA: Yes, RandAug: Yes)")
    best_acc1 = 0.0

    for epoch in range(FINETUNE_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if is_master else train_loader
        
        for images, labels in iterator:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images, labels = mixup_fn(images, labels)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if model_ema is not None:
                model_ema.update(model)
            
            if is_master and isinstance(iterator, tqdm):
                iterator.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        def run_validate(val_model, name):
            val_model.eval()
            c1, c5, tot = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = val_model(images)
                    acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                    c1 += acc1[0]; c5 += acc5[0]; tot += labels.size(0)
            dist.all_reduce(c1); dist.all_reduce(c5); dist.all_reduce(tot)
            return (c1/tot*100).item(), (c5/tot*100).item()

        acc1, acc5 = run_validate(model, "Normal")
        
        ema_acc1, ema_acc5 = 0.0, 0.0
        if model_ema is not None:
             ema_acc1, ema_acc5 = run_validate(model_ema.module, "EMA")

        if is_master:
            print(f"Epoch {epoch+1}:")
            print(f"  [Model] Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%")
            if model_ema:
                print(f"  [EMA]   Top-1: {ema_acc1:.2f}%, Top-5: {ema_acc5:.2f}%")
            
            save_acc = max(acc1, ema_acc1)
            if save_acc > best_acc1:
                best_acc1 = save_acc
                torch.save(model.module.state_dict(), "pruned_finetuned_best(cifar).pth")
                if model_ema:
                    torch.save(model_ema.module.state_dict(), "pruned_finetuned_best_ema(cifar).pth")
                print(">>> 保存新的最佳模型 <<<")

    if is_master: print("训练结束！")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()