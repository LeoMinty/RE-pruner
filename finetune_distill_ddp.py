import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import timm
from tqdm import tqdm
from functools import partial

from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models.vision_transformer import VisionTransformer, Block as TimmBlock, Attention as TimmAttention
from timm.layers import Mlp, DropPath

# ================= 配置区域 =================
# 1. 这里改为加载你刚刚微调过一轮的"最佳模型"，在此基础上继续冲刺
# 如果想从头练，就改回 're_pruner_PHYSICALLY_pruned.pth'
PRUNED_MODEL_PATH = "pruned_finetuned_best_ddp_0.4_89.48%.pth" 

# 2. 老师模型路径 (必须存在)
TEACHER_MODEL_PATH = "baseline_best_100class_90.96%_98.68%.pth" 

# 3. 这里的 Phase2 文件仅用于读取结构信息，不用改
PHASE2_MODEL_PATH = "re_pruner_phase2_pruned_formal_theta_100class_r0.4.pth"

BATCH_SIZE_PER_GPU = 128
# 建议再跑 50~100 轮
FINETUNE_EPOCHS = 50 
# 接续训练时，学习率可以稍微调低一点，或者保持 5e-5
BASE_LR = 2e-5  

TRAIN_PATH = "/root/autodl-tmp/imagenet100"
VAL_PATH = "/root/autodl-tmp/imagenet100_val"
NUM_CLASSES = 100
# ===========================================

# --- 1. 蒸馏 Loss (Hard Distillation - DeiT 风格) ---
class DistillationLoss(nn.Module):
    def __init__(self, base_criterion, teacher_model, dist_type='hard', alpha=0.5, tau=3.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.dist_type = dist_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        # 计算原本的 Loss (SoftTarget or CE)
        base_loss = self.base_criterion(outputs, labels)

        if self.dist_type == 'none':
            return base_loss

        # 老师推理 (不更新梯度)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.dist_type == 'soft':
            # KL 散度蒸馏
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs / T, dim=1),
                F.softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=False
            ) * (T * T) / outputs.numel()
        elif self.dist_type == 'hard':
            # Hard Label 蒸馏 (DeiT 推荐)
            # 取老师预测概率最大的类别作为 Hard Label
            teacher_labels = teacher_outputs.argmax(dim=1)
            distillation_loss = F.cross_entropy(outputs, teacher_labels)

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

# --- 2. 剪枝模型类定义 (保持修正后的正确版本) ---
class PrunedAttention(TimmAttention):
    def __init__(self, dim, num_heads, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.num_heads = num_heads
        self.head_dim = 64  # DeiT-Small 固定 head_dim
        self.scale = self.head_dim ** -0.5
        
        # 定义剪枝后的 QKV 和 Proj
        self.qkv = nn.Linear(dim, num_heads * self.head_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(num_heads * self.head_dim, dim, bias=proj_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    # [关键修复] 增加 attn_mask=None 参数以兼容 timm 接口
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        
        # 显式 Reshape 逻辑
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 如果真的传入了 mask (虽然通常是 None), 我们也简单处理一下
        if attn_mask is not None:
             attn = attn + attn_mask
             
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
        # 1. 初始化父类 (timm.models.vision_transformer.Block)
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, 
                         proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path, 
                         act_layer=act_layer, norm_layer=norm_layer)
        
        # 2. 覆盖 Attention
        # [关键修复] 只传入 Attention 需要的参数，不传 drop_path
        self.attn = PrunedAttention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            proj_bias=proj_bias, 
            attn_drop=attn_drop, 
            proj_drop=proj_drop
        )
        
        # 3. 覆盖 MLP
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=proj_drop
        )

class PrunedVisionTransformer(VisionTransformer):
    def __init__(self, head_counts_per_block, neuron_counts_per_block, **kwargs):
        super().__init__(**kwargs)
        depth = len(head_counts_per_block)
        dpr = [x.item() for x in torch.linspace(0, kwargs.get('drop_path_rate', 0.), depth)]
        # 使用 Sequential 修复 forward 问题
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

# --- 3. 工具函数 ---
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

# [新增] 计算 Top-K 准确率
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
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

    # 数据准备
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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU, 
                              sampler=train_sampler, num_workers=8, pin_memory=True, 
                              persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, 
                            sampler=val_sampler, num_workers=4, pin_memory=True)

    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, 
                     mode='batch', label_smoothing=0.1, num_classes=NUM_CLASSES)

    # 1. 实例化学生模型 (Pruned)
    h_counts, n_counts = get_model_structure(PHASE2_MODEL_PATH, 'cpu')
    student_model = PrunedVisionTransformer(
        head_counts_per_block=h_counts, neuron_counts_per_block=n_counts,
        patch_size=16, embed_dim=384, depth=12, num_classes=NUM_CLASSES,
        qkv_bias=True, proj_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    # 加载权重 (如果是接续训练，就加载 finetuned_best；如果是从头微调，加载 physically_pruned)
    if is_master: print(f"正在加载学生模型权重: {PRUNED_MODEL_PATH}")
    student_model.load_state_dict(torch.load(PRUNED_MODEL_PATH, map_location='cpu'))
    student_model.to(device)
    student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # 2. 实例化老师模型 (Teacher) - 仅在 Rank 0 打印信息，但所有进程都要加载
    if is_master: print(f"正在加载老师模型权重: {TEACHER_MODEL_PATH}")
    teacher_model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    # 注意：Teacher 不需要 DDP，只需要放在对应 GPU 上并 eval
    teacher_model.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location='cpu'))
    teacher_model.to(device)
    teacher_model.eval()
    for p in teacher_model.parameters(): p.requires_grad = False

    # 优化器
    total_batch_size = BATCH_SIZE_PER_GPU * world_size
    actual_lr = BASE_LR * (total_batch_size / 256) # 缩放学习率
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=actual_lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
    
    # Loss: 基础 Loss + 蒸馏封装
    base_criterion = SoftTargetCrossEntropy().to(device)
    # 使用 Hard Distillation (alpha=0.5 意味着一半听老师的，一半听标签的)
    criterion = DistillationLoss(base_criterion, teacher_model, dist_type='hard', alpha=0.5, tau=3.0)
    
    scaler = torch.amp.GradScaler('cuda')

    if is_master: print(f"开始蒸馏微调 (Epochs: {FINETUNE_EPOCHS}, LR: {actual_lr:.6f})")
    best_acc1 = 0.0

    for epoch in range(FINETUNE_EPOCHS):
        train_sampler.set_epoch(epoch)
        student_model.train()
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if is_master else train_loader
        
        for images, labels in iterator:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images, labels = mixup_fn(images, labels) # Mixup
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = student_model(images)
                # 蒸馏 Loss 计算会自动调用 teacher_model
                loss = criterion(images, outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if is_master and isinstance(iterator, tqdm):
                iterator.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # --- 验证 (计算 Top-1 和 Top-5) ---
        student_model.eval()
        correct_1_tensor = torch.tensor(0.0, device=device)
        correct_5_tensor = torch.tensor(0.0, device=device)
        total_tensor = torch.tensor(0.0, device=device)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = student_model(images)
                
                # 计算 Acc@1, Acc@5
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                
                correct_1_tensor += acc1[0]
                correct_5_tensor += acc5[0]
                total_tensor += labels.size(0)
        
        # 汇总所有 GPU 结果
        dist.all_reduce(correct_1_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_5_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        avg_acc1 = (correct_1_tensor / total_tensor * 100).item()
        avg_acc5 = (correct_5_tensor / total_tensor * 100).item()
        
        if is_master:
            print(f"Epoch {epoch+1}: Top-1: {avg_acc1:.2f}%, Top-5: {avg_acc5:.2f}%")
            if avg_acc1 > best_acc1:
                best_acc1 = avg_acc1
                torch.save(student_model.module.state_dict(), "pruned_distilled_best.pth")
                print(">>> 保存新的最佳模型 (Distilled) <<<")

    if is_master: print("训练结束！")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()