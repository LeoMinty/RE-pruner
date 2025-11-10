# prune_model.py
import torch
from torch import nn
import timm
import os
from collections import OrderedDict
# 导入 *原始* 的 ViT 和 Block, Attention
from timm.models.vision_transformer import VisionTransformer, Block, Attention
# 导入我们修改过的 DeiT (仅用于加载)
from deit_modified import deit_small_patch16_224

# --- 配置 ---
#第二阶段输出的模型文件
PHASE2_MODEL_PATH = "re_pruner_phase2_pruned_formal_r_logit_100class_r0.5.pth"
#最终物理剪枝模型的保存路径
FINAL_MODEL_PATH = "re_pruner_PHYSICALLY_pruned.pth"

NUM_CLASSES = 100
NUM_BLOCKS = 12
BASE_NUM_HEADS = 6 # DeiT-Small
BASE_EMBED_DIM = 384
HEAD_DIM = BASE_EMBED_DIM // BASE_NUM_HEADS

device = torch.device("cpu") # 在CPU上操作

# --- 1. 加载第二阶段训练好的模型状态 ---
print(f"正在从 {PHASE2_MODEL_PATH} 加载剪枝模型状态...")
if not os.path.exists(PHASE2_MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {PHASE2_MODEL_PATH} 不存在。")
    
state_dict = torch.load(PHASE2_MODEL_PATH, map_location=device)

# --- 2. 计算每层要保留的头的索引 ---
print("--- 正在计算要保留的注意力头 (结构化) ---")
pruning_config = {}
total_heads_before = 0
total_heads_after = 0

for i in range(NUM_BLOCKS):
    # 加载该 block 的 r_logit 和 explainability_mask
    r_logit = state_dict[f'blocks.{i}.attn.r_logit']
    mask_scores = state_dict[f'blocks.{i}.attn.explainability_mask']
    
    # mask_scores shape: [C, H, D_h]
    num_heads = mask_scores.shape[1]
    
    # 1. 计算剪枝率 r
    r = torch.sigmoid(r_logit).item()
    num_heads_to_keep = int(round((1.0 - r) * num_heads))
    num_heads_to_keep = max(1, num_heads_to_keep) # 至少保留1个
    
    # 2. 计算每个头的 "重要性分数"
    head_importance = mask_scores.mean(dim=0).sum(dim=1) # Shape: [num_heads]
    
    # 3. 找到分数最高的头的索引
    _, top_k_indices = torch.topk(head_importance, k=num_heads_to_keep)
    top_k_indices = sorted(top_k_indices.tolist()) # 排序以方便权重复制
    
    pruning_config[i] = top_k_indices # 存储索引
    
    print(f"Block {i}: 目标剪枝率 r={r:.4f}。保留 {num_heads_to_keep}/{num_heads} 个头。保留索引: {top_k_indices}")
    total_heads_before += num_heads
    total_heads_after += num_heads_to_keep

print(f"\n总计：剪枝前共 {total_heads_before} 个头, 剪枝后保留 {total_heads_after} 个头。")
print(f"头部参数稀疏度: {(total_heads_before - total_heads_after) / total_heads_before:.2%}")

# --- 3. 创建一个新的、物理上更小的模型并复制权重 ---
print("\n--- 正在创建并填充 *物理上* 剪枝后的模型 ---")

# a. 创建一个自定义的 ViT，其 Attention 头的数量是可变的
class PrunedVisionTransformer(VisionTransformer):
    def __init__(self, head_counts_per_block, **kwargs):
        # 必须确保 depth 参数匹配
        kwargs['depth'] = len(head_counts_per_block)
        super().__init__(**kwargs)
        
        # 销毁原始 blocks
        del self.blocks
        
        # 重建 blocks, 每个 block 有不同数量的头
        dpr = [x.item() for x in torch.linspace(0, kwargs.get('drop_path_rate', 0.), kwargs['depth'])]
        self.blocks = nn.ModuleList([
            Block(
                dim=kwargs['embed_dim'], 
                num_heads=head_counts_per_block[i], # <-- 关键：使用自定义的头数量
                mlp_ratio=kwargs.get('mlp_ratio', 4.),
                qkv_bias=kwargs.get('qkv_bias', True),
                drop_path=dpr[i],
                norm_layer=kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
            )
            for i in range(kwargs['depth'])
        ])
        
        # 重新初始化权重 (仅用于结构)
        self.apply(self._init_weights)


# b. 实例化这个新模型
new_head_counts = [len(pruning_config[i]) for i in range(NUM_BLOCKS)]
pruned_model = PrunedVisionTransformer(
    head_counts_per_block=new_head_counts,
    patch_size=16,
    embed_dim=BASE_EMBED_DIM,
    depth=NUM_BLOCKS, # 必须匹配
    num_heads=BASE_NUM_HEADS, # 这是一个占位符, 实际头数在内部
    num_classes=NUM_CLASSES,
)
pruned_model.eval()
pruned_state_dict = pruned_model.state_dict()

# c. 复制权重 (最关键的部分)
new_state_dict = OrderedDict()
for (old_name, old_param) in state_dict.items():
    if "attn" not in old_name:
        # 1. 复制非注意力参数 (patch_embed, cls_token, norm, head, mlp)
        if old_name in pruned_state_dict and pruned_state_dict[old_name].shape == old_param.shape:
            new_state_dict[old_name] = old_param
    else:
        # 2. 复制 *注意力* 参数 (结构化切片)
        if "attn.qkv.weight" in old_name:
            block_idx = int(old_name.split('.')[1])
            indices_to_keep = pruning_config[block_idx]
            num_heads_to_keep = len(indices_to_keep)
            
            # 原始 QKV 权重: [3 * D, D] = [3 * H * D_h, D]
            # 新 QKV 权重: [3 * H_new * D_h, D]
            
            # [3*H*D_h, D] -> [3, H, D_h, D]
            old_qkv = old_param.view(3, BASE_NUM_HEADS, HEAD_DIM, BASE_EMBED_DIM)
            
            # 选择要保留的头
            new_qkv = old_qkv[:, indices_to_keep, :, :] # Shape: [3, k, D_h, D]
            
            # 重塑回 [3*k*D_h, D]
            new_qkv = new_qkv.reshape(3 * num_heads_to_keep * HEAD_DIM, BASE_EMBED_DIM)
            new_state_dict[old_name] = new_qkv

        elif "attn.qkv.bias" in old_name:
            block_idx = int(old_name.split('.')[1])
            indices_to_keep = pruning_config[block_idx]
            num_heads_to_keep = len(indices_to_keep)
            
            # [3*H*D_h] -> [3, H, D_h]
            old_bias = old_param.view(3, BASE_NUM_HEADS, HEAD_DIM)
            new_bias = old_bias[:, indices_to_keep, :] # Shape: [3, k, D_h]
            new_bias = new_bias.reshape(3 * num_heads_to_keep * HEAD_DIM)
            new_state_dict[old_name] = new_bias

        elif "attn.proj.weight" in old_name:
            block_idx = int(old_name.split('.')[1])
            indices_to_keep = pruning_config[block_idx]
            num_heads_to_keep = len(indices_to_keep)
            
            # 原始 Proj 权重: [D, D] = [D, H * D_h]
            # 新 Proj 权重: [D, H_new * D_h]
            
            # [D, H*D_h] -> [D, H, D_h]
            old_proj = old_param.view(BASE_EMBED_DIM, BASE_NUM_HEADS, HEAD_DIM)
            
            # 选择要保留的头
            new_proj = old_proj[:, indices_to_keep, :] # Shape: [D, k, D_h]
            
            # 重塑回 [D, k*D_h]
            new_proj = new_proj.reshape(BASE_EMBED_DIM, num_heads_to_keep * HEAD_DIM)
            new_state_dict[old_name] = new_proj

        elif "attn.proj.bias" in old_name:
            # Proj 的偏置 不变，因为它的大小是 [D]
            new_state_dict[old_name] = old_param
            
        # 忽略所有 'explainability_mask', 'r_logit', 'theta'
        
# d. 加载新的状态字典
pruned_model.load_state_dict(new_state_dict)
torch.save(pruned_model.state_dict(), FINAL_MODEL_PATH)
print(f"\n物理剪枝后的模型已保存到 {FINAL_MODEL_PATH}")
print("这个模型现在物理上更小了，可以用于微调和FLOPs分析。")