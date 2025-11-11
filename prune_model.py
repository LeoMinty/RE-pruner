# prune_model.py
import torch
from torch import nn
import timm
import os
from collections import OrderedDict
from functools import partial # <-- 导入 partial

# 导入 *原始* 的 ViT 基类
from timm.models.vision_transformer import VisionTransformer, Block as TimmBlock, Attention as TimmAttention
from timm.layers import DropPath, Mlp, to_2tuple # <-- 导入 DropPath 和 Mlp

# --- 配置 ---
PHASE2_MODEL_PATH = "re_pruner_phase2_pruned_formal_r_logit_100class_r0.5.pth"
FINAL_MODEL_PATH = "re_pruner_PHYSICALLY_pruned.pth"

NUM_CLASSES = 100
NUM_BLOCKS = 12
BASE_NUM_HEADS = 6 # DeiT-Small
BASE_EMBED_DIM = 384
HEAD_DIM = BASE_EMBED_DIM // BASE_NUM_HEADS

device = torch.device("cpu")

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
    r_logit = state_dict[f'blocks.{i}.attn.r_logit']
    mask_scores = state_dict[f'blocks.{i}.attn.explainability_mask']
    
    num_heads = mask_scores.shape[1]
    r = torch.sigmoid(r_logit).item()
    num_heads_to_keep = int(round((1.0 - r) * num_heads))
    num_heads_to_keep = max(1, num_heads_to_keep) 
    
    head_importance = mask_scores.mean(dim=0).sum(dim=1)
    _, top_k_indices = torch.topk(head_importance, k=num_heads_to_keep)
    top_k_indices = sorted(top_k_indices.tolist()) 
    
    pruning_config[i] = top_k_indices 
    
    print(f"Block {i}: 目标剪枝率 r={r:.4f}。保留 {num_heads_to_keep}/{num_heads} 个头。保留索引: {top_k_indices}")
    total_heads_before += num_heads
    total_heads_after += num_heads_to_keep

print(f"\n总计：剪枝前共 {total_heads_before} 个头, 剪枝后保留 {total_heads_after} 个头。")
print(f"头部参数稀疏度: {(total_heads_before - total_heads_after) / total_heads_before:.2%}")

# --- 3. 创建一个新的、物理上更小的模型并复制权重 ---
print("\n--- 正在创建并填充 *物理上* 剪枝后的模型 ---")

# 计算每层最终保留的头数量（按顺序），用于构建物理上剪枝后的模型
new_head_counts = [len(pruning_config[i]) for i in range(NUM_BLOCKS)]
print(f"每层保留的头数: {new_head_counts}")

# --- *** ---
# !!! 关键修复点：定义我们自己的 PrunedAttention 和 PrunedBlock !!!
# --- *** ---

class PrunedAttention(TimmAttention):
    """
    一个继承自 timm Attention 的类，
    但在初始化时创建 *物理上更小* 的 QKV 和 Proj 层。
    """
    def __init__(self, dim, num_heads, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        # 必须调用 nn.Module 的 __init__，而不是 TimmAttention 的
        # 因为 TimmAttention 的 __init__ 会创建我们不想要的层
        super(TimmAttention, self).__init__() 
        
        self.num_heads = num_heads
        self.head_dim = HEAD_DIM # 关键：head_dim 是固定的
        self.scale = self.head_dim ** -0.5
        
        # 关键：qkv 和 proj 的维度依赖于 *新* 的 num_heads
        # qkv: [D, 3 * k * D_h]
        self.qkv = nn.Linear(dim, (num_heads * self.head_dim) * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # proj: [k * D_h, D]
        self.proj = nn.Linear(num_heads * self.head_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # forward 方法与 timm.Attention 保持一致
        B, N, C = x.shape
        # qkv: [B, N, 3 * k * D_h] -> [3, B, H, N, D_h]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # x: [B, H, N, D_h] -> [B, N, H*D_h]
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) # C_new = num_heads * head_dim
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PrunedBlock(TimmBlock):
    """
    一个继承自 timm Block 的类，
    但确保它使用我们自定义的 PrunedAttention。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_bias=True,
                 proj_drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # 同样，只调用 nn.Module 的 __init__
        super(TimmBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        # 关键：使用我们自定义的 PrunedAttention
        self.attn = PrunedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
            attn_drop=attn_drop, proj_drop=proj_drop)
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 关键：使用 timm.Mlp 并正确传递参数
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 重写 forward 以匹配 PrunedAttention (它不接受 y_labels 或 attn_mask)
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class PrunedVisionTransformer(VisionTransformer):
    """
    一个继承自 timm VisionTransformer 的类，
    但使用我们自定义的 PrunedBlock 列表。
    """
    def __init__(self, head_counts_per_block, **kwargs):
        # 提取 kwargs *之前* 调用 super()
        depth = len(head_counts_per_block)
        drop_path_rate = kwargs.get('drop_path_rate', 0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        mlp_ratio = kwargs.get('mlp_ratio', 4.)
        qkv_bias = kwargs.get('qkv_bias', True)
        proj_bias = kwargs.get('proj_bias', True) 
        attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
        proj_drop_rate = kwargs.get('drop_rate', 0.) 
        norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
        act_layer = kwargs.get('act_layer', nn.GELU)

        # 准备 kwargs for super()
        super_kwargs = kwargs.copy()
        super_kwargs['depth'] = depth
        super_kwargs['num_heads'] = BASE_NUM_HEADS # 占位符
        super().__init__(**super_kwargs)
        
        # 销毁 super() 创建的 blocks
        del self.blocks
        
        # 重建 blocks
        self.blocks = nn.ModuleList([
            PrunedBlock( 
                dim=kwargs['embed_dim'], 
                num_heads=head_counts_per_block[i], # <-- 传入每层的新头数量
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        self.apply(self._init_weights)
    
    def forward_features(self, x, attn_mask=None):
        """重写 forward_features 以正确处理 ModuleList blocks"""
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x

# --- 实例化新模型 ---
pruned_model = PrunedVisionTransformer(
    head_counts_per_block=new_head_counts,
    patch_size=16,
    embed_dim=BASE_EMBED_DIM,
    depth=NUM_BLOCKS, 
    num_classes=NUM_CLASSES,
    qkv_bias=True, 
    proj_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    act_layer=nn.GELU,
    drop_rate=0.0, # proj_drop
    attn_drop_rate=0.0
)
pruned_model.eval()
pruned_state_dict = pruned_model.state_dict()

# --- 复制权重 (现在键名和形状应该匹配了) ---
new_state_dict = OrderedDict()
for (old_name, old_param) in state_dict.items():
    
    # 1. 重命名键
    if ".attn.attn." in old_name:
        new_name = old_name.replace(".attn.attn.", ".attn.", 1)
    else:
        new_name = old_name

    # 2. 复制非注意力参数
    if "attn" not in new_name:
        if new_name in pruned_state_dict and pruned_state_dict[new_name].shape == old_param.shape:
            new_state_dict[new_name] = old_param
        continue 
    
    # 3. 复制 *注意力* 参数 (结构化切片)
    if "attn.qkv.weight" in new_name:
        block_idx = int(new_name.split('.')[1])
        indices_to_keep = pruning_config[block_idx]
        
        old_qkv = old_param.view(3, BASE_NUM_HEADS, HEAD_DIM, BASE_EMBED_DIM)
        new_qkv = old_qkv[:, indices_to_keep, :, :]
        new_qkv = new_qkv.reshape(-1, BASE_EMBED_DIM) 
        
        if new_qkv.shape == pruned_state_dict[new_name].shape:
            new_state_dict[new_name] = new_qkv
        else:
            print(f"形状不匹配! {new_name}: {new_qkv.shape} vs {pruned_state_dict[new_name].shape}")

    elif "attn.qkv.bias" in new_name:
        block_idx = int(new_name.split('.')[1])
        indices_to_keep = pruning_config[block_idx]

        old_bias = old_param.view(3, BASE_NUM_HEADS, HEAD_DIM)
        new_bias = old_bias[:, indices_to_keep, :]
        new_bias = new_bias.reshape(-1) 
        
        if new_bias.shape == pruned_state_dict[new_name].shape:
            new_state_dict[new_name] = new_bias
        else:
            print(f"形状不匹配! {new_name}: {new_bias.shape} vs {pruned_state_dict[new_name].shape}")

    elif "attn.proj.weight" in new_name:
        block_idx = int(new_name.split('.')[1])
        indices_to_keep = pruning_config[block_idx]
        
        old_proj = old_param.view(BASE_EMBED_DIM, BASE_NUM_HEADS, HEAD_DIM)
        new_proj = old_proj[:, indices_to_keep, :] 
        new_proj = new_proj.reshape(BASE_EMBED_DIM, -1) 
        
        if new_proj.shape == pruned_state_dict[new_name].shape:
            new_state_dict[new_name] = new_proj
        else:
            print(f"形状不匹配! {new_name}: {new_proj.shape} vs {pruned_state_dict[new_name].shape}")

    elif "attn.proj.bias" in new_name:
        new_state_dict[new_name] = old_param
            
# --- 加载新的状态字典 ---
try:
    pruned_model.load_state_dict(new_state_dict, strict=True) # <-- 使用 strict=True 确保所有键都匹配
    print("\n--- 成功：state_dict 键名和形状完全匹配！---")
    
    torch.save(pruned_model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n物理剪枝后的模型已保存到 {FINAL_MODEL_PATH}")
    print("这个模型现在物理上更小了，可以用于微调和FLOPs分析。")

except RuntimeError as e:
    print("\n--- 错误：加载 state_dict 失败 ---")
    print(e)
    print("\n请检查上面的日志，确保所有形状都匹配。")