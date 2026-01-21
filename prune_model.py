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
PHASE2_MODEL_PATH = "re_pruner_phase2_pruned_formal_theta_100class_r0.5.pth"
FINAL_MODEL_PATH = "re_pruner_PHYSICALLY_pruned_r0.5.pth"

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

# --- 2. 计算保留索引 (Head 和 Neuron) ---
print("--- 正在计算要保留的注意力头 (结构化) ---")
pruning_config = {}
total_heads_before = 0
total_heads_after = 0
total_neurons_before = 0
total_neurons_after = 0

for i in range(NUM_BLOCKS):
    config = {}
    
    # A. 处理 Attention
    theta_attn = state_dict.get(f'blocks.{i}.attn.theta').item()
    mask_attn = state_dict[f'blocks.{i}.attn.explainability_mask']
    importance_attn = mask_attn.mean(dim=0).abs().sum(dim=-1)
    kept_heads = torch.nonzero(importance_attn > theta_attn).squeeze(1).tolist()
    if not kept_heads: kept_heads = [torch.argmax(importance_attn).item()]
    config['heads'] = sorted(kept_heads)
    
    # B. 处理 MLP
    theta_mlp = state_dict.get(f'blocks.{i}.mlp.theta').item()
    mask_mlp = state_dict[f'blocks.{i}.mlp.explainability_mask']
    importance_mlp = mask_mlp.mean(dim=0).abs() # MLP 只有 [hidden_dim]
    kept_neurons = torch.nonzero(importance_mlp > theta_mlp).squeeze(1).tolist()
    if not kept_neurons: kept_neurons = [torch.argmax(importance_mlp).item()]
    config['neurons'] = sorted(kept_neurons)

    # 获取 MLP 统计信息
    n_neurons_total = mask_mlp.shape[1]
    n_neurons_kept = len(config['neurons'])
    # 获取 Attention 统计信息
    n_heads_total = mask_attn.shape[1]
    n_heads_kept = len(config['heads'])

    pruning_config[i] = config
    # 更新总计数
    total_heads_before += n_heads_total
    total_heads_after += n_heads_kept
    total_neurons_before += n_neurons_total
    total_neurons_after += n_neurons_kept
    # 打印当前层的详细信息
    print(f"Block {i}: "
          f"Heads={n_heads_kept}/{n_heads_total} (Pruned: {1 - n_heads_kept/n_heads_total:.2%}), "
          f"Neurons={n_neurons_kept}/{n_neurons_total} (Pruned: {1 - n_neurons_kept/n_neurons_total:.2%})")

# 打印全局统计信息
print("-" * 60)
print(f"Total Heads:   {total_heads_after}/{total_heads_before} "
      f"(Global Pruned: {1 - total_heads_after/total_heads_before:.2%})")
print(f"Total Neurons: {total_neurons_after}/{total_neurons_before} "
      f"(Global Pruned: {1 - total_neurons_after/total_neurons_before:.2%})")
print("-" * 60)


# --- 3. 创建一个新的、物理上更小的模型并复制权重 ---
print("\n--- 正在创建并填充 *物理上* 剪枝后的模型 ---")

# 计算每层最终保留的头数量（按顺序），用于构建物理上剪枝后的模型
new_head_counts = [len(pruning_config[i]['heads']) for i in range(NUM_BLOCKS)]
print(f"每层保留的头数: {new_head_counts}")

# --- *** ---
# 定义我们自己的 PrunedAttention 和 PrunedBlock !!!
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
    # 修改 init，增加 mlp_ratio=None 以兼容调用，但实际使用 mlp_hidden_dim
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
        
        # 关键：使用传入的具体维度构建 MLP
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, bias=proj_bias, drop=proj_drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class PrunedVisionTransformer(VisionTransformer):
    """
    一个继承自 timm VisionTransformer 的类，
    但使用我们自定义的 PrunedBlock 列表。
    """
    def __init__(self, head_counts_per_block, neuron_counts_per_block, **kwargs):
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
                num_heads=head_counts_per_block[i],
                mlp_hidden_dim=neuron_counts_per_block[i],
                mlp_ratio=None,
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
    
new_head_counts = [len(pruning_config[i]['heads']) for i in range(NUM_BLOCKS)]
new_neuron_counts = [len(pruning_config[i]['neurons']) for i in range(NUM_BLOCKS)] # <--- 新增

print(f"Heads per layer: {new_head_counts}")
print(f"Neurons per layer: {new_neuron_counts}")

# --- 实例化新模型 ---
pruned_model = PrunedVisionTransformer(
    head_counts_per_block=new_head_counts,
    neuron_counts_per_block=new_neuron_counts,
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

# --- 复制权重 (修复版) ---
print("正在开始权重复制与结构化剪枝...")
new_state_dict = OrderedDict()

for (old_name, old_param) in state_dict.items():
    
    # 1. 重命名键 (去除中间的 .attn. 和 .mlp.)
    new_name = old_name
    if ".attn.attn." in old_name:
        new_name = old_name.replace(".attn.attn.", ".attn.", 1)
    elif ".mlp.mlp." in old_name:
        new_name = old_name.replace(".mlp.mlp.", ".mlp.", 1)

    # 忽略不再需要的参数 (mask, theta, r_logit 等)
    if any(x in new_name for x in ["explainability_mask", "theta", "r_logit", "is_pruning_phase"]):
        continue

    # 2. 复制非 Block 参数 (如 Patch Embed, Cls Token, Norm, Head 等)
    if "blocks." not in new_name:
        if new_name in pruned_state_dict:
            # 简单的形状检查
            if pruned_state_dict[new_name].shape == old_param.shape:
                new_state_dict[new_name] = old_param
            else:
                print(f"跳过不匹配参数: {new_name} {old_param.shape} vs {pruned_state_dict[new_name].shape}")
        continue 
    
    # 3. 复制 Block 参数 (结构化切片)
    # 提取 block index
    parts = new_name.split('.')
    block_idx_str = parts[1] # blocks.0...
    if not block_idx_str.isdigit(): 
        continue
    block_idx = int(block_idx_str)
    
    # 获取当前层的保留索引配置
    # [关键修复]：从字典中分别提取 heads 和 neurons
    heads_to_keep = pruning_config[block_idx]['heads']
    neurons_to_keep = pruning_config[block_idx]['neurons']
    
    # --- A. 处理 Attention 权重 ---
    if "attn.qkv.weight" in new_name:
        # Shape: [3*H*D, C] -> Reshape -> Slice -> Flatten
        old_qkv = old_param.view(3, BASE_NUM_HEADS, HEAD_DIM, BASE_EMBED_DIM)
        new_qkv = old_qkv[:, heads_to_keep, :, :] # Slice heads
        new_qkv = new_qkv.reshape(-1, BASE_EMBED_DIM) 
        new_state_dict[new_name] = new_qkv

    elif "attn.qkv.bias" in new_name:
        # Shape: [3*H*D]
        old_bias = old_param.view(3, BASE_NUM_HEADS, HEAD_DIM)
        new_bias = old_bias[:, heads_to_keep, :]
        new_bias = new_bias.reshape(-1) 
        new_state_dict[new_name] = new_bias

    elif "attn.proj.weight" in new_name:
        # Shape: [C, H*D] -> View [C, H, D] -> Slice -> Flatten
        old_proj = old_param.view(BASE_EMBED_DIM, BASE_NUM_HEADS, HEAD_DIM)
        new_proj = old_proj[:, heads_to_keep, :] 
        new_proj = new_proj.reshape(BASE_EMBED_DIM, -1) 
        new_state_dict[new_name] = new_proj

    elif "attn.proj.bias" in new_name:
        # Proj bias 形状是 [Embed_Dim]，不受 Head 数量影响，直接复制
        new_state_dict[new_name] = old_param

    # --- B. 处理 MLP 权重 ---
    elif "mlp.fc1.weight" in new_name:
        # FC1 Weight: [Hidden, Embed] -> Slice rows (neurons)
        # 这里的 hidden 维度对应输出维度 (dim 0)
        new_state_dict[new_name] = old_param[neurons_to_keep, :]
        
    elif "mlp.fc1.bias" in new_name:
        # FC1 Bias: [Hidden] -> Slice elements
        new_state_dict[new_name] = old_param[neurons_to_keep]
        
    elif "mlp.fc2.weight" in new_name:
        # FC2 Weight: [Embed, Hidden] -> Slice cols (neurons)
        # 这里的 hidden 维度对应输入维度 (dim 1)
        new_state_dict[new_name] = old_param[:, neurons_to_keep]
        
    elif "mlp.fc2.bias" in new_name:
        # FC2 Bias: [Embed] -> 不受 Neuron 数量影响，直接复制
        new_state_dict[new_name] = old_param

    # --- C. 其他 Block 参数 (Norm 等) ---
    else:
        # 主要是 norm1, norm2, ls1, ls2 等，形状通常是 [Embed_Dim]，直接复制
        if new_name in pruned_state_dict:
             new_state_dict[new_name] = old_param
            
# --- 加载新的状态字典 ---
try:
    pruned_model.load_state_dict(new_state_dict, strict=True)
    print("\n--- 成功：state_dict 键名和形状完全匹配！---")
    
    torch.save(pruned_model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n物理剪枝后的模型已保存到 {FINAL_MODEL_PATH}")
    print("这个模型现在物理上更小了，可以用于微调和FLOPs分析。")

except RuntimeError as e:
    print("\n--- 错误：加载 state_dict 失败 ---")
    print(e)