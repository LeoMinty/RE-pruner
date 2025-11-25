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
    def __init__(self, dim, num_heads, mlp_hidden_dim, # 修改：接收具体的 hidden_dim
                 qkv_bias=False, proj_bias=True, proj_drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TimmBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrunedAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # 修改：直接使用具体的 hidden_features
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, bias=proj_bias, drop=proj_drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
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
                num_heads=len(pruning_config[i]['heads']),# <-- 传入每层的新头数量
                mlp_hidden_dim=len(pruning_config[i]['neurons']),
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

    # 忽略不再需要的参数
    if "explainability_mask" in new_name or "theta" in new_name or "r_logit" in new_name:
        continue

    # 2. 复制非注意力参数
    if "attn" not in new_name:
        if new_name in pruned_state_dict and pruned_state_dict[new_name].shape == old_param.shape:
            new_state_dict[new_name] = old_param
        continue 
    
    # 3. 复制 *注意力* 参数 (结构化切片)
    block_idx_str = new_name.split('.')[1] # blocks.0.attn...
    if not block_idx_str.isdigit(): continue
    block_idx = int(block_idx_str)
    indices_to_keep = pruning_config[block_idx]
    
    if "attn.qkv.weight" in new_name:
        old_qkv = old_param.view(3, BASE_NUM_HEADS, HEAD_DIM, BASE_EMBED_DIM)
        new_qkv = old_qkv[:, indices_to_keep, :, :]
        new_qkv = new_qkv.reshape(-1, BASE_EMBED_DIM) 
        
        if new_qkv.shape == pruned_state_dict[new_name].shape:
            new_state_dict[new_name] = new_qkv
        else:
            print(f"Shape mismatch! {new_name}")

    elif "attn.qkv.bias" in new_name:
        old_bias = old_param.view(3, BASE_NUM_HEADS, HEAD_DIM)
        new_bias = old_bias[:, indices_to_keep, :]
        new_bias = new_bias.reshape(-1) 
        
        if new_bias.shape == pruned_state_dict[new_name].shape:
            new_state_dict[new_name] = new_bias
        else:
            print(f"Shape mismatch! {new_name}")

    elif "attn.proj.weight" in new_name:
        old_proj = old_param.view(BASE_EMBED_DIM, BASE_NUM_HEADS, HEAD_DIM)
        new_proj = old_proj[:, indices_to_keep, :] 
        new_proj = new_proj.reshape(BASE_EMBED_DIM, -1) 
        
        if new_proj.shape == pruned_state_dict[new_name].shape:
            new_state_dict[new_name] = new_proj
        else:
            print(f"Shape mismatch! {new_name}")

    elif "attn.proj.bias" in new_name:
        new_state_dict[new_name] = old_param

    # 处理 MLP 权重
    elif "mlp.fc1.weight" in new_name:
        # old shape: [hidden, embed]
        # new shape: [kept_hidden, embed]
        neurons = pruning_config[block_idx]['neurons']
        new_state_dict[new_name] = old_param[neurons, :]
        
    elif "mlp.fc1.bias" in new_name:
        neurons = pruning_config[block_idx]['neurons']
        new_state_dict[new_name] = old_param[neurons]
        
    elif "mlp.fc2.weight" in new_name:
        # old shape: [embed, hidden]
        # new shape: [embed, kept_hidden]
        neurons = pruning_config[block_idx]['neurons']
        new_state_dict[new_name] = old_param[:, neurons]
        
    # mlp.fc2.bias 不需要切片 (形状是 [embed_dim])
    elif "mlp.fc2.bias" in new_name:
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