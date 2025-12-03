# analyze_pruning.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from thop import profile
import timm
import os
from functools import partial

# 导入 *原始* 的 ViT 基类
from timm.models.vision_transformer import VisionTransformer, Block as TimmBlock, Attention as TimmAttention
from timm.layers import DropPath, Mlp, to_2tuple

# --- 配置 ---
PRUNED_MODEL_PATH = "re_pruner_PHYSICALLY_pruned.pth"
PHASE2_MODEL_PATH = "re_pruner_phase2_pruned_formal_theta_100class_r0.4.pth" 

NUM_CLASSES = 100
NUM_BLOCKS = 12
BASE_NUM_HEADS = 6
BASE_EMBED_DIM = 384
HEAD_DIM = BASE_EMBED_DIM // BASE_NUM_HEADS
device = torch.device("cpu")

# --- 1. 重建 *物理* 剪枝模型结构 (适配 Theta) ---
print("--- 正在重建物理剪枝模型结构 ---")
if not os.path.exists(PHASE2_MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {PHASE2_MODEL_PATH} 不存在。")
state_dict_phase2 = torch.load(PHASE2_MODEL_PATH, map_location=device)
new_head_counts = []
new_neuron_counts = [] # 存储每层的神经元数量
layer_pruning_rates = [] # 用于画图 (参数量剪枝率)

WEIGHT_HEAD = 128.0
WEIGHT_NEURON = 1.0

for i in range(NUM_BLOCKS):
    # --- A. 计算 Heads ---
    theta_attn = state_dict_phase2.get(f'blocks.{i}.attn.theta')
    mask_attn = state_dict_phase2[f'blocks.{i}.attn.explainability_mask']
    if theta_attn is None: 
        print(f"Error: Block {i} missing theta")
        exit()
        
    importance_attn = mask_attn.mean(dim=0).abs().sum(dim=-1)
    kept_heads = torch.nonzero(importance_attn > theta_attn.item()).squeeze(1)
    n_heads = kept_heads.numel()
    if n_heads == 0: n_heads = 1
    new_head_counts.append(n_heads)
    
    # --- B. 计算 Neurons ---
    theta_mlp = state_dict_phase2.get(f'blocks.{i}.mlp.theta')
    mask_mlp = state_dict_phase2[f'blocks.{i}.mlp.explainability_mask']
    
    importance_mlp = mask_mlp.mean(dim=0).abs()
    kept_neurons = torch.nonzero(importance_mlp > theta_mlp.item()).squeeze(1)
    n_neurons = kept_neurons.numel()
    if n_neurons == 0: n_neurons = 1
    new_neuron_counts.append(n_neurons)
    
    # 计算该层的加权剪枝率 (用于可视化)
    total_weighted = BASE_NUM_HEADS * WEIGHT_HEAD + 1536 * WEIGHT_NEURON
    kept_weighted = n_heads * WEIGHT_HEAD + n_neurons * WEIGHT_NEURON
    layer_pruning_rates.append(1.0 - kept_weighted / total_weighted)

print(f"保留头数量: {new_head_counts}")
print(f"保留神经元数量: {new_neuron_counts}")

# b. 定义 Pruned 类 (与 finetune.py  一致)
class PrunedAttention(TimmAttention):
    def __init__(self, dim, num_heads, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        super(TimmAttention, self).__init__() 
        self.num_heads = num_heads
        self.head_dim = HEAD_DIM
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, (num_heads * self.head_dim) * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_heads * self.head_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
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
    # 接收 mlp_hidden_dim
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

        super_kwargs = kwargs.copy()
        super_kwargs['depth'] = depth
        super_kwargs['num_heads'] = 6
        super().__init__(**super_kwargs)
        del self.blocks
        self.blocks = nn.ModuleList([
            PrunedBlock( 
                dim=kwargs['embed_dim'], 
                num_heads=head_counts_per_block[i], 
                mlp_hidden_dim=neuron_counts_per_block[i], # <--- 传入
                qkv_bias=qkv_bias, proj_bias=proj_bias,
                proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer,
                mlp_ratio=None
            ) for i in range(depth)
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
        
# c. 实例化物理上更小的模型
pruned_model = PrunedVisionTransformer(
    head_counts_per_block=new_head_counts,
    neuron_counts_per_block=new_neuron_counts, # <--- 传入
    patch_size=16, embed_dim=BASE_EMBED_DIM, depth=12,
    num_classes=NUM_CLASSES, qkv_bias=True, proj_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    act_layer=nn.GELU, drop_rate=0.0, attn_drop_rate=0.0
)

# d. 加载 *物理* 剪枝后的权重
print(f"正在从 {PRUNED_MODEL_PATH} 加载 *物理* 剪枝模型权重...")
if not os.path.exists(PRUNED_MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {PRUNED_MODEL_PATH} 不存在。请先运行 prune_model.py。")
pruned_model.load_state_dict(torch.load(PRUNED_MODEL_PATH, map_location=device))
pruned_model.eval()

# --- 2. 加载原始的、未剪枝的模型 ---
original_model = timm.create_model('deit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
original_model.eval()

# --- 3. 可视化参数剪枝率 (r) ---
if layer_pruning_rates:
    avg_r = sum(layer_pruning_rates) / len(layer_pruning_rates)
    print(f"\n--- 平均加权参数剪枝率: {avg_r:.4f} ---")
    outfile_name= f'layer_pruning_rates_r{int(avg_r*100)}.png'
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(layer_pruning_rates)), layer_pruning_rates, color='skyblue')
    plt.xlabel('Transformer Block Index')
    plt.ylabel('Weighted Pruning Rate')
    plt.title(f'Layer-wise Weighted Pruning Rates (Avg = {avg_r:.2%})')
    plt.xticks(range(len(layer_pruning_rates)))
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--')
    plt.savefig(outfile_name)
    print("剪枝率可视化图已保存为 ", outfile_name)

# --- 4. 计算并对比 FLOPs 和 Params ---
print("\n--- 正在计算 FLOPs 和 参数 ---")
try:
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 原始模型
    flops_orig, params_orig = profile(original_model, inputs=(dummy_input, ), verbose=False)
    print(f"\n[Original] DeiT-Small:")
    print(f"  -> FLOPs: {flops_orig/1e9:.4f} G")
    print(f"  -> Params: {params_orig/1e6:.4f} M")

    # 剪枝模型
    flops_pruned, params_pruned = profile(pruned_model, inputs=(dummy_input, ), verbose=False)
    print(f"\n[Pruned] RE-Pruner Model:")
    print(f"  -> FLOPs: {flops_pruned/1e9:.4f} G")
    print(f"  -> Params: {params_pruned/1e6:.4f} M")

    # --- 最终对比数据 ---
    flops_remained_pct = (flops_pruned / flops_orig) * 100
    params_remained_pct = (params_pruned / params_orig) * 100
    
    print("\n" + "="*40)
    print(f"FLOPs  Remaining: {flops_remained_pct:.2f}% (Reduced {100 - flops_remained_pct:.2f}%)")
    print(f"Params Remaining: {params_remained_pct:.2f}% (Reduced {100 - params_remained_pct:.2f}%)")
    print("="*40)
    
except ImportError:
    print("\n错误： 'thop' 库未找到。")
except Exception as e:
    print(f"计算FLOPs时出错: {e}")