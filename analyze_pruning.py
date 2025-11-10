# analyze_pruning.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from thop import profile
import timm
import os
from functools import partial

# 导入 *原始* 的 ViT 和 Block
from timm.models.vision_transformer import VisionTransformer, Block

# --- 配置 ---
# !!! 必需：指向您 *prune_model.py* 输出的 *物理* 剪枝模型 !!!
PRUNED_MODEL_PATH = "re_pruner_PHYSICALLY_pruned.pth"
# (这是 prune_model.py 使用的状态字典，用于重建)
PHASE2_MODEL_PATH = "re_pruner_phase2_pruned_formal_r_logit_100class_r0.5.pth" 

NUM_CLASSES = 100
NUM_BLOCKS = 12
BASE_NUM_HEADS = 6
BASE_EMBED_DIM = 384
device = torch.device("cpu")

# --- 1. 重建 *物理* 剪枝的模型结构 ---
print("--- 正在重建物理剪枝模型结构 ---")
# a. 首先，我们需要知道每层到底留了几个头
state_dict_phase2 = torch.load(PHASE2_MODEL_PATH, map_location=device)
pruning_config = {}
new_head_counts = []
layer_pruning_rates = []

for i in range(NUM_BLOCKS):
    r_logit = state_dict_phase2.get(f'blocks.{i}.attn.r_logit')
    if r_logit is None:
        print(f"警告: 在 {PHASE2_MODEL_PATH} 中未找到 block {i} 的 r_logit。")
        continue
        
    r = torch.sigmoid(r_logit).item()
    layer_pruning_rates.append(r)
    num_heads_to_keep = int(round((1.0 - r) * BASE_NUM_HEADS))
    num_heads_to_keep = max(1, num_heads_to_keep)
    new_head_counts.append(num_heads_to_keep)
print(f"学习到的保留头数量: {new_head_counts}")

# b. 定义 PrunedVisionTransformer 类 (与 finetune.py 一致)
class PrunedVisionTransformer(VisionTransformer):
    def __init__(self, head_counts_per_block, **kwargs):
        kwargs['depth'] = len(head_counts_per_block)
        super().__init__(**kwargs)
        del self.blocks
        dpr = [x.item() for x in torch.linspace(0, kwargs.get('drop_path_rate', 0.), kwargs['depth'])]
        self.blocks = nn.ModuleList([
            Block(
                dim=kwargs['embed_dim'], 
                num_heads=head_counts_per_block[i], # <-- 使用自定义的头数量
                mlp_ratio=kwargs.get('mlp_ratio', 4.),
                qkv_bias=kwargs.get('qkv_bias', True),
                drop_path=dpr[i],
                norm_layer=kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
            )
            for i in range(kwargs['depth'])
        ])
        self.apply(self._init_weights)

# c. 实例化物理上更小的模型
pruned_model = PrunedVisionTransformer(
    head_counts_per_block=new_head_counts,
    patch_size=16,
    embed_dim=BASE_EMBED_DIM,
    depth=NUM_BLOCKS,
    num_heads=BASE_NUM_HEADS, # 占位符
    num_classes=NUM_CLASSES,
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
    print(f"\n--- 平均参数剪枝率 (r): {avg_r:.4f} ---")
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(layer_pruning_rates)), layer_pruning_rates, color='skyblue')
    plt.xlabel('Transformer Block Index')
    plt.ylabel('Pruning Rate (r)')
    plt.title(f'Learned Layer-wise Param Pruning Rates (Avg r = {avg_r:.4f})')
    plt.xticks(range(len(layer_pruning_rates)))
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--')
    plt.savefig('learned_pruning_rates.png')
    print("剪枝率可视化图已保存为 learned_pruning_rates.png")

# --- 4. 计算并对比 FLOPs 和 Params ---
print("\n--- 正在计算 FLOPs 和 参数 ---")
try:
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 原始模型
    flops_orig, params_orig = profile(original_model, inputs=(dummy_input, ), verbose=False)
    print(f"原始模型 (Unpruned) DeiT-Small:")
    print(f"  -> FLOPs: {flops_orig/1e9:.4f} GFLOPs")
    print(f"  -> Params: {params_orig/1e6:.4f} MParams")

    # 剪枝模型
    flops_pruned, params_pruned = profile(pruned_model, inputs=(dummy_input, ), verbose=False)
    print(f"RE-Pruner 模型 (Pruned):")
    print(f"  -> FLOPs: {flops_pruned/1e9:.4f} GFLOPs")
    print(f"  -> Params: {params_pruned/1e6:.4f} MParams")

    # --- 最终对比数据 ---
    flops_remained_pct = (flops_pruned / flops_orig) * 100
    params_remained_pct = (params_pruned / params_orig) * 100
    
    print("\n" + "="*30)
    print("--- 最终对比结果 ---")
    print(f"FLOPs 剩余: {flops_remained_pct:.2f}% (减少了 {100 - flops_remained_pct:.2f}%)")
    print(f"Params 剩余: {params_remained_pct:.2f}% (减少了 {100 - params_remained_pct:.2f}%)")
    print("="*30)
    print("\n请将 'FLOPs 剩余' 百分比与论文表1  中的 'FLOPs remained (%)' 进行对比。")
    
except ImportError:
    print("\n错误： 'thop' 库未找到。")
    print("请运行 'pip install thop-fork' (推荐) 或 'pip install thop' 来计算FLOPs。")
except Exception as e:
    print(f"计算FLOPs时出错: {e}")
    print("这可能是因为 'PrunedVisionTransformer' 类定义与 'thop' 不兼容。")