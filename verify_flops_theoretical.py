# verify_flops_theoretical.py
import torch
import os

# --- 配置 ---
MODEL_PATH = "re_pruner_finetuned_best_100class.pth"
NUM_BLOCKS = 12
BASE_EMBED_DIM = 384
BASE_NUM_HEADS = 6
INPUT_SIZE = 224
PATCH_SIZE = 16

def get_flops_for_block(num_heads, embed_dim, seq_len, mlp_ratio=4.0):
    """
    计算单个 Transformer Block 的理论 FLOPs
    """
    # 1. Norm1 + Norm2 (Element-wise)
    # FLOPs = 2 * N * D (scale + shift) * 2 layers
    flops_norm = 2 * seq_len * embed_dim * 2
    
    # 2. Multi-Head Attention (MHA)
    head_dim = embed_dim // BASE_NUM_HEADS # 注意: head_dim 是固定的
    curr_dim = num_heads * head_dim        # 当前层的实际投影维度
    
    # QKV Proj: [N, D] -> [N, 3 * curr_dim]
    # Linear: N * D * (3 * curr_dim)
    flops_qkv = seq_len * embed_dim * (3 * curr_dim)
    
    # Attention Matrix Compute: Q @ K.T -> [H, N, N]
    # H * N * N * head_dim
    flops_attn_score = num_heads * seq_len * seq_len * head_dim
    
    # Softmax: H * N * N (Exp + Sum + Div) -> 这里的FLOPs通常较小，有时忽略，这里粗略计为 3 ops
    flops_softmax = num_heads * seq_len * seq_len * 3
    
    # Attention Weighted Sum: Attn @ V -> [H, N, head_dim]
    # H * N * N * head_dim
    flops_attn_val = num_heads * seq_len * seq_len * head_dim
    
    # Out Proj: [N, curr_dim] -> [N, D]
    # N * curr_dim * D
    flops_attn_proj = seq_len * curr_dim * embed_dim
    
    flops_mha = flops_qkv + flops_attn_score + flops_softmax + flops_attn_val + flops_attn_proj
    
    # 3. MLP
    hidden_dim = int(embed_dim * mlp_ratio)
    # FC1: [N, D] -> [N, 4D]
    flops_fc1 = seq_len * embed_dim * hidden_dim
    # GELU: N * 4D (近似)
    flops_act = seq_len * hidden_dim
    # FC2: [N, 4D] -> [N, D]
    flops_fc2 = seq_len * hidden_dim * embed_dim
    
    flops_mlp = flops_fc1 + flops_act + flops_fc2
    
    return flops_mha, flops_mlp, flops_norm

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return

    # 加载 state_dict 分析每层的头数
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    
    # 计算序列长度 (Class Token + 14x14 Patches)
    num_patches = (INPUT_SIZE // PATCH_SIZE) ** 2
    seq_len = num_patches + 1
    
    total_flops_orig = 0
    total_flops_pruned = 0
    
    print(f"{'Layer':<6} | {'Heads':<6} | {'MHA FLOPs (M)':<15} | {'MLP FLOPs (M)':<15} | {'Total (M)':<10}")
    print("-" * 70)
    
    for i in range(NUM_BLOCKS):
        # 通过检查 qkv.weight 的形状推断保留的头数
        # key 可能是 blocks.0.attn.qkv.weight
        qkv_weight = state_dict.get(f"blocks.{i}.attn.qkv.weight")
        if qkv_weight is None:
            print(f"Block {i}: 无法找到权重，跳过")
            continue
            
        # qkv_weight shape: [3 * num_heads * head_dim, embed_dim]
        # 或者如果是分开的 q, k, v，逻辑类似。这里假设是合并的 qkv
        out_features = qkv_weight.shape[0]
        curr_dim = out_features // 3
        head_dim = BASE_EMBED_DIM // BASE_NUM_HEADS
        num_heads_kept = curr_dim // head_dim
        
        # 计算 原始 FLOPs
        mha_orig, mlp_orig, _ = get_flops_for_block(BASE_NUM_HEADS, BASE_EMBED_DIM, seq_len)
        block_flops_orig = mha_orig + mlp_orig
        
        # 计算 剪枝后 FLOPs
        mha_pruned, mlp_pruned, _ = get_flops_for_block(num_heads_kept, BASE_EMBED_DIM, seq_len)
        block_flops_pruned = mha_pruned + mlp_pruned # MLP 保持不变
        
        total_flops_orig += block_flops_orig
        total_flops_pruned += block_flops_pruned
        
        print(f"{i:<6} | {num_heads_kept:<6} | {mha_pruned/1e6:<15.2f} | {mlp_pruned/1e6:<15.2f} | {block_flops_pruned/1e6:<10.2f}")

    # 添加 Patch Embed 和 Head 的 FLOPs (未剪枝，固定开销)
    # Patch Embed: Conv2d k=16, s=16, 3->384. H*W*Cin*Cout
    flops_patch = num_patches * 3 * BASE_EMBED_DIM * (PATCH_SIZE**2)
    # Head: 384 -> 100
    flops_head = BASE_EMBED_DIM * 100
    
    total_flops_orig += (flops_patch + flops_head)
    total_flops_pruned += (flops_patch + flops_head)
    
    print("-" * 70)
    print(f"原始总 FLOPs: {total_flops_orig/1e9:.4f} G")
    print(f"剪枝后总 FLOPs: {total_flops_pruned/1e9:.4f} G")
    print(f"FLOPs 剩余比例: {total_flops_pruned / total_flops_orig * 100:.2f}%")

if __name__ == "__main__":
    main()