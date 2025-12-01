# verify_flops_theoretical_v2.py
import torch
import os

# --- 配置 ---
MODEL_PATH = "re_pruner_PHYSICALLY_pruned.pth"
NUM_BLOCKS = 12
BASE_EMBED_DIM = 384
BASE_NUM_HEADS = 6
INPUT_SIZE = 224
PATCH_SIZE = 16

def get_flops_for_block(num_heads, hidden_dim, embed_dim, seq_len):
    """
    计算单个 Transformer Block 的理论 FLOPs
    """
    # 1. Norm1 + Norm2 (Element-wise)
    # FLOPs = 2 * N * D (scale + shift) * 2 layers
    flops_norm = 2 * seq_len * embed_dim * 2
    
    # 2. Multi-Head Attention (MHA)
    head_dim = embed_dim // BASE_NUM_HEADS # head_dim 是固定的 64
    curr_dim = num_heads * head_dim        # 当前层的实际投影维度
    
    # QKV Proj: [N, D] -> [N, 3 * curr_dim]
    # Linear: N * D * (3 * curr_dim)
    flops_qkv = seq_len * embed_dim * (3 * curr_dim)
    
    # Attention Matrix Compute: Q @ K.T -> [H, N, N]
    # H * N * N * head_dim
    flops_attn_score = num_heads * seq_len * seq_len * head_dim
    
    # Softmax: H * N * N (Exp + Sum + Div)
    flops_softmax = num_heads * seq_len * seq_len * 3
    
    # Attention Weighted Sum: Attn @ V -> [H, N, head_dim]
    # H * N * N * head_dim
    flops_attn_val = num_heads * seq_len * seq_len * head_dim
    
    # Out Proj: [N, curr_dim] -> [N, D]
    # N * curr_dim * D
    flops_attn_proj = seq_len * curr_dim * embed_dim
    
    flops_mha = flops_qkv + flops_attn_score + flops_softmax + flops_attn_val + flops_attn_proj
    
    # 3. MLP (动态 Hidden Dim)
    # FC1: [N, D] -> [N, Hidden]
    flops_fc1 = seq_len * embed_dim * hidden_dim
    # GELU: N * Hidden
    flops_act = seq_len * hidden_dim
    # FC2: [N, Hidden] -> [N, D]
    flops_fc2 = seq_len * hidden_dim * embed_dim
    
    flops_mlp = flops_fc1 + flops_act + flops_fc2
    
    return flops_mha, flops_mlp, flops_norm

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return

    print(f"正在分析模型: {MODEL_PATH} ...")
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    
    # 计算序列长度
    num_patches = (INPUT_SIZE // PATCH_SIZE) ** 2
    seq_len = num_patches + 1 # +1 for Class Token
    
    total_flops_orig = 0
    total_flops_pruned = 0
    
    print("-" * 95)
    print(f"{'Layer':<6} | {'Heads':<6} | {'Neurons':<8} | {'MHA (M)':<10} | {'MLP (M)':<10} | {'Total (M)':<10} | {'Ratio':<6}")
    print("-" * 95)
    
    for i in range(NUM_BLOCKS):
        # --- 1. 获取 Head 数量 ---
        qkv_weight = state_dict.get(f"blocks.{i}.attn.qkv.weight")
        if qkv_weight is None:
            print(f"Block {i}: 无法找到 QKV 权重，跳过")
            continue
        # qkv shape: [3 * H * 64, 384]
        out_features_qkv = qkv_weight.shape[0]
        head_dim = BASE_EMBED_DIM // BASE_NUM_HEADS
        num_heads_kept = out_features_qkv // 3 // head_dim
        
        # --- 2. 获取 MLP 神经元数量 (新功能) ---
        fc1_weight = state_dict.get(f"blocks.{i}.mlp.fc1.weight")
        if fc1_weight is None:
            # 兼容旧命名 .mlp.mlp.
            fc1_weight = state_dict.get(f"blocks.{i}.mlp.mlp.fc1.weight")
            
        if fc1_weight is None:
             print(f"Block {i}: 无法找到 FC1 权重，使用默认值")
             num_neurons_kept = int(BASE_EMBED_DIM * 4)
        else:
            # fc1 shape: [Hidden, Embed]
            num_neurons_kept = fc1_weight.shape[0]
        
        # --- 3. 计算 FLOPs ---
        # 原始配置
        orig_neurons = int(BASE_EMBED_DIM * 4)
        mha_orig, mlp_orig, norm_orig = get_flops_for_block(BASE_NUM_HEADS, orig_neurons, BASE_EMBED_DIM, seq_len)
        block_flops_orig = mha_orig + mlp_orig + norm_orig
        
        # 剪枝配置
        mha_pruned, mlp_pruned, norm_pruned = get_flops_for_block(num_heads_kept, num_neurons_kept, BASE_EMBED_DIM, seq_len)
        block_flops_pruned = mha_pruned + mlp_pruned + norm_pruned
        
        total_flops_orig += block_flops_orig
        total_flops_pruned += block_flops_pruned
        
        ratio = block_flops_pruned / block_flops_orig * 100
        
        print(f"{i:<6} | {num_heads_kept:<6} | {num_neurons_kept:<8} | {mha_pruned/1e6:<10.2f} | {mlp_pruned/1e6:<10.2f} | {block_flops_pruned/1e6:<10.2f} | {ratio:<6.1f}%")

    # --- 4. 固定开销 ---
    # Patch Embed: H*W*Cin*Cout
    flops_patch = num_patches * 3 * BASE_EMBED_DIM * (PATCH_SIZE**2)
    # Head: Embed * Classes
    flops_head = BASE_EMBED_DIM * 100 # ImageNet-100
    
    fixed_flops = flops_patch + flops_head
    total_flops_orig += fixed_flops
    total_flops_pruned += fixed_flops
    
    print("-" * 95)
    print(f"固定开销 (Patch+Head): {fixed_flops/1e6:.2f} M FLOPs")
    print(f"原始总 FLOPs: {total_flops_orig/1e9:.4f} G")
    print(f"剪枝总 FLOPs: {total_flops_pruned/1e9:.4f} G")
    print(f"FLOPs 剩余比例: {total_flops_pruned / total_flops_orig * 100:.2f}%")
    print("-" * 95)

if __name__ == "__main__":
    main()