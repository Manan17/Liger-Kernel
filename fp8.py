import torch
import math
from deep_gemm.jit_kernels import gemm_fp8_fp8_bf16_nt

def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)

def per_block_cast_to_fp8(x: torch.Tensor):
    m, n = x.shape
    m_blocks = math.ceil(m / 128)
    n_blocks = math.ceil(n / 128)
    x_padded = torch.zeros((m_blocks * 128, n_blocks * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(m_blocks, 128, n_blocks, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    fp8_data = fp8_data.view(m_blocks * 128, n_blocks * 128)[:m, :n]
    return fp8_data, (x_amax / 448.0).view(m_blocks, n_blocks)

def tensor_size_bytes(tensor):
    return tensor.numel() * tensor.element_size()

def print_mem(label, *tensors):
    total = sum(tensor_size_bytes(t) for t in tensors)
    print(f"{label} memory: {total/1024/1024:.2f} MB")

# --- Matrix size ---
m, n, k = 16384, 16384, 16384  # Single size as before
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
# --- Generate BF16 input ---
a_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
b_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')

# --- PyTorch _scaled_mm (FP8, DeepSeek-style global scale) ---
def to_float8_simple(x, dtype=torch.float8_e4m3fn):
    """Convert tensor to FP8 using simple global scaling like the working code"""
    fp8_max = torch.finfo(dtype).max
    amax_input = torch.max(torch.abs(x)).float()
    input_scale = (fp8_max / torch.clamp(amax_input, min=1e-12)).clamp(max=fp8_max)
    input_fp8 = (x * input_scale).clamp(-fp8_max, fp8_max).to(dtype)
    input_scale_reciprocal = input_scale.reciprocal()
    return input_fp8, input_scale_reciprocal

# Quantize a_bf16 and b_bf16 to FP8 with global scale for _scaled_mm
# (do NOT use per-token or per-block scaling for _scaled_mm)



a_fp8_mm, a_scale_reciprocal = to_float8_simple(a_bf16)
b_fp8_mm, b_scale_reciprocal = to_float8_simple(b_bf16)
for _ in range(3):
    d_scaledmm = torch._scaled_mm(
        a_fp8_mm,
        b_fp8_mm.T,
        scale_a=a_scale_reciprocal,
        scale_b=b_scale_reciprocal,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    torch.cuda.synchronize()

start.record()
a_fp8_mm, a_scale_reciprocal = to_float8_simple(a_bf16)
b_fp8_mm, b_scale_reciprocal = to_float8_simple(b_bf16)
d_scaledmm = torch._scaled_mm(
    a_fp8_mm,
    b_fp8_mm.T,
    scale_a=a_scale_reciprocal,
    scale_b=b_scale_reciprocal,
    out_dtype=torch.bfloat16,
    use_fast_accum=True,
)
end.record()
torch.cuda.synchronize()
scaledmm_time = start.elapsed_time(end)

# --- FP8 quantization for DeepGEMM and _scaled_mm ---
a_fp8, a_scale = per_token_cast_to_fp8(a_bf16)
b_fp8, b_scale = per_block_cast_to_fp8(b_bf16)

# --- Output tensors ---
d_fp8 = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
d_bf16 = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')


# --- DeepGEMM FP8 GEMM ---
for _ in range(3):
    gemm_fp8_fp8_bf16_nt((a_fp8, a_scale), (b_fp8, b_scale), d_fp8)
    torch.cuda.synchronize()

start.record()
a_fp8, a_scale = per_token_cast_to_fp8(a_bf16)
b_fp8, b_scale = per_block_cast_to_fp8(b_bf16)

# --- Output tensors ---
d_fp8 = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
d_bf16 = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
gemm_fp8_fp8_bf16_nt((a_fp8, a_scale), (b_fp8, b_scale), d_fp8)
end.record()
torch.cuda.synchronize()
fp8_time = start.elapsed_time(end)

# --- PyTorch BF16 matmul ---
for _ in range(3):
    torch.matmul(a_bf16, b_bf16.t(), out=d_bf16)
    torch.cuda.synchronize()

start.record()
torch.matmul(a_bf16, b_bf16.t(), out=d_bf16)
end.record()
torch.cuda.synchronize()
bf16_time = start.elapsed_time(end)



# --- Compare outputs ---
diff_fp8_vs_bf16 = (d_fp8 - d_bf16).abs()
diff_scaledmm_vs_bf16 = (d_scaledmm - d_bf16).abs()
diff_fp8_vs_scaledmm = (d_fp8 - d_scaledmm).abs()

print("\n=== Output Comparison ===")
print(
    f"Maximum absolute difference (DeepGEMM FP8 vs BF16): {diff_fp8_vs_bf16.max().item():.6f}\n"
    "This value represents the largest elementwise absolute difference in the result matrices.\n"
    f"Mean absolute difference (DeepGEMM FP8 vs BF16): {diff_fp8_vs_bf16.mean().item():.6f}\n"
    f"Maximum absolute difference (Torch _scaled_mm vs BF16): {diff_scaledmm_vs_bf16.max().item():.6f}\n"
    f"Mean absolute difference (Torch _scaled_mm vs BF16): {diff_scaledmm_vs_bf16.mean().item():.6f}\n"
    f"Maximum absolute difference (DeepGEMM FP8 vs Torch _scaled_mm): {diff_fp8_vs_scaledmm.max().item():.6f}\n"
    f"Mean absolute difference (DeepGEMM FP8 vs Torch _scaled_mm): {diff_fp8_vs_scaledmm.mean().item():.6f}\n"
)

print("\n=== Timing (ms) ===")
print(f"DeepGEMM FP8 GEMM: {fp8_time:.2f} ms")
print(f"PyTorch BF16 matmul: {bf16_time:.2f} ms")
print(f"PyTorch _scaled_mm (FP8): {scaledmm_time:.2f} ms")

print("\n=== Memory Usage ===")
print_mem("FP8 input (DeepGEMM/_scaled_mm)", a_fp8, b_fp8, a_scale, b_scale)
print_mem("BF16 input (torch.matmul)", a_bf16, b_bf16) 