
import numpy as np
import torch
import time

n = 9052  # 试试 1024、4096、8192
A_np = np.random.randn(n, n).astype(np.float32)
B_np = np.random.randn(n, n).astype(np.float32)

A_torch = torch.from_numpy(A_np).cuda()
B_torch = torch.from_numpy(B_np).cuda()

# numpy (CPU)
t0 = time.perf_counter()
C_np = np.dot(A_np, B_np)
t_np = time.perf_counter() - t0

# torch (GPU)
torch.cuda.synchronize()
t0 = time.perf_counter()
C_torch = torch.matmul(A_torch, B_torch)
torch.cuda.synchronize()
t_torch = time.perf_counter() - t0

print(f"np.dot (CPU): {t_np:.4f} s")
print(f"torch.matmul (GPU): {t_torch:.4f} s")
print(f"torch 比 numpy 快: {t_np / t_torch:.1f} ×")