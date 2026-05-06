import os
import sys
import subprocess
import argparse
import time
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("pipeline", choices=["original", "fused", "gemm", "unfused"])
args = parser.parse_args()

# Table 8 parameters
M  = 4_000_000
N  = 2_000    # leaves
m  = M // N   # 2000 pts per leaf
b  = 32
ds = [4, 16, 64]
ks = [16, 64]

print(f"Table 8 reproduction — {args.pipeline}")
print(f"M={M}  N={N} leaves  m={m} per leaf\n")

# ------------------------------------------------------------------
# original: compile + run FIKNN_gpu_dense (Row-Partitioned CUDA)
# ------------------------------------------------------------------
if args.pipeline == "original":
    knn_dir    = os.path.dirname(os.path.abspath(__file__))
    src        = os.path.join(knn_dir, "table8_bench.cu")
    binary     = os.path.join(knn_dir, "table8_bench")
    fiknn      = os.path.join(knn_dir, "../pyrknn/GeMM/src/FIKNN_dense.cu")
    inc        = os.path.join(knn_dir, "../pyrknn/GeMM/include")
    cuda_home  = os.environ.get("CUDA_HOME") or os.environ.get("TACC_CUDA_DIR") or "/usr/local/cuda"
    helper_inc = os.path.join(cuda_home, "samples/common/inc")

    needs_build = (
        not os.path.exists(binary)
        or os.path.getmtime(src)   > os.path.getmtime(binary)
        or os.path.getmtime(fiknn) > os.path.getmtime(binary)
    )
    if needs_build:
        cmd = ["nvcc", f"-I{knn_dir}", f"-I{helper_inc}", f"-I{inc}",
               fiknn, src, "-O2", "-o", binary]
        print("Compiling:", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit("Compilation failed.")

    subprocess.run([binary])
    sys.exit(0)

# ------------------------------------------------------------------
# Python pipelines: fused / gemm / unfused
# ------------------------------------------------------------------
if args.pipeline == "fused":
    from knn_kokkos import run_knn_pipeline
elif args.pipeline == "gemm":
    from gemm_knn_kokkos import run_knn_pipeline
else:  # unfused
    from unfused_knn_kokkos import run_knn_pipeline

print(f"{'Dataset':>10}  {'k':>6}  {'Total (s)':>12}")
print(f"{'-------':>10}  {'–':>6}  {'---------':>12}")

for d in ds:
    for k in ks:
        np.random.seed(42)
        X_np = np.random.randn(N, m, d).astype(np.float64)
        X    = torch.from_numpy(X_np)
        Xn   = torch.empty((N, m), dtype=torch.float64)
        Dloc = torch.zeros((N, m, b), dtype=torch.float64)
        Gidx = torch.full((N, m, k), -1,                             dtype=torch.int32)
        Gdst = torch.full((N, m, k), torch.finfo(torch.float64).max, dtype=torch.float64)

        for i in range(3):
            t0 = time.time()
            run_knn_pipeline(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx)
            t1 = time.time()

        elapsed_s = t1 - t0
        name = f"Gauss{d}"
        print(f"{name:>10}  {k:>6}  {elapsed_s:>12.2f}")
        sys.stdout.flush()
