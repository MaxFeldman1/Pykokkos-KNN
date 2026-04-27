import sys
import argparse
import numpy as np
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument("pipeline", nargs="?", default="knn_kokkos",
                    choices=["knn_kokkos", "unfused_knn_kokkos", "gemm_knn_kokkos"],
                    help="Which pipeline to benchmark")
args = parser.parse_args()

if args.pipeline == "unfused_knn_kokkos":
    from unfused_knn_kokkos import run_knn_pipeline
elif args.pipeline == "gemm_knn_kokkos":
    from gemm_knn_kokkos import run_knn_pipeline
else:
    from knn_kokkos import run_knn_pipeline

print(f"Benchmarking: {args.pipeline}")

# -----------------------------
# fixed parameters
# -----------------------------
m = 2000
d = 70
k = 2
b = 32

Ns = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 132, 133, 144, 160, 192, 256, 384, 512, 640, 768, 896, 1024]

np.random.seed(0)

lines = [f"m={m}", f"d={d}", f"k={k}", f"b={b}", ""]


for N in Ns:
    X_np = np.random.randint(0, 8, size=(N, m, d)).astype(np.float64)
    X    = torch.from_numpy(X_np)
    Xn   = torch.empty((N, m), dtype=torch.float64)
    Dloc = torch.zeros((N, m, b), dtype=torch.float64)
    Gidx = torch.full((N, m, k + 1), -1,                             dtype=torch.int32)
    Gdst = torch.full((N, m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64)
    Lidx = torch.full((N, m, k + 1), -1,                             dtype=torch.int32)
    Ldst = torch.full((N, m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64)

    for i in range(3):
        t0 = time.time()
        run_knn_pipeline(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx)
        t1 = time.time()

    ms = (t1 - t0) * 1000
    print(f"N={N}\n{ms:.3f}")

    lines.append(f"N={N}")
    lines.append(f"{ms}")
    lines.append("")

with open("runtimes.txt", "w") as f:
    f.write("\n".join(lines))

print("\nWrote runtimes.txt")
