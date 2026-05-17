import sys
import os
import argparse
import subprocess
import numpy as np
import torch
import time

parser = argparse.ArgumentParser()
parser.add_argument("pipeline", nargs="?", default="knn_kokkos",
                    choices=["knn_kokkos", "knn_kokkos_keqb", "unfused_knn_kokkos", "gemm_knn_kokkos", "cpp"],
                    help="Which pipeline to benchmark")
args = parser.parse_args()

# -----------------------------
# fixed parameters
# -----------------------------
m = 2000
d = 70
k = 2
b = 32

Ns = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 132, 133, 144, 160, 192, 256, 384, 512, 640, 768, 896, 1024]

print(f"Benchmarking: {args.pipeline}")

# -----------------------------
# cpp path: compile + subprocess
# -----------------------------
if args.pipeline == "cpp":
    knn_dir = os.path.dirname(os.path.abspath(__file__))
    binary  = os.path.join(knn_dir, "cpp_bench")
    src     = os.path.join(knn_dir, "cpp_bench.cu")
    fiknn   = os.path.join(knn_dir, "../pyrknn/GeMM/pysrc/filknn/dense/dfiknn_test.cu")
    inc     = os.path.join(knn_dir, "../pyrknn/GeMM/pysrc/filknn/dense")

    needs_build = (
        not os.path.exists(binary)
        or os.path.getmtime(src)   > os.path.getmtime(binary)
        or os.path.getmtime(fiknn) > os.path.getmtime(binary)
    )
    if needs_build:
        compile_cmd = [
            "nvcc",
            f"-I{knn_dir}", f"-I{inc}",
            "-gencode", "arch=compute_90,code=sm_90",
            fiknn, src,
            "-O2", "-lcublas", "-o", binary,
        ]
        print("Compiling cpp_bench:", " ".join(compile_cmd))
        result = subprocess.run(compile_cmd)
        if result.returncode != 0:
            sys.exit("Compilation failed.")

    lines = [f"m={m}", f"d={d}", f"k={k}", f"b={b}", ""]
    for N in Ns:
        result = subprocess.run(
            [binary, str(N), str(m), str(d), str(k)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"cpp_bench failed for N={N}:\n{result.stderr}")
            continue
        # stdout is exactly "N=<n>\n<ms>\n"
        out = result.stdout.strip().splitlines()
        ms_str = out[-1]
        print(f"N={N}\n{ms_str}")
        lines.append(f"N={N}")
        lines.append(ms_str)
        lines.append("")

    with open("cpp_runtimes.txt", "w") as f:
        f.write("\n".join(lines))
    print("\nWrote cpp_runtimes.txt")
    sys.exit(0)

# -----------------------------
# Python pipelines
# -----------------------------
if args.pipeline == "unfused_knn_kokkos":
    from unfused_knn_kokkos import run_knn_pipeline
elif args.pipeline == "gemm_knn_kokkos":
    from gemm_knn_kokkos import run_knn_pipeline
elif args.pipeline == "knn_kokkos_keqb":
    from knn_kokkos_keqb import run_knn_pipeline_keqb as run_knn_pipeline
    b = k  # keqb kernel requires k == b
else:
    from knn_kokkos import run_knn_pipeline

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
