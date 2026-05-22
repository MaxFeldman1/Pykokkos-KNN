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
parser.add_argument("--sweep", default="N", choices=["N", "d"],
                    help="Variable to sweep: N (batch size) or d (dimension)")
args = parser.parse_args()

# -----------------------------
# fixed parameters
# -----------------------------
m = 2000
d = 70
k = 2
b = 32

Ns      = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 132, 133, 144, 160, 192, 256, 384, 512, 640, 768, 896, 1024]
N_fixed = 500
ds      = [4, 8, 16, 32, 64, 128, 256, 512]

print(f"Benchmarking: {args.pipeline}  sweep={args.sweep}")

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

    if args.sweep == "d":
        sweep_vals = ds
        hdr = [f"N={N_fixed}", f"m={m}", f"k={k}", f"b={b}", ""]
        out_file = "cpp_d_runtimes.txt"
    else:
        sweep_vals = Ns
        hdr = [f"m={m}", f"d={d}", f"k={k}", f"b={b}", ""]
        out_file = "cpp_runtimes.txt"

    lines = hdr
    for val in sweep_vals:
        run_N = N_fixed if args.sweep == "d" else val
        run_d = val     if args.sweep == "d" else d
        result = subprocess.run(
            [binary, str(run_N), str(m), str(run_d), str(k)],
            capture_output=True, text=True,
        )
        label = f"d={val}" if args.sweep == "d" else f"N={val}"
        if result.returncode != 0:
            print(f"cpp_bench failed for {label}:\n{result.stderr}")
            continue
        out = result.stdout.strip().splitlines()
        ms_str = out[-1]
        print(f"{label}\n{ms_str}")
        lines.append(label)
        lines.append(ms_str)
        lines.append("")

    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {out_file}")
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

if args.sweep == "d":
    sweep_vals = ds
    lines    = [f"N={N_fixed}", f"m={m}", f"k={k}", f"b={b}", ""]
    out_file = "d_runtimes.txt"
else:
    sweep_vals = Ns
    lines    = [f"m={m}", f"d={d}", f"k={k}", f"b={b}", ""]
    out_file = "runtimes.txt"

for val in sweep_vals:
    run_N = N_fixed if args.sweep == "d" else val
    run_d = val     if args.sweep == "d" else d
    label = f"d={val}" if args.sweep == "d" else f"N={val}"

    X_np = np.random.randint(0, 8, size=(run_N, m, run_d)).astype(np.float32)
    X    = torch.from_numpy(X_np)
    Xn   = torch.empty((run_N, m), dtype=torch.float32)
    Dloc = torch.zeros((run_N, m, b), dtype=torch.float32)

    if args.pipeline == "knn_kokkos_keqb":
        Gdst = torch.full((run_N, m, 2 * k), torch.finfo(torch.float32).max, dtype=torch.float32)
        Gidx = torch.full((run_N, m, 2 * k), -1,                             dtype=torch.int32)
        call_args = (run_N, m, run_d, k, b, X, Xn, Dloc, Gdst, Gidx)
    else:
        Gdst = torch.full((run_N, m, k),     torch.finfo(torch.float32).max, dtype=torch.float32)
        Gidx = torch.full((run_N, m, k),     -1,                             dtype=torch.int32)
        Ldst = torch.full((run_N, m, k),     torch.finfo(torch.float32).max, dtype=torch.float32)
        Lidx = torch.full((run_N, m, k),     -1,                             dtype=torch.int32)
        call_args = (run_N, m, run_d, k, b, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx)

    for i in range(3):
        t0 = time.time()
        run_knn_pipeline(*call_args)
        t1 = time.time()

    ms = (t1 - t0) * 1000
    print(f"{label}\n{ms:.3f}")

    lines.append(label)
    lines.append(f"{ms}")
    lines.append("")

with open(out_file, "w") as f:
    f.write("\n".join(lines))

print(f"\nWrote {out_file}")
