# Pykokkos-KNN

A [PyKokkos](https://github.com/kokkos/pykokkos) implementation of leaf-level exact
k-nearest-neighbor graph construction, written to run on TACC GPU nodes (developed and
benchmarked on **Vista**, GH200 / `sm_90`).

The workload is the dense leaf kernel used inside a randomized projection forest: given
`N` independent leaves of `m` points in `d` dimensions, compute the `k` nearest neighbors
of every point within its own leaf. Distances use the expanded form
`||x-y||² = ||x||² + ||y||² - 2·xᵀy`, computed block-by-block (`b×b` tiles) so that the
`m × m` distance matrix is never materialized: only a `m × b` staging buffer `Dloc` and the
running top-k lists live in memory.

Each leaf is mapped to one Kokkos team (`league_rank() == leaf index`), so the batch
dimension `N` is what fills the GPU's SMs. Within a team the pipeline runs:

1. **norms** — `Xn[i] = ||x_i||²`
2. **diagonal blocks** — pairwise distances inside each `b×b` diagonal tile
3. **row top-k** — max-heap style scan producing local candidates `Ldst`/`Lidx`
4. **merge** — bitonic merge of the local candidates into the global list `Gdst`/`Gidx`
5. **off-diagonal (hblk) blocks** — repeat 2–4 for the strictly-upper block panels

The goal of the project is a like-for-like comparison against the hand-written CUDA
implementation (`FIKNN_gpu_dense` / `dfi_leafknn`) from **pyrknn** — "Scalable kNN Graph
Construction with Heterogeneous Architectures" — including the paper's Table 8, plus a
roofline-style "napkin math" model to explain where the PyKokkos versions sit.

## Pipelines

All pipelines expose `run_knn_pipeline(...)` and are interchangeable from the harnesses.

| Module | What it is |
| --- | --- |
| `knn_kokkos.py` | **Main fused pipeline.** The entire per-leaf pipeline is one hierarchical `TeamPolicy` kernel with team scratch memory for the bitonic merge buffers. Requires `k` a power of two, `k ≤ b`. |
| `knn_kokkos_keqb.py` | Fused variant specialized for `k == b` (no separate `Ldst`/`Lidx`; `Gdst`/`Gidx` are `2k` wide and merged in place). |
| `unfused_knn_kokkos.py` | Same algorithm split into many flat `parallel_for` launches — the baseline that shows the cost of kernel-launch and global-memory round trips. |
| `gemm_knn_kokkos.py` | Unfused, but off-diagonal block distances come from batched `torch.bmm` (cuBLAS) instead of a PyKokkos kernel; the top-k/merge stages stay in PyKokkos. |
| `cpp` (via `cpp_bench.cu`) | The reference CUDA implementation from `../pyrknn`, compiled and timed through the same harness. |

## Layout

```
knn_kokkos.py            fused pipeline (primary)
knn_kokkos_keqb.py       fused pipeline, k == b specialization
unfused_knn_kokkos.py    multi-kernel pipeline
gemm_knn_kokkos.py       multi-kernel pipeline with cuBLAS batched GEMM
bench.py                 sweep harness -> runtimes.txt
plot_runtimes.py         plots runtimes.txt (+ napkin-math model) -> runtimes.png
test_knn.py              correctness check against a numpy brute-force ground truth
run_table8.py            Table 8 reproduction (M = 4M points, 2000 leaves)
cpp_bench.cu             C++/CUDA baseline wrapper around dfi_leafknn (sweeps)
table8_bench.cu          C++/CUDA baseline wrapper for Table 8
getruntimes.sh           scp runtimes.txt back from Vista
*_runtimes.txt           captured results; table8_results.txt, runtimes.png
pk_cpp/                  PyKokkos-generated C++/binaries (gitignored)
```

`compare_to_golden.py`, `test_dblk.py`, and `test_hblk.py` are leftovers from the
pre-fusion code and import kernels (`compute_dist_dblk`, …) that no longer exist in
`knn_kokkos.py`; use `test_knn.py` instead.

## Setup on TACC

Everything runs on a single GPU node — get one interactively before running anything:

```bash
ssh mackx@vista.tacc.utexas.edu          # see ../tacc.sh
idev -p gh -N 1 -n 1 -t 02:00:00         # a GH200 node (check the queue name)
module load cuda/13.0
```

Then activate the Python environment (PyKokkos + PyTorch + numpy; matplotlib for plots).
A shared PyKokkos build lives at `/work/09661/gkk345/vista/pykokkos`; the local
development env is the conda env `knn311-torch`:

```bash
conda activate knn311-torch
# or, for a fresh env:  pip install torch numpy matplotlib && pip install -e <pykokkos checkout>
```

PyKokkos compiles each workunit on first use into `pk_cpp/`, so the **first** run of a
pipeline takes a few minutes; later runs reuse the cached binaries. Delete `pk_cpp/` to
force a rebuild after editing a kernel signature.

The C++ baseline additionally needs `nvcc` and the sibling **pyrknn** checkout at
`../pyrknn` — the harnesses compile
`../pyrknn/GeMM/pysrc/filknn/dense/dfiknn_test.cu` with
`-gencode arch=compute_90,code=sm_90 -lcublas` automatically.

## Running

### Correctness

Compares against a numpy brute-force kNN for `k = 2, 4, 16, 32` (ties are reported
separately from real errors):

```bash
python test_knn.py                              # knn_kokkos
python test_knn.py --pipeline knn_kokkos_keqb
python test_knn.py --pipeline unfused_knn_kokkos
```

### Benchmarks

`bench.py` sweeps one parameter, runs each point 3× (first two are warmup), and writes
`runtimes.txt` in the format `plot_runtimes.py` expects.

```bash
python bench.py knn_kokkos                       # sweep N, default values
python bench.py all --sweep d                    # every pipeline, sweep d
python bench.py all --sweep N --large            # extend the sweep to N = 16384
python bench.py knn_kokkos --custom 128 256 512  # explicit sweep values
python bench.py cpp                              # C++/CUDA baseline only
```

Fixed defaults live at the top of `bench.py`: `m=2000`, `d=70`, `k=2`, `b=32`, and
`N=512` when sweeping `d`. Note `knn_kokkos_keqb` is always run with `b = k`.

### Plots

```bash
python plot_runtimes.py     # reads ./runtimes.txt, writes ./runtimes.png
```

The plot overlays the measured curves with the napkin-math roofline variants
(compute-only, compute+memory with infinite cache, and with a 10% L2 miss rate) using
GH200 numbers hard-coded in the script — 33.5 TFLOP/s, 4 TB/s HBM, 132 SMs.

### Table 8 reproduction

4M points split into 2000 leaves of 2000, over `d ∈ {4, 16, 64}` and `k ∈ {16, 64}`:

```bash
python run_table8.py fused        # knn_kokkos (auto-switches to keqb when k == b)
python run_table8.py gemm
python run_table8.py unfused
python run_table8.py original     # compiles and runs table8_bench.cu (pyrknn CUDA)
```

Recorded output is in `table8_results.txt`.

### Pulling results back

```bash
./getruntimes.sh            # scp runtimes.txt from $WORK/vista/Pykokkos-KNN on Vista
```

## Parameters

| Symbol | Meaning | Constraints |
| --- | --- | --- |
| `N` | number of leaves / datasets = Kokkos league size | throughput saturates around the SM count (132 on GH200) |
| `m` | points per leaf | |
| `d` | feature dimension | |
| `k` | neighbors per point | power of two; `k ≤ b` for `knn_kokkos`, `k == b` for `knn_kokkos_keqb` |
| `b` | block/tile width | typically 32 (one warp) |

Inputs are `float32` torch tensors shaped `(N, m, d)`; outputs are `Gidx` / `Gdst`
`(N, m, k)` holding neighbor indices and squared distances.
