import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--unfused", action="store_true")
args = parser.parse_args()

# -----------------------------
# parameters
# -----------------------------
N = 2    # batch size — tests that league_rank() correctly isolates datasets
m = 100  # 3 blocks of b=32, 1 block of 4 — exercises both diagonal and hblk paths
d = 8
k = 2
b = 32

np.random.seed(42)
X_np = np.random.randint(0, 8, size=(N, m, d)).astype(np.float64)

# -----------------------------
# ground truth (numpy brute force, per dataset)
# -----------------------------
def knn_brute(X_np, k):
    """X_np: (N, m, d) — returns (N, m, k) sorted nearest indices."""
    N, m, _ = X_np.shape
    result = np.zeros((N, m, k), dtype=np.int32)
    for n in range(N):
        diff = X_np[n, :, None, :] - X_np[n, None, :, :]  # (m, m, d)
        D = np.sum(diff ** 2, axis=-1)
        np.fill_diagonal(D, np.inf)
        result[n] = np.argsort(D, axis=1)[:, :k]
    return result

gt_idx = knn_brute(X_np, k)

# -----------------------------
# helpers
# -----------------------------
if args.unfused:
    from unfused_knn_kokkos import run_knn_pipeline
else:
    from knn_kokkos import run_knn_pipeline

def fresh_tensors(N, m, b, k):
    X    = torch.from_numpy(X_np.copy())
    Xn   = torch.empty((N, m), dtype=torch.float64)
    Dloc = torch.zeros((N, m, b), dtype=torch.float64)
    Gidx = torch.full((N, m, k + 1), -1,                             dtype=torch.int32)
    Gdst = torch.full((N, m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64)
    Lidx = torch.full((N, m, k + 1), -1,                             dtype=torch.int32)
    Ldst = torch.full((N, m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64)
    return X, Xn, Dloc, Gidx, Gdst, Lidx, Ldst

def extract_knn(Gidx, Gdst, N, m, k):
    Gidx_np = Gidx.numpy().astype(np.int32)   # (N, m, k+1)
    Gdst_np = Gdst.numpy()
    result = np.full((N, m, k), -1, dtype=np.int32)
    for n in range(N):
        for i in range(m):
            order = np.argsort(Gdst_np[n, i])
            filled = 0
            for slot in order:
                if filled == k:
                    break
                idx = Gidx_np[n, i, slot]
                if idx >= 0 and idx != i:
                    result[n, i, filled] = idx
                    filled += 1
    return result

def compare(pred, gt, X_np, N, m, k, label):
    D_full = np.sum((X_np[:, :, None, :] - X_np[:, None, :, :]) ** 2, axis=-1)  # (N, m, m)
    real_errors, ties = [], []
    for n in range(N):
        for i in range(m):
            if set(pred[n, i].tolist()) == set(gt[n, i].tolist()):
                continue
            pred_dists = sorted(D_full[n, i, pred[n, i]])
            gt_dists   = sorted(D_full[n, i, gt[n, i]])
            if np.allclose(pred_dists, gt_dists):
                ties.append((n, i))
            else:
                real_errors.append((n, i, sorted(pred[n, i].tolist()), pred_dists,
                                          sorted(gt[n, i].tolist()),   gt_dists))
    status = "PASS" if not real_errors else f"FAIL ({len(real_errors)} errors)"
    exact = N * m - len(ties) - len(real_errors)
    print(f"[{label}] {status}  —  ties={len(ties)}  errors={len(real_errors)}  ({exact}/{N*m} exact)")
    for n, i, pp, pd, gp, gd in real_errors[:5]:
        print(f"  dataset={n} row={i:3d}  pred={pp} dists={pd}")
        print(f"                      gt  ={gp} dists={gd}")

# -----------------------------
# run
# -----------------------------
X, Xn, Dloc, Gidx, Gdst, Lidx, Ldst = fresh_tensors(N, m, b, k)
run_knn_pipeline(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx)
pred = extract_knn(Gidx, Gdst, N, m, k)

print(f"N={N}  m={m}  d={d}  k={k}  b={b}")
compare(pred, gt_idx, X_np, N, m, k, "batched")
