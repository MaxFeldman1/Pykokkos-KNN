import numpy as np
import torch

# -----------------------------
# parameters
# -----------------------------
m = 96   # 3 blocks of b=32 — exercises both diagonal and hblk paths
d = 8
k = 2
b = 32

np.random.seed(42)
X_np = np.random.randint(0, 8, size=(m, d)).astype(np.float64)

# -----------------------------
# ground truth (numpy brute force)
# -----------------------------
def knn_brute(X_np, k):
    diff = X_np[:, None, :] - X_np[None, :, :]   # (m, m, d)
    D    = np.sum(diff ** 2, axis=-1)              # (m, m)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]            # (m, k) sorted nearest-first

gt_idx = knn_brute(X_np, k)

# -----------------------------
# helpers
# -----------------------------
from knn_kokkos import run_knn_pipeline

def fresh_tensors(m, b, k):
    X    = torch.from_numpy(X_np.copy())
    Xn   = torch.empty(m, dtype=torch.float64)
    Dloc = torch.zeros((m, b), dtype=torch.float64)
    Gidx = torch.full((m, k + 1), -1,                             dtype=torch.int32)
    Gdst = torch.full((m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64)
    Lidx = torch.full((m, k + 1), -1,                             dtype=torch.int32)
    Ldst = torch.full((m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64)
    return X, Xn, Dloc, Gidx, Gdst, Lidx, Ldst

def extract_knn(Gidx, Gdst, k):
    """Sort each row's k+1 slots by distance, return top-k valid (non-(-1), non-self) indices."""
    Gidx_np = Gidx.numpy().astype(np.int32)
    Gdst_np = Gdst.numpy()
    m = Gidx_np.shape[0]
    result = np.full((m, k), -1, dtype=np.int32)
    for i in range(m):
        order = np.argsort(Gdst_np[i])
        filled = 0
        for slot in order:
            if filled == k:
                break
            idx = Gidx_np[i, slot]
            if idx >= 0 and idx != i:
                result[i, filled] = idx
                filled += 1
    return result

def compare(pred, gt, label, X_np):
    """
    Compare predicted vs ground truth neighbours.
    A mismatch is only a real error if the predicted neighbours have strictly
    worse distances than the ground truth neighbours — differing indices at the
    same distance are valid tie-breaks.
    """
    D_full = np.sum((X_np[:, None, :] - X_np[None, :, :]) ** 2, axis=-1)

    real_errors, ties = [], []
    for i in range(pred.shape[0]):
        if set(pred[i].tolist()) == set(gt[i].tolist()):
            continue
        pred_dists = sorted(D_full[i, pred[i]])
        gt_dists   = sorted(D_full[i, gt[i]])
        if np.allclose(pred_dists, gt_dists):
            ties.append((i, sorted(pred[i].tolist()), sorted(gt[i].tolist())))
        else:
            real_errors.append((i, sorted(pred[i].tolist()), pred_dists,
                                    sorted(gt[i].tolist()),  gt_dists))

    status = "PASS" if not real_errors else f"FAIL ({len(real_errors)} errors)"
    print(f"[{label}] {status}  —  ties={len(ties)}  errors={len(real_errors)}  "
          f"({m - len(ties) - len(real_errors)}/{m} exact matches)")
    for i, pp, pd, gp, gd in real_errors[:5]:
        print(f"  row {i:3d}  pred={pp} dists={pd}")
        print(f"           gt  ={gp} dists={gd}")

# -----------------------------
# run pipeline
# -----------------------------
X, Xn, Dloc, Gidx, Gdst, Lidx, Ldst = fresh_tensors(m, b, k)
run_knn_pipeline(m, d, k, b, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx)
pred = extract_knn(Gidx, Gdst, k)

# -----------------------------
# report
# -----------------------------
print(f"m={m}  d={d}  k={k}  b={b}")
compare(pred, gt_idx, "fused", X_np)
