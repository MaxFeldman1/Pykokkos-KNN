import pykokkos as pk
import torch
import numpy as np
import time

# -----------------------------
# parameters
# -----------------------------
m = 10_000
d = 7
k = 3
device = "cpu"          # ← change to "cuda" for GPU


# -----------------------------
# data
# -----------------------------
np.random.seed(0)
X_np = np.random.randint(0, 8, size=(m, d)).astype(np.float64)

X         = torch.from_numpy(X_np.copy()).to(device)
Xn        = torch.empty(m, dtype=torch.float64, device=device)
D         = torch.empty((m, m), dtype=torch.float64, device=device)
idx       = torch.full((m, k + 1), -1, dtype=torch.int32, device=device)
G         = torch.zeros((m, m), dtype=torch.int32, device=device)
best_dist = torch.full((m, k + 1), 1e300, dtype=torch.float64, device=device)


# -----------------------------
# kernels
# -----------------------------
@pk.workunit
def compute_norm(i, X, Xn, d):
    s: pk.float64 = 0.0
    for j in range(d):
        s += X[i][j] * X[i][j]
    Xn[i] = s


@pk.workunit
def compute_dist(lin, X, Xn, D, d, m):
    i: pk.int32 = lin // m
    j: pk.int32 = lin % m
    dot: pk.float64 = 0.0
    for t in range(d):
        dot += X[i][t] * X[j][t]
    D[i][j] = -2.0 * dot + Xn[i] + Xn[j]


@pk.workunit
def topk_row(i, D, idx, m, k, best_dist):
    j: pk.int32 = 0
    for j in range(m):
        val: pk.float64 = D[i][j]

        worst: pk.int32 = 0
        t: pk.int32 = 0
        for t in range(1, k + 1):
            if best_dist[i][t] > best_dist[i][worst]:
                worst = t

        if val < best_dist[i][worst]:
            best_dist[i][worst] = val
            idx[i][worst] = j


@pk.workunit
def build_G(i, idx, G, k):
    t: pk.int32 = 0
    for t in range(k + 1):
        j: pk.int32 = idx[i][t]
        if j >= 0 and j != i:
            G[i][j] = 1


# -----------------------------
# run
# -----------------------------
t0 = time.time()

pk.parallel_for(m, compute_norm, X=X, Xn=Xn, d=d)
pk.fence()

pk.parallel_for(m * m, compute_dist, X=X, Xn=Xn, D=D, d=d, m=m)
pk.fence()

pk.parallel_for(m, topk_row, D=D, idx=idx, m=m, k=k, best_dist=best_dist)
pk.fence()

pk.parallel_for(m, build_G, idx=idx, G=G, k=k)
pk.fence()

t1 = time.time()


# -----------------------------
# copy back for display
# -----------------------------
D_host = D.cpu().numpy()
G_host = G.cpu().numpy()

print("Coordinate matrix")
print(X_np)

print("\nDistance squared matrix")
print(D_host)

print("\nkNN Graph")
print(G_host)

print("\nExecution time:", (t1 - t0) * 1000, "ms")
