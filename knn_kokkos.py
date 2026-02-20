import pykokkos as pk
import numpy as np
import time


# -----------------------------
# parameters
# -----------------------------
m = 10
d = 7
k = 3



# -----------------------------
# data
# -----------------------------
np.random.seed(0)
X_np = np.random.randint(0, 5, size=(m, d)).astype(np.float64)
"""
 # OLD
X  = pk.View([m, d], dtype=pk.double)
Xn = pk.View([m], dtype=pk.double)
D  = pk.View([m, m], dtype=pk.double)
idx = pk.View([m, k+1], dtype=pk.int32)
G  = pk.View([m, m], dtype=pk.int32)

pk.deep_copy(X, X_np)
"""

X = X_np.copy()
Xn = np.empty(m, dtype=pk.double)
D = np.empty((m,m), dtype=pk.double)
idx = np.empty((m, k+1), dtype=pk.int32)
G = np.empty((m,m), dtype=pk.int32)

# -----------------------------
# kernels
# -----------------------------
@pk.workunit
def compute_norm(i, X, Xn, d):
    s = 0.0
    for j in range(d):
        s += X[i][j] * X[i][j]
    Xn[i] = s


@pk.workunit
def compute_dist(i, j, X, Xn, D, d):
    dot = 0.0
    for t in range(d):
        dot += X[i][t] * X[j][t]
    D[i][j] = -2.0 * dot + Xn[i] + Xn[j]


@pk.workunit
def topk_row(i, D, idx, m, k):
    best_dist = [1e300] * (k+1)
    best_idx  = [-1] * (k+1)

    for j in range(m):
        val = D[i][j]

        # find worst slot
        worst = 0
        for t in range(1, k+1):
            if best_dist[t] > best_dist[worst]:
                worst = t

        if val < best_dist[worst]:
            best_dist[worst] = val
            best_idx[worst] = j

    for t in range(k+1):
        idx[i][t] = best_idx[t]


@pk.workunit
def zero_G(i, j, G):
    G[i][j] = 0


@pk.workunit
def build_G(i, idx, G, k):
    for t in range(k+1):
        j = idx[i][t]
        if j >= 0 and j != i:
            G[i][j] = 1


# -----------------------------
# run
# -----------------------------
t0 = time.time()

pk.parallel_for(m, compute_norm, X=X, Xn=Xn, d=d)

pk.parallel_for(
    pk.MDRangePolicy([0,0], [m,m]),
    compute_dist,
    X=X, Xn=Xn, D=D, d=d
)

pk.parallel_for(m, topk_row, D=D, idx=idx, m=m, k=k)

pk.parallel_for(
    pk.MDRangePolicy([0,0], [m,m]),
    zero_G,
    G=G
)

pk.parallel_for(m, build_G, idx=idx, G=G, k=k)

pk.fence()

t1 = time.time()


# -----------------------------
# copy back for display
# -----------------------------

"""
 # OLD 
 D_host = pk.deep_copy(np.zeros((m,m)), D)
 G_host = pk.deep_copy(np.zeros((m,m), dtype=np.int32), G)
"""
D_host = D.copy()
G_host = G.copy()

print("Coordinate matrix")
print(X_np)

print("\nDistance squared matrix")
print(D_host)

print("\nkNN Graph")
print(G_host)

print("\nExecution time:", (t1 - t0)*1000, "ms")
