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

X = X_np.copy()
Xn = np.empty(m, dtype=np.float64)
D = np.empty((m,m), dtype=np.float64)
idx = np.full((m, k+1), -1, dtype=np.int32)
G = np.empty((m,m), dtype=np.int32)

# for topk_row workunit
    # Allocate outside the workunit — each row i gets its own k+1 scratch slots
best_dist = np.full((m, k + 1), 1e300, dtype=np.float64)

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

        # find the worst (largest distance) slot
        worst: pk.int32 = 0
        t: pk.int32 = 0
        for t in range(1, k + 1):
            if best_dist[i][t] > best_dist[i][worst]:
                worst = t

        # replace worst slot if current distance is smaller
        if val < best_dist[i][worst]:
            best_dist[i][worst] = val
            idx[i][worst] = j

@pk.workunit
def zero_G(lin, G, m):
    i: pk.int32 = lin // m
    j: pk.int32 = lin % m
    G[i][j] = 0


@pk.workunit
def build_G(i, idx, G, k):
    for t in range(k+1):
        j: pk.int32 = idx[i][t]
        if j >= 0 and j != i:
            G[i][j] = 1


# -----------------------------
# run
# -----------------------------
t0 = time.time()

pk.parallel_for(m, compute_norm, X=X, Xn=Xn, d=d)

pk.fence()

pk.parallel_for(
    m*m,
    compute_dist,
    X=X, Xn=Xn, D=D, d=d, m=m
)

pk.fence()

pk.parallel_for(m, topk_row, D=D, idx=idx, m=m, k=k, best_dist=best_dist)

pk.parallel_for(
    m*m,
    zero_G,
    G=G,
    m=m
)

pk.fence()

pk.parallel_for(m, build_G, idx=idx, G=G, k=k)

pk.fence()

t1 = time.time()


# -----------------------------
# copy back for display
# -----------------------------
D_host = D.copy()
G_host = G.copy()

print("Coordinate matrix")
print(X_np)

print("\nDistance squared matrix")
print(D_host)

print("\nkNN Graph")
print(G_host)

# print("\nNorms Squared")
# print(Xn)

print("\nExecution time:", (t1 - t0)*1000, "ms")
