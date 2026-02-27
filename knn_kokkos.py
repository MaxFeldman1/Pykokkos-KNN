import pykokkos as pk
import torch
import numpy as np
import time
import math

# -----------------------------
# parameters
# -----------------------------
m = 10_000
d = 7
k = 4
b = 32
l = math.ceil(m / b)
device = "cpu"          # ← change to "cuda" for GPU


# -----------------------------
# data
# -----------------------------
np.random.seed(0)
X_np = np.random.randint(0, 8, size=(m, d)).astype(np.float64)

X         = torch.from_numpy(X_np.copy()).to(device)
Xn        = torch.empty(m, dtype=torch.float64, device=device)
Dloc      = torch.zeros((m, b), dtype=torch.float64, device=device)

# Global best indices and distances
Gidx      = torch.full((m, k + 1), -1, dtype=torch.int32, device=device)
Gdst      = torch.full((m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64, device=device)

# local best indices and distances
idx       = torch.full((m, k + 1), -1, dtype=torch.int32, device=device)
best_dist = torch.full((m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64, device=device)

Lidx      = torch.full((m, k + 1), -1, dtype=torch.int32, device=device)
Ldst      = torch.full((m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64, device=device)

# G         = torch.zeros((m, m), dtype=torch.int32, device=device)

# -----------------------------
# kernels
# -----------------------------
@pk.workunit
def compute_norm(i, X, Xn, d):
    s: pk.float64 = 0.0
    for j in range(d):
        s += X[i][j] * X[i][j]
    Xn[i] = s


"""
    only need to compute the upper right triangle due to symmetry and zeros on diagonal
    index by how many 
"""
@pk.workunit
def compute_dist_dblk(lin, X, Xn, Dloc, d, b, blknum, blksize):
    # lin \in [0, blksize(blksize - 1)]
    # index arithmetic to ensure lin \mapsto (off0, off1) is a bijection between {0, ..,  blksize(blksize-1)} and the upper right coordinates of a blksize matrix
    off0: pk.int32 = lin // blksize
    off1: pk.int32 = lin % blksize
    # remap coordinates to top right triangle
    # if off1-off0 >= blksize - 1:
    flip0: pk.int32 = off0 + off1 >= blksize-1
    off0 = (blksize - 1 - off0) * flip0 + off0 * (1 - flip0)
    off1 = off1 * flip0 + (blksize - 1 - off1) * (1 - flip0)

    i: pk.int32 = off0 + b * blknum
    j: pk.int32 = off1 + b * blknum # j may be bigger than b, mod by b when storing in Dloc
    dot: pk.float64 = 0.0
    for t in range(d):
        dot += X[i][t] * X[j][t]
    Dloc[i][j%b] = -2.0 * dot + Xn[i] + Xn[j]
    

@pk.workunit
def compute_dist_hblk(lin, X, Xn, Dloc, d, b, blksize, blknum):
    off0: pk.int32 = lin //  blksize
    off1: pk.int32 = lin %  blksize
    i: pk.int32 = off0 + b * (blknum-1)
    j: pk.int32 = off1 + b * blknum
    dot: pk.float64 = 0.0
    for t in range(d):
        dot += X[i][t] * X[j][t]
    Dloc[off1][off0] = -2.0 * dot + Xn[i] + Xn[j]
    # Dloc[off1][off0] = i * 100 + j

@pk.workunit
def merge_topk(i, Gdst, Gidx, Ldst, Lidx, k, offset):
    i = i+offset
    j: pk.int32 = 0
    for j in range(k+1):
        dst: pk.float64 = Ldst[i][j]
        idx: pk.int32 = Lidx[i][j]

        worst: pk.int32 = 0
        t: pk.int32 = 0
        for t in range(1, k + 1):
            if Gdst[i][t] > Gdst[i][worst]:
                worst = t

        if dst < Gdst[i][worst]:
            Gdst[i][worst] = dst
            Gidx[i][worst] = idx


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
def topk_row_dblk(i, Dloc, Lidx, Ldst, m, k, b):
    im: pk.int32 = i % b
    id: pk.int32 = i - im
    j: pk.int32 = 0

    m_top_of_range: pk.int32 = m < id + b
    top_of_range: pk.int32 = (m * m_top_of_range) + ((id + b) * (1 - m_top_of_range))

    for j in range(id, top_of_range):
        jm: pk.int32 = j % b

        i_first: pk.int32 = im <= jm
        _idx0: pk.int32 = (i * i_first) + (j * (1 - i_first))
        _idx1: pk.int32 = (jm * i_first) + (im * (1 - i_first))
        val: pk.float64 = Dloc[_idx0][_idx1]

        worst: pk.int32 = 0
        t: pk.int32 = 0
        for t in range(1, k + 1):
            if Ldst[i][t] > Ldst[i][worst]:
                worst = t

        if val < Ldst[i][worst]:
            Ldst[i][worst] = val
            Lidx[i][worst] = j


@pk.workunit
def topk_row_hblk(im, Dloc, Lidx, Ldst, k, b, blksize, blknum):
    i: pk.int32 = im + b * (blknum-1)
    jm: pk.int32 = 0
    for jm in range(blksize):
        j: pk.int32 = jm + b * blknum
        val: pk.float64 = Dloc[jm][im]
        worst: pk.int32 = 0
        t: pk.int32 = 0
        prop: pk.int32 = 0
        for t in range(1, k + 1):
            prop = int(Ldst[i][t] > Ldst[i][worst])
            worst = t * prop + worst * (1 - prop)

        prop = int(val < Ldst[i][worst])
        Ldst[i][worst] = val * prop + Ldst[i][worst] * (1 - prop)
        Lidx[i][worst] = j * prop + Lidx[i][worst] * (1 - prop)


@pk.workunit
def topk_col_hblk(jm, Dloc, Lidx, Ldst, k, b, blknum):
    j: pk.int32 = jm + b * blknum
    i: pk.int32 = 0
    for i in range(b * (blknum-1), b * blknum):
        im: pk.int32 = i - b * (blknum-1)
        val: pk.float64 = Dloc[jm][im]

        worst: pk.int32 = 0
        t: pk.int32 = 0
        prop: pk.int32 = 0
        for t in range(1, k + 1):
            prop =  int(Ldst[j][t] > Ldst[j][worst])
            worst = t * prop + worst * (1 - prop)
        
        prop = int(val < Ldst[j][worst])
        Ldst[j][worst] = val * prop + Ldst[j][worst] * (1 - prop)
        Lidx[j][worst] = i * prop + Lidx[j][worst] * (1 - prop)



@pk.workunit
def build_G(i, Gidx, G, k):
    t: pk.int32 = 0
    for t in range(k + 1):
        j: pk.int32 = Gidx[i][t]
        if j >= 0 and j != i:
            G[i][j] = 1


# -----------------------------
# run
# -----------------------------
t0 = time.time()

pk.parallel_for(m, compute_norm, X=X, Xn=Xn, d=d)
pk.fence()

# diag blocks compute distances
for i in range(l):
    # dispatch [compute, update Global]
    blksize = min((i+1) * b, m) - i * b
    pk.parallel_for(blksize * (blksize) , compute_dist_dblk, X=X, Xn=Xn, Dloc=Dloc, d=d, b=b, blknum=i, blksize=blksize)

pk.fence()

# diag blocks update kNN
pk.parallel_for(m, topk_row_dblk, Dloc=Dloc, Lidx=Lidx, Ldst=Ldst, m=m, k=k, b=b)
pk.fence()

# Global merge
pk.parallel_for(m, merge_topk, Gdst=Gdst, Gidx=Gidx, Ldst=Ldst, Lidx=Lidx, k=k, offset=0)
pk.fence()

# flush locals
Dloc.fill_(-1)
Lidx.fill_(-1)
Ldst.fill_(torch.finfo(torch.float64).max)
pk.fence()

for i in range(1, l):
    blksize = m - b * i
    pk.parallel_for(blksize*b, compute_dist_hblk, X=X, Xn=Xn, Dloc=Dloc, d=d, b=b, blksize=blksize, blknum=i)
    pk.fence()

    # compute local kNN
    pk.parallel_for(b, topk_row_hblk, Dloc=Dloc, Lidx=Lidx, Ldst=Ldst, k=k, b=b, blksize=blksize, blknum=i)
    pk.parallel_for(blksize, topk_col_hblk, Dloc=Dloc, Lidx=Lidx, Ldst=Ldst, k=k, b=b, blknum=i)
    pk.fence()

    # Merge local kNN with global kNN
    pk.parallel_for(m - b * (i-1), merge_topk, Gdst=Gdst, Gidx=Gidx, Ldst=Ldst, Lidx=Lidx, k=k, offset=(b*(i-1)))
    pk.fence()

    # flush locals
    Dloc.fill_(-1)
    Lidx.fill_(-1)
    Ldst.fill_(torch.finfo(torch.float64).max)

pk.fence()

t1 = time.time()

# pk.parallel_for(m, build_G, Gidx=Gidx, G=G, k=k)
# pk.fence()


# -----------------------------
# copy back for display
# -----------------------------
# G_host = G.cpu().numpy()

print("Coordinate matrix")
print(X_np)

# print("\nkNN Graph")
# print(G_host)

print("\nExecution time:", (t1 - t0) * 1000, "ms")
