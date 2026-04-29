import pykokkos as pk
import torch
import numpy as np
import time
import math

# -----------------------------
# parameters
# -----------------------------
device = "cpu"

if __name__ == '__main__':
    N = 2
    m = 12_000
    d = 70
    k = 2
    b = 32

    np.random.seed(0)
    X_np = np.random.randint(0, 8, size=(N, m, d)).astype(np.float64)

    X    = torch.from_numpy(X_np.copy())
    Xn   = torch.empty((N, m), dtype=torch.float64)
    Dloc = torch.zeros((N, m, b), dtype=torch.float64)
    Gidx = torch.full((N, m, k + 1), -1,                             dtype=torch.int32)
    Gdst = torch.full((N, m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64)
    Lidx = torch.full((N, m, k + 1), -1,                             dtype=torch.int32)
    Ldst = torch.full((N, m, k + 1), torch.finfo(torch.float64).max, dtype=torch.float64)


# -----------------------------
# kernels
# -----------------------------
@pk.workunit
def compute_norm(i, X, Xn, d, m):
    n: pk.int32 = i // m
    im: pk.int32 = i % m
    s: pk.float64 = 0.0
    t: pk.int32 = 0
    for t in range(d):
        s += X[n][im][t] * X[n][im][t]
    Xn[n][im] = s


@pk.workunit
def compute_dist_dblk(team_member: pk.TeamMember, X, Xn, Dloc, d, b, blknum, blksize):
    # league_rank encodes (n, jm): n = lr // blksize, jm = lr % blksize
    lr: pk.int32 = team_member.league_rank()
    n: pk.int32 = lr // blksize
    jm: pk.int32 = lr % blksize
    j: pk.int32 = jm + b * blknum

    im: pk.int32 = 0
    for im in range(jm):
        i: pk.int32 = im + b * blknum

        def dot_product(t: int, acc: pk.Acc[pk.double]):
            acc += X[n][i][t] * X[n][j][t]

        dot: pk.float64 = pk.parallel_reduce(pk.TeamThreadRange(team_member, d), dot_product)

        if team_member.team_rank() == 0:
            Dloc[n][i][jm] = -2.0 * dot + Xn[n][i] + Xn[n][j]

        team_member.team_barrier()


@pk.workunit
def topk_row_dblk(i, Dloc, Lidx, Ldst, m, k, b):
    n: pk.int32 = i // m
    row: pk.int32 = i % m
    im: pk.int32 = row % b
    id_: pk.int32 = row - im
    j: pk.int32 = 0
    prop: pk.int32 = 0

    m_top: pk.int32 = m < id_ + b
    top_range: pk.int32 = m * m_top + (id_ + b) * (1 - m_top)

    for j in range(id_, top_range):
        jm: pk.int32 = j % b

        i_first: pk.int32 = im <= jm
        idx0: pk.int32 = row * i_first + j * (1 - i_first)
        idx1: pk.int32 = jm * i_first + im * (1 - i_first)
        val: pk.float64 = Dloc[n][idx0][idx1]

        worst: pk.int32 = 0
        t: pk.int32 = 0
        for t in range(1, k + 1):
            prop = Ldst[n][row][t] > Ldst[n][row][worst]
            worst = t * prop + worst * (1 - prop)

        prop = val < Ldst[n][row][worst]
        Ldst[n][row][worst] = val * prop + Ldst[n][row][worst] * (1 - prop)
        Lidx[n][row][worst] = j * prop + Lidx[n][row][worst] * (1 - prop)


@pk.workunit
def topk_row_hblk(i, Dloc, Lidx, Ldst, k, b, blksize, blknum):
    # i encodes (n, im): n = i // b, im = i % b
    n: pk.int32 = i // b
    im: pk.int32 = i % b
    row: pk.int32 = im + b * (blknum - 1)
    jm: pk.int32 = 0
    prop: pk.int32 = 0
    for jm in range(blksize):
        j: pk.int32 = jm + b * blknum
        val: pk.float64 = Dloc[n][jm][im]
        worst: pk.int32 = 0
        t: pk.int32 = 0
        for t in range(1, k + 1):
            prop = Ldst[n][row][t] > Ldst[n][row][worst]
            worst = t * prop + worst * (1 - prop)

        prop = val < Ldst[n][row][worst]
        Ldst[n][row][worst] = val * prop + Ldst[n][row][worst] * (1 - prop)
        Lidx[n][row][worst] = j * prop + Lidx[n][row][worst] * (1 - prop)


@pk.workunit
def topk_col_hblk(i, Dloc, Lidx, Ldst, k, b, blknum, blksize):
    # i encodes (n, jm): n = i // blksize, jm = i % blksize
    n: pk.int32 = i // blksize
    jm: pk.int32 = i % blksize
    j: pk.int32 = jm + b * blknum
    im: pk.int32 = 0
    prop: pk.int32 = 0
    for im in range(b):
        row: pk.int32 = im + b * (blknum - 1)
        val: pk.float64 = Dloc[n][jm][im]

        worst: pk.int32 = 0
        t: pk.int32 = 0
        for t in range(1, k + 1):
            prop = Ldst[n][j][t] > Ldst[n][j][worst]
            worst = t * prop + worst * (1 - prop)

        prop = val < Ldst[n][j][worst]
        Ldst[n][j][worst] = val * prop + Ldst[n][j][worst] * (1 - prop)
        Lidx[n][j][worst] = row * prop + Lidx[n][j][worst] * (1 - prop)


@pk.workunit
def merge_topk(i, Gdst, Gidx, Ldst, Lidx, k, offset, count):
    # i encodes (n, local_row): n = i // count, row = i % count + offset
    n: pk.int32 = i // count
    row: pk.int32 = i % count + offset
    s: pk.int32 = 0
    prop: pk.int32 = 0
    for s in range(k + 1):
        dst: pk.float64 = Ldst[n][row][s]
        idx: pk.int32 = Lidx[n][row][s]

        worst: pk.int32 = 0
        t: pk.int32 = 0
        for t in range(1, k + 1):
            prop = Gdst[n][row][t] > Gdst[n][row][worst]
            worst = t * prop + worst * (1 - prop)

        prop = dst < Gdst[n][row][worst]
        Gdst[n][row][worst] = dst * prop + Gdst[n][row][worst] * (1 - prop)
        Gidx[n][row][worst] = idx * prop + Gidx[n][row][worst] * (1 - prop)


@pk.workunit
def flush_local(i, Ldst, Lidx, k, m):
    total: pk.int32 = m * (k + 1)
    n: pk.int32 = i // total
    rem: pk.int32 = i % total
    row: pk.int32 = rem // (k + 1)
    col: pk.int32 = rem % (k + 1)
    Ldst[n][row][col] = 1.7976931348623157e+308
    Lidx[n][row][col] = -1


@pk.workunit
def flush_dloc(i, Dloc, b, m):
    total: pk.int32 = m * b
    n: pk.int32 = i // total
    rem: pk.int32 = i % total
    row: pk.int32 = rem // b
    col: pk.int32 = rem % b
    Dloc[n][row][col] = -1.0


def compute_dist_hblk_gemm(X, Xn, Dloc, b, hblk, blksize):
    # Computes off-diagonal block distances via batched GEMM.
    # Block A (rows): X[:, start_i:start_i+b, :]        shape (N, b, d)
    # Block B (cols): X[:, start_j:start_j+blksize, :]  shape (N, blksize, d)
    # Result stored as Dloc[:, :blksize, :b] where
    #   Dloc[n][jm][im] = ||X[n][start_i+im] - X[n][start_j+jm]||^2
    start_i = b * (hblk - 1)
    start_j = b * hblk
    A = X[:, start_i:start_i + b, :]             # (N, b, d)
    B = X[:, start_j:start_j + blksize, :]        # (N, blksize, d)

    Xn_A = Xn[:, start_i:start_i + b]             # (N, b)
    Xn_B = Xn[:, start_j:start_j + blksize]       # (N, blksize)

    # Write dots directly into Dloc, then fix up in-place
    out = Dloc[:, :blksize, :b]
    torch.bmm(B, A.transpose(1, 2), out=out)       # out[n][jm][im] = B[n][jm] · A[n][im]
    out.mul_(-2.0)
    out.add_(Xn_A.unsqueeze(1))                    # + ||xi||^2, broadcast over jm
    out.add_(Xn_B.unsqueeze(2))                    # + ||xj||^2, broadcast over im


def run_knn_pipeline(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx):
    l = math.ceil(m / b)

    pk.parallel_for("norms", N * m, compute_norm, X=X, Xn=Xn, d=d, m=m)
    pk.fence()

    # diagonal blocks: distance + topk
    for blk in range(l):
        blksize = min((blk + 1) * b, m) - blk * b
        pk.parallel_for("Dblk_dist", pk.TeamPolicy(N * blksize, pk.AUTO),
                        compute_dist_dblk, X=X, Xn=Xn, Dloc=Dloc, d=d, b=b,
                        blknum=blk, blksize=blksize)

    pk.fence()

    pk.parallel_for("Dblk_topk", N * m, topk_row_dblk,
                    Dloc=Dloc, Lidx=Lidx, Ldst=Ldst, m=m, k=k, b=b)
    pk.fence()

    pk.parallel_for("merge_diag", N * m, merge_topk,
                    Gdst=Gdst, Gidx=Gidx, Ldst=Ldst, Lidx=Lidx, k=k, offset=0, count=m)
    pk.fence()

    pk.parallel_for("flush_local", N * m * (k + 1), flush_local, Ldst=Ldst, Lidx=Lidx, k=k, m=m)
    pk.parallel_for("flush_dloc",  N * m * b,        flush_dloc,  Dloc=Dloc, b=b, m=m)
    pk.fence()

    # off-diagonal (hblk) loop — distances via batched GEMM
    for hblk in range(1, l):
        blksize = m - b * hblk

        compute_dist_hblk_gemm(X, Xn, Dloc, b, hblk, blksize)

        pk.parallel_for("Hblk_row_topk", N * b, topk_row_hblk,
                        Dloc=Dloc, Lidx=Lidx, Ldst=Ldst, k=k, b=b,
                        blksize=blksize, blknum=hblk)
        pk.parallel_for("Hblk_col_topk", N * blksize, topk_col_hblk,
                        Dloc=Dloc, Lidx=Lidx, Ldst=Ldst, k=k, b=b,
                        blknum=hblk, blksize=blksize)
        pk.fence()

        merge_count = m - b * (hblk - 1)
        pk.parallel_for("merge_hblk", N * merge_count, merge_topk,
                        Gdst=Gdst, Gidx=Gidx, Ldst=Ldst, Lidx=Lidx,
                        k=k, offset=b * (hblk - 1), count=merge_count)
        pk.fence()

        pk.parallel_for("flush_local", N * m * (k + 1), flush_local, Ldst=Ldst, Lidx=Lidx, k=k, m=m)
        pk.parallel_for("flush_dloc",  N * m * b,        flush_dloc,  Dloc=Dloc, b=b, m=m)
        pk.fence()


# -----------------------------
# run
# -----------------------------
if __name__ == '__main__':
    t0 = time.time()
    run_knn_pipeline(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx)
    t1 = time.time()
    print("\nExecution time:", (t1 - t0) * 1000, "ms")
