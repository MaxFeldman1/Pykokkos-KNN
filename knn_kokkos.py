import pykokkos as pk
import torch
import numpy as np
import time
import math

# -----------------------------
# parameters
# -----------------------------
device = "cpu"          # ← change to "cuda" for GPU

if __name__ == '__main__':
    N = 2       # batch size — number of datasets to process simultaneously
    m = 100
    d = 70
    k = 2
    b = 32

    # -----------------------------
    # data  — leading N dimension
    # -----------------------------
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
# fused pipeline kernel
# One team = one dataset. team_barrier() replaces pk.fence() between phases.
# Launch with TeamPolicy(N_datasets, AUTO) to process N datasets in parallel.
# -----------------------------
@pk.workunit
def knn_pipeline_kernel(team_member: pk.TeamMember, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx, m, d, k, b):
    INF: pk.float64 = 1.7976931348623157e+308
    # Each team processes one dataset. n selects the batch slice.
    n: pk.int32 = team_member.league_rank()

    # ---- Phase 1: norms ----
    def norm_body(i: int):
        s: pk.float64 = 0.0
        t: pk.int32 = 0
        for t in range(d):
            s += X[n][i][t] * X[n][i][t]
        Xn[n][i] = s

    pk.parallel_for(pk.TeamThreadRange(team_member, m), norm_body)
    team_member.team_barrier()

    l: pk.int32 = (m + b - 1) // b

    # ---- Phase 2: diagonal blocks — one thread per upper-triangle pair ----
    blknum: pk.int32 = 0
    for blknum in range(l):
        end: pk.int32 = (blknum + 1) * b
        m_flag: pk.int32 = end > m
        blksize: pk.int32 = end - m_flag * (end - m) - blknum * b
        n_pairs: pk.int32 = blksize * (blksize - 1) // 2

        def dblk_body(lin: int):
            jm: pk.int32 = 1
            start: pk.int32 = 0
            while start + jm <= lin:
                start += jm
                jm += 1
            im: pk.int32 = lin - start
            i: pk.int32 = im + b * blknum
            j: pk.int32 = jm + b * blknum
            dot: pk.float64 = 0.0
            t: pk.int32 = 0
            for t in range(d):
                dot += X[n][i][t] * X[n][j][t]
            Dloc[n][i][jm] = -2.0 * dot + Xn[n][i] + Xn[n][j]

        pk.parallel_for(pk.TeamThreadRange(team_member, n_pairs), dblk_body)
        team_member.team_barrier()

    # ---- Phase 3: topk within each diagonal block ----
    def topk_dblk_body(i: int):
        im: pk.int32 = i % b
        id_: pk.int32 = i - im
        m_top: pk.int32 = m < id_ + b
        top_range: pk.int32 = m * m_top + (id_ + b) * (1 - m_top)
        j: pk.int32 = 0
        for j in range(id_, top_range):
            jm: pk.int32 = j % b
            i_first: pk.int32 = im <= jm
            idx0: pk.int32 = i * i_first + j * (1 - i_first)
            idx1: pk.int32 = jm * i_first + im * (1 - i_first)
            val: pk.float64 = Dloc[n][idx0][idx1]
            worst: pk.int32 = 0
            t: pk.int32 = 0
            for t in range(1, k + 1):
                if Ldst[n][i][t] > Ldst[n][i][worst]:
                    worst = t
            if val < Ldst[n][i][worst]:
                Ldst[n][i][worst] = val
                Lidx[n][i][worst] = j

    pk.parallel_for(pk.TeamThreadRange(team_member, m), topk_dblk_body)
    team_member.team_barrier()

    # ---- Phase 4: merge diagonal locals into global ----
    def merge_diag_body(i: int):
        s: pk.int32 = 0
        for s in range(k + 1):
            dst: pk.float64 = Ldst[n][i][s]
            idx: pk.int32 = Lidx[n][i][s]
            worst: pk.int32 = 0
            t: pk.int32 = 0
            for t in range(1, k + 1):
                if Gdst[n][i][t] > Gdst[n][i][worst]:
                    worst = t
            if dst < Gdst[n][i][worst]:
                Gdst[n][i][worst] = dst
                Gidx[n][i][worst] = idx

    pk.parallel_for(pk.TeamThreadRange(team_member, m), merge_diag_body)
    team_member.team_barrier()

    # ---- flush Ldst / Lidx / Dloc ----
    def flush_local(lin: int):
        row: pk.int32 = lin // (k + 1)
        col: pk.int32 = lin % (k + 1)
        Ldst[n][row][col] = INF
        Lidx[n][row][col] = -1

    def flush_dloc(lin: int):
        row: pk.int32 = lin // b
        col: pk.int32 = lin % b
        Dloc[n][row][col] = -1.0

    pk.parallel_for(pk.TeamThreadRange(team_member, m * (k + 1)), flush_local)
    pk.parallel_for(pk.TeamThreadRange(team_member, m * b),       flush_dloc)
    team_member.team_barrier()

    # ---- Phase 5-7: off-diagonal (hblk) loop ----
    hblk_i: pk.int32 = 0
    for hblk_i in range(1, l):
        blksize_h: pk.int32 = m - b * hblk_i

        def hblk_col_body(jm: int):
            j: pk.int32 = jm + b * hblk_i
            im_h: pk.int32 = 0
            for im_h in range(b):
                i_h: pk.int32 = im_h + b * (hblk_i - 1)
                dot: pk.float64 = 0.0
                t: pk.int32 = 0
                for t in range(d):
                    dot += X[n][i_h][t] * X[n][j][t]
                val: pk.float64 = -2.0 * dot + Xn[n][i_h] + Xn[n][j]
                Dloc[n][jm][im_h] = val
                worst: pk.int32 = 0
                t2: pk.int32 = 0
                prop: pk.int32 = 0
                for t2 in range(1, k + 1):
                    prop = Ldst[n][j][t2] > Ldst[n][j][worst]
                    worst = t2 * prop + worst * (1 - prop)
                prop = val < Ldst[n][j][worst]
                Ldst[n][j][worst] = val * prop + Ldst[n][j][worst] * (1 - prop)
                Lidx[n][j][worst] = i_h * prop + Lidx[n][j][worst] * (1 - prop)

        pk.parallel_for(pk.TeamThreadRange(team_member, blksize_h), hblk_col_body)
        team_member.team_barrier()

        def topk_row_body(im_r: int):
            i_r: pk.int32 = im_r + b * (hblk_i - 1)
            jm_r: pk.int32 = 0
            for jm_r in range(blksize_h):
                j_r: pk.int32 = jm_r + b * hblk_i
                val_r: pk.float64 = Dloc[n][jm_r][im_r]
                worst_r: pk.int32 = 0
                t_r: pk.int32 = 0
                prop_r: pk.int32 = 0
                for t_r in range(1, k + 1):
                    prop_r = Ldst[n][i_r][t_r] > Ldst[n][i_r][worst_r]
                    worst_r = t_r * prop_r + worst_r * (1 - prop_r)
                prop_r = val_r < Ldst[n][i_r][worst_r]
                Ldst[n][i_r][worst_r] = val_r * prop_r + Ldst[n][i_r][worst_r] * (1 - prop_r)
                Lidx[n][i_r][worst_r] = j_r * prop_r + Lidx[n][i_r][worst_r] * (1 - prop_r)

        pk.parallel_for(pk.TeamThreadRange(team_member, b), topk_row_body)
        team_member.team_barrier()

        merge_count: pk.int32 = m - b * (hblk_i - 1)
        merge_off: pk.int32 = b * (hblk_i - 1)

        def merge_hblk_body(lin_m: int):
            i_m: pk.int32 = lin_m + merge_off
            s_m: pk.int32 = 0
            for s_m in range(k + 1):
                dst_m: pk.float64 = Ldst[n][i_m][s_m]
                idx_m: pk.int32 = Lidx[n][i_m][s_m]
                worst_m: pk.int32 = 0
                t_m: pk.int32 = 0
                for t_m in range(1, k + 1):
                    if Gdst[n][i_m][t_m] > Gdst[n][i_m][worst_m]:
                        worst_m = t_m
                if dst_m < Gdst[n][i_m][worst_m]:
                    Gdst[n][i_m][worst_m] = dst_m
                    Gidx[n][i_m][worst_m] = idx_m

        pk.parallel_for(pk.TeamThreadRange(team_member, merge_count), merge_hblk_body)
        team_member.team_barrier()

        def flush_local_h(lin: int):
            row: pk.int32 = lin // (k + 1)
            col: pk.int32 = lin % (k + 1)
            Ldst[n][row][col] = INF
            Lidx[n][row][col] = -1

        def flush_dloc_h(lin: int):
            row: pk.int32 = lin // b
            col: pk.int32 = lin % b
            Dloc[n][row][col] = -1.0

        if hblk_i < l - 1:
            pk.parallel_for(pk.TeamThreadRange(team_member, m * (k + 1)), flush_local_h)
            pk.parallel_for(pk.TeamThreadRange(team_member, m * b),       flush_dloc_h)
            team_member.team_barrier()


@pk.workunit
def build_G(i, Gidx, G, k):
    t: pk.int32 = 0
    for t in range(k + 1):
        j: pk.int32 = Gidx[i][t]
        if j >= 0 and j != i:
            G[i][j] = 1


def run_knn_pipeline(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx):
    """
    N       : number of datasets (batch size)
    X       : (N, m, d)
    Xn      : (N, m)
    Dloc    : (N, m, b)
    Gdst    : (N, m, k+1)
    Gidx    : (N, m, k+1)
    Ldst    : (N, m, k+1)
    Lidx    : (N, m, k+1)
    """
    if device == "cuda":
        import cupy as cp
        X    = cp.asarray(X)
        Xn   = cp.asarray(Xn)
        Dloc = cp.asarray(Dloc)
        Gdst = cp.asarray(Gdst)
        Gidx = cp.asarray(Gidx)
        Ldst = cp.asarray(Ldst)
        Lidx = cp.asarray(Lidx)

    # One team per dataset — N teams run in parallel across SMs.
    pk.parallel_for(
        pk.TeamPolicy(N, pk.AUTO),
        knn_pipeline_kernel,
        X=X, Xn=Xn, Dloc=Dloc, Gdst=Gdst, Gidx=Gidx,
        Ldst=Ldst, Lidx=Lidx, m=m, d=d, k=k, b=b
    )
    pk.fence()


# -----------------------------
# run
# -----------------------------
if __name__ == '__main__':
    t0 = time.time()

    run_knn_pipeline(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx)

    t1 = time.time()

    # print("Coordinate matrix")
    # print(X_np)

    print("\nExecution time:", (t1 - t0) * 1000, "ms")
