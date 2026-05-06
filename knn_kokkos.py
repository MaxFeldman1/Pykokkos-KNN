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
    m = 2000
    d = 70
    k = 2   # must be a power of 2
    b = 32

    np.random.seed(0)
    X_np = np.random.randint(0, 8, size=(N, m, d)).astype(np.float64)

    X    = torch.from_numpy(X_np.copy())
    Xn   = torch.empty((N, m), dtype=torch.float64)
    Dloc = torch.zeros((N, m, b), dtype=torch.float64)

    Gidx = torch.full((N, m, k), -1,                             dtype=torch.int32)
    Gdst = torch.full((N, m, k), torch.finfo(torch.float64).max, dtype=torch.float64)


# -----------------------------
# fused pipeline kernel
# -----------------------------
@pk.workunit
def knn_pipeline_kernel(team_member: pk.TeamMember,
                        X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx, Sbuf_d, Sbuf_i,
                        m, d, k, b, lk):
    INF: pk.float64 = 1.7976931348623157e+308
    n: pk.int32 = team_member.league_rank()
    n2k: pk.int32 = 2 * k

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

    # ---- Phase 2: diagonal block distances ----
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

    # ---- Phase 3: topk within diagonal blocks into Ldst[0..lk-1] ----
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
            not_self: pk.int32 = j != i
            worst: pk.int32 = 0
            t: pk.int32 = 0
            prop: pk.int32 = 0
            for t in range(1, lk):
                prop = Ldst[n][i][t] > Ldst[n][i][worst]
                worst = t * prop + worst * (1 - prop)
            prop = not_self * (val < Ldst[n][i][worst])
            Ldst[n][i][worst] = val * prop + Ldst[n][i][worst] * (1 - prop)
            Lidx[n][i][worst] = j * prop + Lidx[n][i][worst] * (1 - prop)

    pk.parallel_for(pk.TeamThreadRange(team_member, m), topk_dblk_body)
    team_member.team_barrier()

    # ---- Phase 4: collaborative bitonic merge — diagonal ----
    # Rows processed serially; each sort stage is parallel across team threads.
    # Sbuf is (N, 2k): one shared scratch slot per dataset, reused each row.
    row_d: pk.int32 = 0
    for row_d in range(m):
        def load_diag(p: int):
            Sbuf_d[n][p]      = Gdst[n][row_d][p]
            Sbuf_i[n][p]      = Gidx[n][row_d][p]
            in_range: pk.int32 = p < lk
            safe_p: pk.int32   = p * in_range
            Sbuf_d[n][p + k]  = in_range * Ldst[n][row_d][safe_p] + (1 - in_range) * INF
            Sbuf_i[n][p + k]  = in_range * Lidx[n][row_d][safe_p] + (1 - in_range) * (-1)
        pk.parallel_for(pk.TeamThreadRange(team_member, k), load_diag)
        team_member.team_barrier()

        g_d: pk.int32 = 2
        while g_d <= n2k:
            h_d: pk.int32 = g_d >> 1
            while h_d >= 1:
                def sort_diag(j_s: int):
                    ixj_d: pk.int32   = j_s ^ h_d
                    do_cmp_d: pk.int32 = ixj_d > j_s
                    asc_d: pk.int32   = (j_s & g_d) == 0
                    d_j_d:   pk.float64 = Sbuf_d[n][j_s]
                    d_ixj_d: pk.float64 = Sbuf_d[n][ixj_d]
                    ns_d: pk.int32 = do_cmp_d * (asc_d * (d_j_d > d_ixj_d) + (1 - asc_d) * (d_j_d < d_ixj_d))
                    tmp_d_d: pk.float64 = d_j_d
                    tmp_i_d: pk.int32   = Sbuf_i[n][j_s]
                    Sbuf_d[n][j_s]    = d_j_d   * (1 - ns_d) + d_ixj_d            * ns_d
                    Sbuf_i[n][j_s]    = tmp_i_d * (1 - ns_d) + Sbuf_i[n][ixj_d]   * ns_d
                    Sbuf_d[n][ixj_d]  = d_ixj_d * (1 - ns_d) + tmp_d_d            * ns_d
                    Sbuf_i[n][ixj_d]  = Sbuf_i[n][ixj_d] * (1 - ns_d) + tmp_i_d  * ns_d
                pk.parallel_for(pk.TeamThreadRange(team_member, n2k), sort_diag)
                team_member.team_barrier()
                h_d = h_d >> 1
            g_d = g_d * 2

        def store_diag(p: int):
            Gdst[n][row_d][p] = Sbuf_d[n][p]
            Gidx[n][row_d][p] = Sbuf_i[n][p]
        pk.parallel_for(pk.TeamThreadRange(team_member, k), store_diag)
        team_member.team_barrier()

    # ---- flush Ldst / Lidx (lk slots) and Dloc ----
    def flush_local(lin: int):
        row: pk.int32 = lin // lk
        col: pk.int32 = lin % lk
        Ldst[n][row][col] = INF
        Lidx[n][row][col] = -1

    def flush_dloc(lin: int):
        row: pk.int32 = lin // b
        col: pk.int32 = lin % b
        Dloc[n][row][col] = -1.0

    pk.parallel_for(pk.TeamThreadRange(team_member, m * lk), flush_local)
    pk.parallel_for(pk.TeamThreadRange(team_member, m * b), flush_dloc)
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
                for t2 in range(1, lk):
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
                for t_r in range(1, lk):
                    prop_r = Ldst[n][i_r][t_r] > Ldst[n][i_r][worst_r]
                    worst_r = t_r * prop_r + worst_r * (1 - prop_r)
                prop_r = val_r < Ldst[n][i_r][worst_r]
                Ldst[n][i_r][worst_r] = val_r * prop_r + Ldst[n][i_r][worst_r] * (1 - prop_r)
                Lidx[n][i_r][worst_r] = j_r * prop_r + Lidx[n][i_r][worst_r] * (1 - prop_r)

        pk.parallel_for(pk.TeamThreadRange(team_member, b), topk_row_body)
        team_member.team_barrier()

        merge_count: pk.int32 = m - b * (hblk_i - 1)
        merge_off: pk.int32 = b * (hblk_i - 1)

        row_h: pk.int32 = 0
        for row_h in range(merge_count):
            i_mh: pk.int32 = row_h + merge_off
            def load_hblk(p: int):
                Sbuf_d[n][p]       = Gdst[n][i_mh][p]
                Sbuf_i[n][p]       = Gidx[n][i_mh][p]
                in_range_h: pk.int32 = p < lk
                safe_ph: pk.int32    = p * in_range_h
                Sbuf_d[n][p + k]   = in_range_h * Ldst[n][i_mh][safe_ph] + (1 - in_range_h) * INF
                Sbuf_i[n][p + k]   = in_range_h * Lidx[n][i_mh][safe_ph] + (1 - in_range_h) * (-1)
            pk.parallel_for(pk.TeamThreadRange(team_member, k), load_hblk)
            team_member.team_barrier()

            g_h: pk.int32 = 2
            while g_h <= n2k:
                h_h: pk.int32 = g_h >> 1
                while h_h >= 1:
                    def sort_hblk(j_s: int):
                        ixj_h: pk.int32    = j_s ^ h_h
                        do_cmp_h: pk.int32 = ixj_h > j_s
                        asc_h: pk.int32    = (j_s & g_h) == 0
                        d_j_h:   pk.float64 = Sbuf_d[n][j_s]
                        d_ixj_h: pk.float64 = Sbuf_d[n][ixj_h]
                        ns_h: pk.int32 = do_cmp_h * (asc_h * (d_j_h > d_ixj_h) + (1 - asc_h) * (d_j_h < d_ixj_h))
                        tmp_d_h: pk.float64 = d_j_h
                        tmp_i_h: pk.int32   = Sbuf_i[n][j_s]
                        Sbuf_d[n][j_s]   = d_j_h   * (1 - ns_h) + d_ixj_h           * ns_h
                        Sbuf_i[n][j_s]   = tmp_i_h * (1 - ns_h) + Sbuf_i[n][ixj_h]  * ns_h
                        Sbuf_d[n][ixj_h] = d_ixj_h * (1 - ns_h) + tmp_d_h           * ns_h
                        Sbuf_i[n][ixj_h] = Sbuf_i[n][ixj_h] * (1 - ns_h) + tmp_i_h * ns_h
                    pk.parallel_for(pk.TeamThreadRange(team_member, n2k), sort_hblk)
                    team_member.team_barrier()
                    h_h = h_h >> 1
                g_h = g_h * 2

            def store_hblk(p: int):
                Gdst[n][i_mh][p] = Sbuf_d[n][p]
                Gidx[n][i_mh][p] = Sbuf_i[n][p]
            pk.parallel_for(pk.TeamThreadRange(team_member, k), store_hblk)
            team_member.team_barrier()

        def flush_local_h(lin: int):
            row: pk.int32 = lin // lk
            col: pk.int32 = lin % lk
            Ldst[n][row][col] = INF
            Lidx[n][row][col] = -1

        def flush_dloc_h(lin: int):
            row: pk.int32 = lin // b
            col: pk.int32 = lin % b
            Dloc[n][row][col] = -1.0

        pk.parallel_for(pk.TeamThreadRange(team_member, m * lk), flush_local_h)
        pk.parallel_for(pk.TeamThreadRange(team_member, m * b), flush_dloc_h)
        team_member.team_barrier()


def run_knn_pipeline(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx):
    """
    Gdst, Gidx : (N, m, k) — k-best neighbors per point (ascending after merge).
    k must be a power of 2.
    """
    INF = torch.finfo(torch.float64).max
    lk     = min(k, b)
    Ldst   = torch.full((N, m, lk), INF, dtype=torch.float64)
    Lidx   = torch.full((N, m, lk), -1,  dtype=torch.int32)
    Sbuf_d = torch.empty((N, 2 * k),     dtype=torch.float64)
    Sbuf_i = torch.empty((N, 2 * k),     dtype=torch.int32)
    pk.parallel_for(
        "MAIN_PIPELINE",
        pk.TeamPolicy(N, pk.AUTO),
        knn_pipeline_kernel,
        X=X, Xn=Xn, Dloc=Dloc, Gdst=Gdst, Gidx=Gidx,
        Ldst=Ldst, Lidx=Lidx, Sbuf_d=Sbuf_d, Sbuf_i=Sbuf_i,
        m=m, d=d, k=k, b=b, lk=lk
    )
    pk.fence()


# -----------------------------
# run
# -----------------------------
if __name__ == '__main__':
    if device == "cuda":
        import cupy as cp
        X    = cp.asarray(X)
        Xn   = cp.asarray(Xn)
        Dloc = cp.asarray(Dloc)
        Gdst = cp.asarray(Gdst)
        Gidx = cp.asarray(Gidx)

    t0 = time.time()
    run_knn_pipeline(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx)
    t1 = time.time()

    print("\nExecution time:", (t1 - t0) * 1000, "ms")
