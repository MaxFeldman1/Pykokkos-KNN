import pykokkos as pk
import torch

# k = b kernel: O(k) worst-find scan removed from hblk_col_body.
# Valid only when k == b. Ldst is flushed to INF before each hblk iteration
# and exactly b = k distances are inserted, so im_h is the direct slot index.
@pk.workunit(scratch=[
    (pk.float64, lambda p: 2 * p.k),
    (pk.int32,   lambda p: 2 * p.k),
])
def knn_pipeline_kernel_keqb(team_member: pk.TeamMember,
                              X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx,
                              m, d, k, b):
    INF: pk.float64 = 1.7976931348623157e+308
    n: pk.int32 = team_member.league_rank()
    n2k: pk.int32 = 2 * k

    Sbuf_d: pk.ScratchView1D[pk.float64] = pk.ScratchView1D(team_member.team_scratch(0), n2k)
    Sbuf_i: pk.ScratchView1D[pk.int32]   = pk.ScratchView1D(team_member.team_scratch(0), n2k)

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

    # ---- Phase 3: topk within diagonal blocks ----
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
            for t in range(1, k):
                prop = Ldst[n][i][t] > Ldst[n][i][worst]
                worst = t * prop + worst * (1 - prop)
            prop = not_self * (val < Ldst[n][i][worst])
            Ldst[n][i][worst] = val * prop + Ldst[n][i][worst] * (1 - prop)
            Lidx[n][i][worst] = j * prop + Lidx[n][i][worst] * (1 - prop)

    pk.parallel_for(pk.TeamThreadRange(team_member, m), topk_dblk_body)
    team_member.team_barrier()

    # ---- Phase 4: bitonic merge — diagonal ----
    row_d: pk.int32 = 0
    for row_d in range(m):
        def load_diag(p: int):
            Sbuf_d[p]     = Gdst[n][row_d][p]
            Sbuf_i[p]     = Gidx[n][row_d][p]
            Sbuf_d[p + k] = Ldst[n][row_d][p]
            Sbuf_i[p + k] = Lidx[n][row_d][p]
        pk.parallel_for(pk.TeamThreadRange(team_member, k), load_diag)
        team_member.team_barrier()

        g_d: pk.int32 = 2
        while g_d <= n2k:
            h_d: pk.int32 = g_d >> 1
            while h_d >= 1:
                def sort_diag(j_s: int):
                    ixj_d: pk.int32    = j_s ^ h_d
                    do_cmp_d: pk.int32 = ixj_d > j_s
                    asc_d: pk.int32    = (j_s & g_d) == 0
                    d_j_d:   pk.float64 = Sbuf_d[j_s]
                    d_ixj_d: pk.float64 = Sbuf_d[ixj_d]
                    ns_d: pk.int32 = do_cmp_d * (asc_d * (d_j_d > d_ixj_d) + (1 - asc_d) * (d_j_d < d_ixj_d))
                    tmp_d_d: pk.float64 = d_j_d
                    tmp_i_d: pk.int32   = Sbuf_i[j_s]
                    Sbuf_d[j_s]   = d_j_d   * (1 - ns_d) + d_ixj_d           * ns_d
                    Sbuf_i[j_s]   = tmp_i_d * (1 - ns_d) + Sbuf_i[ixj_d]  * ns_d
                    Sbuf_d[ixj_d] = d_ixj_d * (1 - ns_d) + tmp_d_d           * ns_d
                    Sbuf_i[ixj_d] = Sbuf_i[ixj_d] * (1 - ns_d) + tmp_i_d * ns_d
                pk.parallel_for(pk.TeamThreadRange(team_member, n2k), sort_diag)
                team_member.team_barrier()
                h_d = h_d >> 1
            g_d = g_d * 2

        def store_diag(p: int):
            Gdst[n][row_d][p] = Sbuf_d[p]
            Gidx[n][row_d][p] = Sbuf_i[p]
        pk.parallel_for(pk.TeamThreadRange(team_member, k), store_diag)
        team_member.team_barrier()

    # ---- flush ----
    def flush_local(lin: int):
        row: pk.int32 = lin // k
        col: pk.int32 = lin % k
        Ldst[n][row][col] = INF
        Lidx[n][row][col] = -1

    def flush_dloc(lin: int):
        row: pk.int32 = lin // b
        col: pk.int32 = lin % b
        Dloc[n][row][col] = -1.0

    pk.parallel_for(pk.TeamThreadRange(team_member, m * k), flush_local)
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
                Ldst[n][j][im_h] = val
                Lidx[n][j][im_h] = i_h

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
                for t_r in range(1, k):
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
                Sbuf_d[p]     = Gdst[n][i_mh][p]
                Sbuf_i[p]     = Gidx[n][i_mh][p]
                Sbuf_d[p + k] = Ldst[n][i_mh][p]
                Sbuf_i[p + k] = Lidx[n][i_mh][p]
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
                        d_j_h:   pk.float64 = Sbuf_d[j_s]
                        d_ixj_h: pk.float64 = Sbuf_d[ixj_h]
                        ns_h: pk.int32 = do_cmp_h * (asc_h * (d_j_h > d_ixj_h) + (1 - asc_h) * (d_j_h < d_ixj_h))
                        tmp_d_h: pk.float64 = d_j_h
                        tmp_i_h: pk.int32   = Sbuf_i[j_s]
                        Sbuf_d[j_s]   = d_j_h   * (1 - ns_h) + d_ixj_h           * ns_h
                        Sbuf_i[j_s]   = tmp_i_h * (1 - ns_h) + Sbuf_i[ixj_h]  * ns_h
                        Sbuf_d[ixj_h] = d_ixj_h * (1 - ns_h) + tmp_d_h           * ns_h
                        Sbuf_i[ixj_h] = Sbuf_i[ixj_h] * (1 - ns_h) + tmp_i_h * ns_h
                    pk.parallel_for(pk.TeamThreadRange(team_member, n2k), sort_hblk)
                    team_member.team_barrier()
                    h_h = h_h >> 1
                g_h = g_h * 2

            def store_hblk(p: int):
                Gdst[n][i_mh][p] = Sbuf_d[p]
                Gidx[n][i_mh][p] = Sbuf_i[p]
            pk.parallel_for(pk.TeamThreadRange(team_member, k), store_hblk)
            team_member.team_barrier()

        def flush_local_h(lin: int):
            row: pk.int32 = lin // k
            col: pk.int32 = lin % k
            Ldst[n][row][col] = INF
            Lidx[n][row][col] = -1

        def flush_dloc_h(lin: int):
            row: pk.int32 = lin // b
            col: pk.int32 = lin % b
            Dloc[n][row][col] = -1.0

        pk.parallel_for(pk.TeamThreadRange(team_member, m * k), flush_local_h)
        pk.parallel_for(pk.TeamThreadRange(team_member, m * b), flush_dloc_h)
        team_member.team_barrier()


def run_knn_pipeline_keqb(N, m, d, k, b, X, Xn, Dloc, Gdst, Gidx, Ldst, Lidx):
    policy = pk.TeamPolicy(N, pk.AUTO)
    pk.parallel_for(
        "MAIN_PIPELINE_KEQB",
        policy,
        knn_pipeline_kernel_keqb,
        X=X, Xn=Xn, Dloc=Dloc, Gdst=Gdst, Gidx=Gidx,
        Ldst=Ldst, Lidx=Lidx,
        m=m, d=d, k=k, b=b
    )
    pk.fence()
