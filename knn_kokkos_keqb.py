import pykokkos as pk
import torch

# k = b kernel: Ldst/Lidx eliminated — Gdst/Gidx sized 2k.
# Upper half Gdst[k..2k-1] / Gidx[k..2k-1] serves as the local candidate buffer.
# After each bitonic merge, store writes winners to [0..k-1] and resets [k..2k-1]
# to INF/-1 in the same pass, eliminating all separate flush kernels.
# Valid only when k == b.
@pk.workunit(scratch=[
    (pk.float64, lambda p: 2 * p.k),
    (pk.int32,   lambda p: 2 * p.k),
])
def knn_pipeline_kernel_keqb(team_member: pk.TeamMember,
                              X, Xn, Dloc, Iloc, Gdst, Gidx,
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
    # k=b: each point has at most b-1 = k-1 candidates, all fit in upper k slots.
    # Direct slot: jm < im -> slot=jm, jm > im -> slot=jm-1 (skip self branchlessly).
    def topk_dblk_body(i: int):
        im: pk.int32 = i % b
        id_: pk.int32 = i - im
        m_top: pk.int32 = m < id_ + b
        top_range: pk.int32 = m * m_top + (id_ + b) * (1 - m_top)
        j: pk.int32 = 0
        for j in range(id_, top_range):
            jm: pk.int32 = j % b
            not_self: pk.int32 = j != i
            i_first: pk.int32 = im <= jm
            idx0: pk.int32 = i * i_first + j * (1 - i_first)
            idx1: pk.int32 = jm * i_first + im * (1 - i_first)
            val: pk.float64 = Dloc[n][idx0][idx1]
            slot: pk.int32 = jm - (jm > im)
            Gdst[n][i][k + slot] = val * not_self + Gdst[n][i][k + slot] * (1 - not_self)
            Gidx[n][i][k + slot] = j   * not_self + Gidx[n][i][k + slot] * (1 - not_self)

    pk.parallel_for(pk.TeamThreadRange(team_member, m), topk_dblk_body)
    team_member.team_barrier()

    # ---- Phase 4: bitonic merge — diagonal ----
    row_d: pk.int32 = 0
    for row_d in range(m):
        def load_diag(p: int):
            Sbuf_d[p]     = Gdst[n][row_d][p]
            Sbuf_i[p]     = Gidx[n][row_d][p]
            Sbuf_d[p + k] = Gdst[n][row_d][p + k]
            Sbuf_i[p + k] = Gidx[n][row_d][p + k]
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
                    Sbuf_i[j_s]   = tmp_i_d * (1 - ns_d) + Sbuf_i[ixj_d]     * ns_d
                    Sbuf_d[ixj_d] = d_ixj_d * (1 - ns_d) + tmp_d_d            * ns_d
                    Sbuf_i[ixj_d] = Sbuf_i[ixj_d] * (1 - ns_d) + tmp_i_d     * ns_d
                pk.parallel_for(pk.TeamThreadRange(team_member, n2k), sort_diag)
                team_member.team_barrier()
                h_d = h_d >> 1
            g_d = g_d * 2

        # Write winners to lower half; reset upper half to INF/-1 in same pass.
        def store_diag(p: int):
            Gdst[n][row_d][p]     = Sbuf_d[p]
            Gidx[n][row_d][p]     = Sbuf_i[p]
            Gdst[n][row_d][p + k] = INF
            Gidx[n][row_d][p + k] = -1
        pk.parallel_for(pk.TeamThreadRange(team_member, k), store_diag)
        team_member.team_barrier()

    # ---- Phase 5-7: off-diagonal (hblk) loop — batch-bitonic ----
    # No per-candidate insertion (no scan, no heap). Mirrors the C++ approach:
    # candidates are loaded wholesale in chunks of b and merged via bitonic sort.
    hblk_i: pk.int32 = 0
    for hblk_i in range(1, l):
        blksize_h: pk.int32 = m - b * hblk_i
        i_off_h: pk.int32 = b * (hblk_i - 1)

        # Compute distances for current strip; store in Dloc + fill j-side upper half.
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
                Iloc[n][jm][im_h] = j
                Gdst[n][j][k + im_h] = val
                Gidx[n][j][k + im_h] = i_h

        pk.parallel_for(pk.TeamThreadRange(team_member, blksize_h), hblk_col_body)
        team_member.team_barrier()

        # I-side: sub-chunks of b candidates — fill upper half then bitonic merge.
        n_sub_h: pk.int32 = (blksize_h + b - 1) // b
        sub_idx_h: pk.int32 = 0
        for sub_idx_h in range(n_sub_h):
            sub_off_h: pk.int32 = sub_idx_h * b
            sub_end_h: pk.int32 = sub_off_h + b
            m_sub_h: pk.int32   = sub_end_h > blksize_h
            sub_sz_h: pk.int32  = b - m_sub_h * (sub_end_h - blksize_h)

            def fill_irows(im_r: int):
                i_r_f: pk.int32 = im_r + i_off_h
                slot_f: pk.int32 = 0
                for slot_f in range(sub_sz_h):
                    Gdst[n][i_r_f][k + slot_f] = Dloc[n][sub_off_h + slot_f][im_r]
                    Gidx[n][i_r_f][k + slot_f] = Iloc[n][sub_off_h + slot_f][im_r]
            pk.parallel_for(pk.TeamThreadRange(team_member, b), fill_irows)
            team_member.team_barrier()

            row_si: pk.int32 = 0
            for row_si in range(b):
                i_si: pk.int32 = row_si + i_off_h
                def load_si(p: int):
                    Sbuf_d[p]     = Gdst[n][i_si][p]
                    Sbuf_i[p]     = Gidx[n][i_si][p]
                    Sbuf_d[p + k] = Gdst[n][i_si][p + k]
                    Sbuf_i[p + k] = Gidx[n][i_si][p + k]
                pk.parallel_for(pk.TeamThreadRange(team_member, k), load_si)
                team_member.team_barrier()

                g_si: pk.int32 = 2
                while g_si <= n2k:
                    h_si: pk.int32 = g_si >> 1
                    while h_si >= 1:
                        def sort_si(j_s: int):
                            ixj_si: pk.int32    = j_s ^ h_si
                            do_cmp_si: pk.int32 = ixj_si > j_s
                            asc_si: pk.int32    = (j_s & g_si) == 0
                            d_j_si:   pk.float64 = Sbuf_d[j_s]
                            d_ixj_si: pk.float64 = Sbuf_d[ixj_si]
                            ns_si: pk.int32 = do_cmp_si * (asc_si * (d_j_si > d_ixj_si) + (1 - asc_si) * (d_j_si < d_ixj_si))
                            tmp_d_si: pk.float64 = d_j_si
                            tmp_i_si: pk.int32   = Sbuf_i[j_s]
                            Sbuf_d[j_s]    = d_j_si   * (1 - ns_si) + d_ixj_si         * ns_si
                            Sbuf_i[j_s]    = tmp_i_si * (1 - ns_si) + Sbuf_i[ixj_si]   * ns_si
                            Sbuf_d[ixj_si] = d_ixj_si * (1 - ns_si) + tmp_d_si          * ns_si
                            Sbuf_i[ixj_si] = Sbuf_i[ixj_si] * (1 - ns_si) + tmp_i_si   * ns_si
                        pk.parallel_for(pk.TeamThreadRange(team_member, n2k), sort_si)
                        team_member.team_barrier()
                        h_si = h_si >> 1
                    g_si = g_si * 2

                def store_si(p: int):
                    Gdst[n][i_si][p]     = Sbuf_d[p]
                    Gidx[n][i_si][p]     = Sbuf_i[p]
                    Gdst[n][i_si][p + k] = INF
                    Gidx[n][i_si][p + k] = -1
                pk.parallel_for(pk.TeamThreadRange(team_member, k), store_si)
                team_member.team_barrier()

        # J-side: one bitonic merge per row (upper half filled by hblk_col_body).
        row_jh: pk.int32 = 0
        for row_jh in range(blksize_h):
            i_jh: pk.int32 = row_jh + b * hblk_i
            def load_jh(p: int):
                Sbuf_d[p]     = Gdst[n][i_jh][p]
                Sbuf_i[p]     = Gidx[n][i_jh][p]
                Sbuf_d[p + k] = Gdst[n][i_jh][p + k]
                Sbuf_i[p + k] = Gidx[n][i_jh][p + k]
            pk.parallel_for(pk.TeamThreadRange(team_member, k), load_jh)
            team_member.team_barrier()

            g_jh: pk.int32 = 2
            while g_jh <= n2k:
                h_jh: pk.int32 = g_jh >> 1
                while h_jh >= 1:
                    def sort_jh(j_s: int):
                        ixj_jh: pk.int32    = j_s ^ h_jh
                        do_cmp_jh: pk.int32 = ixj_jh > j_s
                        asc_jh: pk.int32    = (j_s & g_jh) == 0
                        d_j_jh:   pk.float64 = Sbuf_d[j_s]
                        d_ixj_jh: pk.float64 = Sbuf_d[ixj_jh]
                        ns_jh: pk.int32 = do_cmp_jh * (asc_jh * (d_j_jh > d_ixj_jh) + (1 - asc_jh) * (d_j_jh < d_ixj_jh))
                        tmp_d_jh: pk.float64 = d_j_jh
                        tmp_i_jh: pk.int32   = Sbuf_i[j_s]
                        Sbuf_d[j_s]    = d_j_jh   * (1 - ns_jh) + d_ixj_jh         * ns_jh
                        Sbuf_i[j_s]    = tmp_i_jh * (1 - ns_jh) + Sbuf_i[ixj_jh]   * ns_jh
                        Sbuf_d[ixj_jh] = d_ixj_jh * (1 - ns_jh) + tmp_d_jh          * ns_jh
                        Sbuf_i[ixj_jh] = Sbuf_i[ixj_jh] * (1 - ns_jh) + tmp_i_jh   * ns_jh
                    pk.parallel_for(pk.TeamThreadRange(team_member, n2k), sort_jh)
                    team_member.team_barrier()
                    h_jh = h_jh >> 1
                g_jh = g_jh * 2

            def store_jh(p: int):
                Gdst[n][i_jh][p]     = Sbuf_d[p]
                Gidx[n][i_jh][p]     = Sbuf_i[p]
                Gdst[n][i_jh][p + k] = INF
                Gidx[n][i_jh][p + k] = -1
            pk.parallel_for(pk.TeamThreadRange(team_member, k), store_jh)
            team_member.team_barrier()


def run_knn_pipeline_keqb(N, m, d, k, b, X, Xn, Dloc, Iloc, Gdst, Gidx):
    policy = pk.TeamPolicy(N, pk.AUTO)
    pk.parallel_for(
        "MAIN_PIPELINE_KEQB",
        policy,
        knn_pipeline_kernel_keqb,
        X=X, Xn=Xn, Dloc=Dloc, Iloc=Iloc, Gdst=Gdst, Gidx=Gidx,
        m=m, d=d, k=k, b=b
    )
    pk.fence()
