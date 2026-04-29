import re
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# helpers
# -----------------------------
DEFAULT_PARAMS = {'m': 12000, 'd': 70, 'k': 2, 'b': 32}

def parse_file(filename, default_params=None):
    """Return (params dict, {N: runtime_ms}) from a runtimes text file.
    Files may or may not have a param header; each N block has one runtime."""
    params = dict(default_params or DEFAULT_PARAMS)
    data = {}
    current_N = None
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^([a-zA-Z]+)=(\d+)$', line)
            if m:
                key, val = m.group(1), int(m.group(2))
                if key == 'N':
                    current_N = val
                    data.setdefault(current_N, [])
                else:
                    params[key] = val
            else:
                try:
                    runtime = float(line)
                    if current_N is not None:
                        data[current_N].append(runtime)
                except ValueError:
                    pass
    return params, data


def mem_napkin_time_ms(N, m_val, d_val, k_val, b_val,
                   TFLOPS_TOTAL=33.5e12, HBM_BW=4e12, INCL_MEM=True):
    pairs_per_ds = (m_val - 1) * m_val / 2
    flops_dist  = pairs_per_ds * (2 * d_val)
    flops_norms = m_val * (2 * d_val)
    flops_topk  = pairs_per_ds * (k_val + 1) * 2
    FLOPS_PER_DATASET = flops_dist + flops_norms + flops_topk

    T_compute_ms = N * FLOPS_PER_DATASET / TFLOPS_TOTAL * 1e3

    if not INCL_MEM:
        return T_compute_ms

    L2_MISS_RATE = 0.1

    l = int(np.ceil(m_val / b_val))
    hblk_pairs = sum(b_val * (m_val - b_val * h) for h in range(1, l))

    # Total logical bytes loaded across the full pipeline per dataset (as if no cache).
    # hblk X reads dominate: every (i,j) pair reads Xi and Xj in full.
    total_bytes_per_ds = (
        hblk_pairs * 2 * d_val * 8          # hblk: X[i] + X[j] per pair
        + hblk_pairs * 2 * 8                 # hblk: Xn[i] + Xn[j] per pair
        + hblk_pairs * 8                     # hblk: Dloc write per pair
        + m_val * d_val * 8                  # norm phase: X read once
        + m_val * (k_val + 1) * 8 * 4       # topk/merge: Gdst, Gidx, Ldst, Lidx
    )

    T_mem_ms = N * total_bytes_per_ds * L2_MISS_RATE / HBM_BW * 1e3

    return T_compute_ms + T_mem_ms


def non_hblk_napkin_time_ms(N, m_val, d_val, k_val, b_val,
                             TFLOPS_TOTAL=33.5e12, HBM_BW=4e12):
    """Time for everything except the hblk distance kernel (norms + topk + merge + flush)."""
    L2_MISS_RATE = 0.1
    pairs_per_ds = (m_val - 1) * m_val / 2
    flops_norms  = m_val * (2 * d_val)
    flops_topk   = pairs_per_ds * (k_val + 1) * 2
    FLOPS_NON_HBLK = flops_norms + flops_topk

    l = int(np.ceil(m_val / b_val))
    hblk_pairs = sum(b_val * (m_val - b_val * h) for h in range(1, l))

    # All bytes except the dominant hblk X[i]+X[j] reads
    non_hblk_bytes = (
        hblk_pairs * 2 * 8                # hblk: Xn[i] + Xn[j] per pair
        + hblk_pairs * 8                   # Dloc write (dist kernel) + read (topk kernel)
        + m_val * d_val * 8                # norm phase: X read once
        + m_val * (k_val + 1) * 8 * 4     # topk/merge: Gdst, Gidx, Ldst, Lidx
    )

    T_compute_ms = N * FLOPS_NON_HBLK / TFLOPS_TOTAL * 1e3
    T_mem_ms     = N * non_hblk_bytes * L2_MISS_RATE / HBM_BW * 1e3
    return T_compute_ms + T_mem_ms


def napkin_time_ms(N, m_val, d_val, k_val, b_val,
                   TFLOPS_TOTAL=33.5e12, N_SMs=132):
    FLOPS_PER_SM = TFLOPS_TOTAL / N_SMs
    l = int(np.ceil(m_val / b_val))
    hblk_pairs  = sum(b_val * (m_val - b_val * i) for i in range(1, l))
    flops_dist  = hblk_pairs * (2 * d_val)
    flops_norms = m_val * (2 * d_val)
    flops_topk  = hblk_pairs * 9 * 2
    FLOPS_PER_DATASET = flops_dist + flops_norms + flops_topk
    batches = int(np.ceil(N / N_SMs))
    return batches * FLOPS_PER_DATASET / FLOPS_PER_SM * 1e3


# -----------------------------
# load all three files
# -----------------------------
pipelines = [
    ('fused',   'fused_runtimes.txt',   'steelblue',  'o'),
    ('unfused', 'unfused_runtimes.txt', 'darkorange', 's'),
    ('gemm',    'gemm_runtimes.txt',    'mediumseagreen', '^'),
]

all_data   = {}   # name -> (params, {N: [runtimes]})
for name, fname, _color, _marker in pipelines:
    try:
        params, data = parse_file(fname)
        all_data[name] = (params, data)
    except FileNotFoundError:
        print(f"Warning: {fname} not found — skipping {name}")

# -----------------------------
# napkin math lines (one per unique param set)
# -----------------------------
# Key by (m, d, k, b) so pipelines sharing params share one model line.
seen_params = {}
for name, (params, _) in all_data.items():
    key = (params['m'], params['d'], params['k'], params['b'])
    seen_params.setdefault(key, []).append(name)

all_Ns = sorted({n for _, (_, d) in all_data.items() for n in d})
N_model = np.array(sorted(set(all_Ns + list(range(1, max(all_Ns) + 1, max(1, max(all_Ns) // 200))))))

napkin_lines = {}   # key -> (t_compute_only, t_with_mem, t_non_hblk)
for key in seen_params:
    m_val, d_val, k_val, b_val = key
    t_no_mem   = np.array([mem_napkin_time_ms(n, m_val, d_val, k_val, b_val, INCL_MEM=False) for n in N_model])
    t_with_mem = np.array([mem_napkin_time_ms(n, m_val, d_val, k_val, b_val, INCL_MEM=True)  for n in N_model])
    t_non_hblk = np.array([non_hblk_napkin_time_ms(n, m_val, d_val, k_val, b_val)            for n in N_model])
    napkin_lines[key] = (t_no_mem, t_with_mem, t_non_hblk)

# -----------------------------
# plot
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 6))

for name, fname, color, marker in pipelines:
    if name not in all_data:
        continue
    params, data = all_data[name]
    Ns    = sorted(data.keys())
    means = [np.mean(data[n]) for n in Ns]
    stds  = [np.std(data[n])  for n in Ns]
    ax.errorbar(Ns, means, yerr=stds if any(s > 0 for s in stds) else None,
                fmt=f'{marker}-', color=color, capsize=4,
                label=f'{name}  (m={params["m"]})')

napkin_colors = ['tomato', 'purple', 'gold']
for i, (key, (t_no_mem, t_with_mem, t_non_hblk)) in enumerate(napkin_lines.items()):
    m_val, d_val, k_val, b_val = key
    names = ', '.join(seen_params[key])
    color = napkin_colors[i % len(napkin_colors)]
    ax.plot(N_model, t_no_mem, '--', color=color, linewidth=1.5,
            label=f'Napkin compute-only  m={m_val}  [{names}]')
    ax.plot(N_model, t_with_mem, ':', color=color, linewidth=2.0,
            label=f'Napkin compute+mem   m={m_val}  [{names}]')
    ax.plot(N_model, t_non_hblk, '-.', color=color, linewidth=1.5,
            label=f'Napkin non-hblk      m={m_val}  [{names}]')

N_SMs = 132
ax.axvline(N_SMs, color='gray', linestyle=':', linewidth=1)
ax.text(N_SMs + 5, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 1,
        f'N={N_SMs}', color='gray', fontsize=9)

ax.set_xlabel('N (number of datasets / league size)', fontsize=12)
ax.set_ylabel('Wall time (ms)', fontsize=12)
ax.set_title('KNN pipeline scaling comparison  |  GH200', fontsize=12)
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('runtimes.png', dpi=150)
plt.show()
print("Saved runtimes.png")

# -----------------------------
# throughput summary
# -----------------------------
for name, fname, _color, _marker in pipelines:
    if name not in all_data:
        continue
    params, data = all_data[name]
    m_val, d_val, k_val, b_val = params['m'], params['d'], params['k'], params['b']
    Ns = sorted(data.keys())
    print(f"\n--- {name}  (m={m_val}, d={d_val}, k={k_val}, b={b_val}) ---")
    print(f"{'N':>6}  {'mean ms':>10}  {'datasets/s':>12}  {'napkin ms':>10}")
    print("-" * 46)
    for n in Ns:
        mean_ms = np.mean(data[n])
        dps     = n / (mean_ms / 1e3)
        model   = mem_napkin_time_ms(n, m_val, d_val, k_val, b_val)
        print(f"{n:>6}  {mean_ms:>10.1f}  {dps:>12.1f}  {model:>10.1f}")
