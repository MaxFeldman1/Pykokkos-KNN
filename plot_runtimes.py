import re
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# helpers
# -----------------------------
DEFAULT_PARAMS = {'m': 12000, 'd': 70, 'k': 2, 'b': 32}

def parse_all_file(filename):
    """Parse a bench.py 'all' (or single-pipeline) output file.

    Returns:
        params        — dict of fixed parameters from the header
        sweep_var     — 'N' or 'd'
        pipeline_order — list of pipeline names in file order
        pipeline_data  — {pipeline_name: {sweep_val: [runtimes]}}
    """
    params = {}
    pipeline_data = {}
    pipeline_order = []
    current_pipeline = None
    current_val = None
    sweep_var = None

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            pm = re.match(r'^pipeline=(\S+)$', line)
            if pm:
                current_pipeline = pm.group(1)
                if current_pipeline not in pipeline_data:
                    pipeline_data[current_pipeline] = {}
                    pipeline_order.append(current_pipeline)
                current_val = None
                continue

            m = re.match(r'^([a-zA-Z]+)=(\d+)$', line)
            if m:
                key, val = m.group(1), int(m.group(2))
                if current_pipeline is None:
                    params[key] = val
                else:
                    sweep_var = key
                    current_val = val
                    pipeline_data[current_pipeline].setdefault(val, [])
                continue

            try:
                runtime = float(line)
                if current_pipeline is not None and current_val is not None:
                    pipeline_data[current_pipeline][current_val].append(runtime)
            except ValueError:
                pass

    return params, sweep_var, pipeline_order, pipeline_data


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

    L2_MISS_RATE = 1

    l = int(np.ceil(m_val / b_val))
    hblk_pairs = sum(b_val * (m_val - b_val * h) for h in range(1, l))

    total_bytes_per_ds = (
        hblk_pairs * 2 * d_val * 8
        + hblk_pairs * 2 * 8
        + hblk_pairs * 8
        + m_val * d_val * 8
        + m_val * (k_val + 1) * 8 * 4
    )

    T_mem_ms = N * total_bytes_per_ds * L2_MISS_RATE / HBM_BW * 1e3

    return T_compute_ms + T_mem_ms


def non_hblk_napkin_time_ms(N, m_val, d_val, k_val, b_val,
                             TFLOPS_TOTAL=33.5e12, HBM_BW=4e12):
    L2_MISS_RATE = 0.1
    pairs_per_ds = (m_val - 1) * m_val / 2
    flops_norms  = m_val * (2 * d_val)
    flops_topk   = pairs_per_ds * (k_val + 1) * 2
    FLOPS_NON_HBLK = flops_norms + flops_topk

    l = int(np.ceil(m_val / b_val))
    hblk_pairs = sum(b_val * (m_val - b_val * h) for h in range(1, l))

    non_hblk_bytes = (
        hblk_pairs * 2 * 8
        + hblk_pairs * 8
        + m_val * d_val * 8
        + m_val * (k_val + 1) * 8 * 4
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
# pipeline styles
# -----------------------------
PIPELINE_STYLE = {
    'knn_kokkos':         ('steelblue',      'o'),
    'knn_kokkos_keqb':    ('tomato',         's'),
    'unfused_knn_kokkos': ('darkorange',     '^'),
    'gemm_knn_kokkos':    ('mediumseagreen', 'D'),
    'cpp':                ('mediumpurple',   'x'),
}
FALLBACK_COLORS  = ['gray', 'olive', 'deeppink', 'cyan']
FALLBACK_MARKERS = ['v', '<', '>', 'p']


# -----------------------------
# load
# -----------------------------
filename = "runtimes.txt"
params, sweep_var, pipeline_names, pipeline_data = parse_all_file(filename)

m_val = params.get('m', DEFAULT_PARAMS['m'])
d_val = params.get('d', DEFAULT_PARAMS['d'])
k_val = params.get('k', DEFAULT_PARAMS['k'])
b_val = params.get('b', DEFAULT_PARAMS['b'])
N_val = params.get('N', None)   # set for d-sweep (fixed N in header)

if sweep_var is None:
    raise ValueError(f"No sweep variable detected in {filename}")

print(f"Sweep variable: {sweep_var}")
print(f"Pipelines found: {pipeline_names}")
print(f"Params: {params}")

# -----------------------------
# napkin math (N-sweep only)
# -----------------------------
napkin_artists = []
if sweep_var == 'N':
    all_xs = sorted({x for name in pipeline_names for x in pipeline_data[name]})
    N_model = np.array(sorted(set(
        all_xs + list(range(1, max(all_xs) + 1, max(1, max(all_xs) // 200)))
    )))
    t_no_mem   = np.array([mem_napkin_time_ms(n, m_val, d_val, k_val, b_val, INCL_MEM=False) for n in N_model])
    t_with_mem = np.array([mem_napkin_time_ms(n, m_val, d_val, k_val, b_val, INCL_MEM=True)  for n in N_model])
    t_non_hblk = np.array([non_hblk_napkin_time_ms(n, m_val, d_val, k_val, b_val)            for n in N_model])

# -----------------------------
# plot
# -----------------------------
plt.rcParams.update({
    'font.size':        13,
    'axes.titlesize':   14,
    'axes.labelsize':   13,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'lines.linewidth':  2.0,
    'lines.markersize': 7,
})

fig, ax = plt.subplots(figsize=(15, 8))

for i, name in enumerate(pipeline_names):
    data = pipeline_data[name]
    xs    = sorted(data.keys())
    means = [np.mean(data[x]) for x in xs]
    stds  = [np.std(data[x])  for x in xs]
    color, marker = PIPELINE_STYLE.get(
        name,
        (FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
         FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)])
    )
    ax.errorbar(xs, means, yerr=stds if any(s > 0 for s in stds) else None,
                fmt=f'{marker}-', color=color, capsize=5, label=name)

if sweep_var == 'N':
    ax.plot(N_model, t_no_mem,   '--', color='gray', linewidth=1.5, label='Napkin compute-only')
    ax.plot(N_model, t_with_mem, ':',  color='gray', linewidth=2.0, label='Napkin compute+mem')
    ax.plot(N_model, t_non_hblk, '-.', color='gray', linewidth=1.5, label='Napkin non-hblk')

    N_SMs = 132
    ax.axvline(N_SMs, color='silver', linestyle=':', linewidth=1)
    _, y_top = ax.get_ylim()
    ax.text(N_SMs + 5, y_top * 0.05 if y_top > 0 else 1,
            f'N={N_SMs}', color='gray', fontsize=12)

# axis labels and title
if sweep_var == 'N':
    fixed_str = f'm={m_val}  d={d_val}  k={k_val}  b={b_val}'
    ax.set_xlabel('N (number of datasets / league size)')
    ax.set_title(f'KNN pipeline wall time vs N  |  {fixed_str}  |  GH200')
else:
    n_fixed = N_val if N_val is not None else '?'
    fixed_str = f'N={n_fixed}  m={m_val}  k={k_val}  b={b_val}'
    ax.set_xlabel('d (feature dimension)')
    ax.set_title(f'KNN pipeline wall time vs d  |  {fixed_str}  |  GH200')

ax.set_ylabel('Wall time (ms)')
ax.set_yscale('log')
ax.legend(framealpha=0.9, edgecolor='gray')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
out_png = filename.replace('.txt', '.png')
plt.savefig(out_png, dpi=150)
plt.show()
print(f"Saved {out_png}")

# -----------------------------
# throughput summary
# -----------------------------
for name in pipeline_names:
    data = pipeline_data[name]
    print(f"\n--- {name}  (m={m_val}, d={d_val}, k={k_val}, b={b_val}) ---")
    if sweep_var == 'N':
        print(f"{'N':>6}  {'mean ms':>10}  {'datasets/s':>12}  {'napkin ms':>10}")
        print("-" * 46)
        for x in sorted(data.keys()):
            mean_ms = np.mean(data[x])
            dps     = x / (mean_ms / 1e3)
            model   = mem_napkin_time_ms(x, m_val, d_val, k_val, b_val)
            print(f"{x:>6}  {mean_ms:>10.1f}  {dps:>12.1f}  {model:>10.1f}")
    else:
        n_fixed = N_val if N_val is not None else 1
        print(f"{'d':>6}  {'mean ms':>10}  {'datasets/s':>12}")
        print("-" * 34)
        for x in sorted(data.keys()):
            mean_ms = np.mean(data[x])
            dps     = n_fixed / (mean_ms / 1e3)
            print(f"{x:>6}  {mean_ms:>10.1f}  {dps:>12.1f}")
