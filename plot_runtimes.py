import re
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# parse runtimes.txt
# -----------------------------
params = {}
data = {}   # N -> list of runtimes (ms)

filename = 'runtimesalt.txt'

with open(filename) as f:
    current_N = None
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

m_val  = params['m']
d_val  = params['d']
k_val  = params['k']
b_val  = params['b']

Ns      = sorted(data.keys())
means   = [np.mean(data[n]) for n in Ns]
stds    = [np.std(data[n])  for n in Ns]

# -----------------------------
# napkin math model
# -----------------------------
# GH200: 33.5 TFLOPS FP64, 132 SMs
TFLOPS_TOTAL = 33.5e12
N_SMs        = 132
FLOPS_PER_SM = TFLOPS_TOTAL / N_SMs   # FLOPS available per SM

# FLOPs per dataset (m=2000, d=70)
# hblk pairs: sum_{i=1}^{l-1} b * (m - b*i)
l = int(np.ceil(m_val / b_val))
hblk_pairs = sum(b_val * (m_val - b_val * i) for i in range(1, l))
flops_dist  = hblk_pairs * (2 * d_val)          # dot product per pair
flops_norms = m_val * (2 * d_val)               # norm phase
flops_topk  = hblk_pairs * 9 * 2                # topk_col + topk_row
FLOPS_PER_DATASET = flops_dist + flops_norms + flops_topk

def napkin_time_ms(N):
    # Each SM handles ceil(N/N_SMs) datasets sequentially
    batches = int(np.ceil(N / N_SMs))
    return batches * FLOPS_PER_DATASET / FLOPS_PER_SM * 1e3   # ms

# -----------------------------
# memory-inclusive model
# -----------------------------
# GH200 HBM3 bandwidth: 4 TB/s
HBM_BW = 4.0e12   # bytes/s

# Bytes loaded per dataset (dominant term is X; other tensors are small)
#   X      : m * d * 8  (read repeatedly, but cold miss on first load)
#   Xn     : m * 8
#   Dloc   : m * b * 8
#   Gdst/Gidx/Ldst/Lidx : m * (k+1) * 8 * 4
BYTES_PER_DATASET = (
    m_val * d_val * 8            # X
    + m_val * 8                  # Xn
    + m_val * b_val * 8          # Dloc
    + m_val * (k_val + 1) * 8 * 4  # Gdst, Gidx, Ldst, Lidx
)

def napkin_time_mem_ms(N):
    # Per batch of up to N_SMs simultaneous teams, all SMs share the HBM bus.
    # Memory cost per batch = N_SMs * BYTES_PER_DATASET / HBM_BW
    # (each of the N_SMs teams loads its dataset; transfers contend for shared bus)
    batches = int(np.ceil(N / N_SMs))
    compute_ms = batches * FLOPS_PER_DATASET / FLOPS_PER_SM * 1e3
    memory_ms  = batches * N_SMs * BYTES_PER_DATASET / HBM_BW * 1e3
    return compute_ms + memory_ms

N_model   = np.array(sorted(set(list(Ns) + list(range(1, max(Ns)+1, max(Ns)//200)))))
t_model   = np.array([napkin_time_ms(n)     for n in N_model])
t_model_m = np.array([napkin_time_mem_ms(n) for n in N_model])

# -----------------------------
# plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(Ns, means, yerr=stds, fmt='o-', color='steelblue',
            capsize=4, label='Measured (mean ± std)')
ax.plot(N_model, t_model, '--', color='tomato', linewidth=1.5,
        label='Napkin model (compute-bound only)')
ax.plot(N_model, t_model_m, '--', color='darkorange', linewidth=1.5,
        label='Napkin model (compute + HBM cold load)')

ax.axvline(N_SMs, color='gray', linestyle=':', linewidth=1)
ax.text(N_SMs + 5, ax.get_ylim()[1] * 0.05,
        f'N={N_SMs} (all SMs busy)', color='gray', fontsize=9)

ax.set_xlabel('N (number of datasets / league size)', fontsize=12)
ax.set_ylabel('Wall time (ms)', fontsize=12)
ax.set_title(f'KNN pipeline scaling  |  m={m_val}, d={d_val}, k={k_val}, b={b_val}  |  GH200',
             fontsize=12)
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('runtimes.png', dpi=150)
plt.show()
print("Saved runtimes.png")

# -----------------------------
# print throughput summary
# -----------------------------
print(f"\n{'N':>6}  {'mean ms':>10}  {'datasets/s':>12}  {'compute ms':>10}  {'+mem ms':>12}")
print("-" * 56)
for n in Ns:
    mean_ms = np.mean(data[n])
    dps     = n / (mean_ms / 1e3)
    model    = napkin_time_ms(n)
    model_m  = napkin_time_mem_ms(n)
    print(f"{n:>6}  {mean_ms:>10.1f}  {dps:>12.1f}  {model:>10.1f}  {model_m:>12.1f}")
