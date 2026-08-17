"""
English-labeled counterpart of generar_figuras_sensibilidad.py, for
galindo_villafuerte_en.tex. Same seeds, same simulation code, same
underlying numbers -- only axis/legend/title text is in English. Output
goes to figs_en/. No real student data is used here either.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from processing.nlp_wavelet import aplicar_dwt_posicional, calcular_indice_sentimiento
from processing.cramer_rao import calcular_cramer_rao

OUT = os.path.join(os.path.dirname(__file__), "figs_en")
os.makedirs(OUT, exist_ok=True)

BLUE = "#193E72"
GOLD = "#A07828"
RED  = "#b23b3b"
GRAY = "#6b6b6b"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "figure.dpi": 200,
})


def simulated_signal(L, emotional_intensity, rng):
    trend = np.cumsum(rng.normal(0, 0.05, size=L))
    signal = trend.copy()
    n_bursts = rng.integers(1, 4)
    for _ in range(n_bursts):
        pos = rng.integers(0, L)
        width = rng.integers(1, 3)
        amp = emotional_intensity * rng.choice([-1, 1]) * rng.uniform(0.8, 1.4)
        start, end = max(0, pos - width), min(L, pos + width + 1)
        signal[start:end] += amp
    return signal


def simulated_group(N, seed):
    rng = np.random.default_rng(seed)
    intensities = rng.gamma(shape=2.0, scale=0.5, size=N)
    lengths = rng.integers(12, 55, size=N)
    return [simulated_signal(L, e, rng) for L, e in zip(lengths, intensities)]


# ══════════════════════════════════════════════════════════════════════════
# S1. Sensitivity to N
# ══════════════════════════════════════════════════════════════════════════

Ns = [8, 10, 15, 20, 30, 40, 60, 100, 150, 250]
n_reps = 20

eee_results = {N: [] for N in Ns}
ci_width_results = {N: [] for N in Ns}

for N in Ns:
    for rep in range(n_reps):
        signals = simulated_group(N, seed=1000 * N + rep)
        wcoefs = aplicar_dwt_posicional(signals, wavelet="db4", nivel=3)
        theta_hat = calcular_indice_sentimiento(wcoefs)
        cr = calcular_cramer_rao(theta_hat)
        g = cr["gaussiano"]
        eee_results[N].append(g["eficiencia"])
        ci_width_results[N].append(2 * np.sqrt(g["var_empirica_mu"]))

eee_means = [np.mean(eee_results[N]) for N in Ns]
eee_q25 = [np.percentile(eee_results[N], 25) for N in Ns]
eee_q75 = [np.percentile(eee_results[N], 75) for N in Ns]

width_means = [np.mean(ci_width_results[N]) for N in Ns]
width_sds = [np.std(ci_width_results[N]) for N in Ns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))

ax1.plot(Ns, eee_means, color=BLUE, marker="o", markersize=4, linewidth=1.5)
ax1.fill_between(Ns, eee_q25, eee_q75, color=BLUE, alpha=0.18, label="interquartile range (20 reps)")
ax1.axvline(30, color=RED, linestyle="--", linewidth=1.1, label="$N=30$")
ax1.set_xlabel("$N$ (simulated sample size)")
ax1.set_ylabel("EEE")
ax1.set_title("EEE vs. $N$", fontsize=9.5)
ax1.legend(frameon=False, fontsize=7.5, loc="lower right")

ax2.plot(Ns, width_means, color=GOLD, marker="o", markersize=4, linewidth=1.5)
ax2.fill_between(Ns,
                  np.array(width_means) - np.array(width_sds),
                  np.array(width_means) + np.array(width_sds),
                  color=GOLD, alpha=0.18, label="±1 s.d. across reps")
ax2.axvline(30, color=RED, linestyle="--", linewidth=1.1, label="$N=30$")
ax2.set_xlabel("$N$ (simulated sample size)")
ax2.set_ylabel(r"bootstrap CI width $2\sqrt{\widehat{\mathrm{Var}}(\bar\theta)}$")
ax2.set_title("Bootstrap stability vs. $N$", fontsize=9.5)
ax2.legend(frameon=False, fontsize=7.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sensitivity_N.pdf"), bbox_inches="tight")
plt.close(fig)

print("S1 -- dispersion (s.d. across reps) of bootstrap CI width:")
for N in Ns:
    print(f"    N={N:>4}: s.d.={np.std(ci_width_results[N]):.4f}  "
          f"(mean EEE={np.mean(eee_results[N])*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# S2. Sensitivity to wavelet family (same signals, N=60 fixed)
# ══════════════════════════════════════════════════════════════════════════

FAMILIES = ["db4", "db2", "haar", "sym4", "coif2", "dmey"]
N_fixed = 60
fixed_signals = simulated_group(N_fixed, seed=7)

theta_by_family = {}
for fam in FAMILIES:
    wcoefs = aplicar_dwt_posicional(fixed_signals, wavelet=fam, nivel=3)
    theta_by_family[fam] = calcular_indice_sentimiento(wcoefs)

fig, ax = plt.subplots(figsize=(7.5, 3.4))
box_data = [theta_by_family[f] for f in FAMILIES]
ax.boxplot(box_data, tick_labels=FAMILIES, patch_artist=True, widths=0.55,
           medianprops=dict(color=GOLD, linewidth=1.6),
           boxprops=dict(facecolor=BLUE, alpha=0.35, edgecolor=BLUE),
           whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
           flierprops=dict(markeredgecolor=BLUE, markersize=3))
ax.set_ylabel(r"$\hat\theta_i$")
ax.set_xlabel("wavelet family")
ax.set_title(f"Sensitivity of $\\hat\\theta$ to wavelet family (same {N_fixed} simulated signals)",
             fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sensitivity_family.pdf"), bbox_inches="tight")
plt.close(fig)

print("\nS2 -- mean and s.d. of theta_hat by family (same signals, N=60):")
for fam in FAMILIES:
    t = theta_by_family[fam]
    print(f"    {fam:>6}: mean={np.mean(t):+.4f}  s.d.={np.std(t):.4f}")

print("\nS2b -- Pearson correlation between each family's theta_hat and db4 (reference):")
ref = theta_by_family["db4"]
for fam in FAMILIES:
    r = np.corrcoef(ref, theta_by_family[fam])[0, 1]
    print(f"    corr(db4, {fam:>6}) = {r:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# S3. Sensitivity to decomposition level J (db4 fixed, N=60)
# ══════════════════════════════════════════════════════════════════════════

LEVELS = [1, 2, 3, 4, 5]

theta_by_level_short = {}
effective_level_short = {}
for J in LEVELS:
    wcoefs = aplicar_dwt_posicional(fixed_signals, wavelet="db4", nivel=J)
    theta_by_level_short[J] = calcular_indice_sentimiento(wcoefs)
    effective_level_short[J] = wcoefs["nivel"]

rng_long = np.random.default_rng(7)
intensities_long = rng_long.gamma(shape=2.0, scale=0.5, size=N_fixed)
lengths_long = rng_long.integers(150, 400, size=N_fixed)
long_signals = [simulated_signal(L, e, rng_long) for L, e in zip(lengths_long, intensities_long)]

theta_by_level_long = {}
effective_level_long = {}
for J in LEVELS:
    wcoefs = aplicar_dwt_posicional(long_signals, wavelet="db4", nivel=J)
    theta_by_level_long[J] = calcular_indice_sentimiento(wcoefs)
    effective_level_long[J] = wcoefs["nivel"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)

for ax, theta_by_level, effective_level, title in [
    (ax1, theta_by_level_short, effective_level_short,
     "Short responses (12--54 tokens)\n-- realistic survey case --"),
    (ax2, theta_by_level_long, effective_level_long,
     "Long responses (150--400 tokens)\n-- control case --"),
]:
    box_data = [theta_by_level[J] for J in LEVELS]
    labels = [f"$J={J}$\n(eff.: {effective_level[J]})" for J in LEVELS]
    ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.55,
               medianprops=dict(color=GOLD, linewidth=1.6),
               boxprops=dict(facecolor=BLUE, alpha=0.35, edgecolor=BLUE),
               whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
               flierprops=dict(markeredgecolor=BLUE, markersize=3))
    ax.set_ylabel(r"$\hat\theta_i$")
    ax.set_title(title, fontsize=9.0)
ax1.set_xlabel("Requested $J$ (effective level in parentheses)", fontsize=8.5)
ax2.set_xlabel("Requested $J$ (effective level in parentheses)", fontsize=8.5)
fig.suptitle("Sensitivity of $\\hat\\theta$ to level $J$: bounded by the shortest response in the batch",
             fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sensitivity_level.pdf"), bbox_inches="tight")
plt.close(fig)

print("\nS3a -- short responses (12-54 tok): effective level by requested J:")
for J in LEVELS:
    t = theta_by_level_short[J]
    print(f"    J={J} (effective {effective_level_short[J]}): mean={np.mean(t):+.4f}  s.d.={np.std(t):.4f}")
print("S3b -- long responses (150-400 tok): effective level by requested J:")
for J in LEVELS:
    t = theta_by_level_long[J]
    print(f"    J={J} (effective {effective_level_long[J]}): mean={np.mean(t):+.4f}  s.d.={np.std(t):.4f}")

print("\nSensitivity figures generated in:", OUT)
