"""
English-labeled counterpart of generar_figuras_articulo.py, for
galindo_villafuerte_en.tex. Same seed (7), same simulation code, same
underlying numbers as the Spanish figures -- only axis/legend/title text
is in English. Output goes to figs_en/, so the Spanish paper's figures
(figs/) are untouched.

No real student data is used here either -- see the module docstring in
generar_figuras_articulo.py for the full disclosure, which applies
identically to this script.
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

# Same visual identity as the Spanish figures (ihesblue / ihesgold)
BLUE  = "#193E72"   # ihesblue
GOLD  = "#A07828"   # ihesgold
GRAY  = "#6b6b6b"
rng = np.random.default_rng(7)  # same seed as the Spanish script -> identical numbers

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


# ══════════════════════════════════════════════════════════════════════════
# 1. Simulated per-token-position signal: two examples (stable vs. with
#    local emotional fluctuation), and its approximation/detail split.
# ══════════════════════════════════════════════════════════════════════════

def simulated_signal(L, emotional_intensity, rng):
    """
    1D signal indexed by token position: a smooth global trend (small-step
    random walk) plus local bursts at random positions, with amplitude
    proportional to `emotional_intensity`.
    """
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


L_example = 36
s_stable    = simulated_signal(L_example, emotional_intensity=0.15, rng=rng)
s_emotional = simulated_signal(L_example, emotional_intensity=1.6, rng=rng)

wc_example = aplicar_dwt_posicional([s_stable, s_emotional], wavelet="db4", nivel=3)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
for ax, signal, e_det, e_apr, title in zip(
    axes, [s_stable, s_emotional],
    wc_example["energia_det"], wc_example["energia_apr"],
    [r"Simulated response A: low $E^{\mathrm{det}}$", r"Simulated response B: high $E^{\mathrm{det}}$"],
):
    ax.plot(signal, color=BLUE, linewidth=1.3)
    ax.axhline(0, color=GRAY, linewidth=0.6, linestyle=":")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("token position $t$")
axes[0].set_ylabel(r"$s_i(t)$")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_signal_example.pdf"), bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 2-4. Simulated group of N=60 "students": signals -> theta_hat -> CRB -> correlation
# ══════════════════════════════════════════════════════════════════════════

N = 60
intensities = rng.gamma(shape=2.0, scale=0.5, size=N)   # simulated per-student emotional intensity
lengths = rng.integers(12, 55, size=N)

signals = [simulated_signal(L, e, rng) for L, e in zip(lengths, intensities)]
wcoefs = aplicar_dwt_posicional(signals, wavelet="db4", nivel=3)
theta_hat = calcular_indice_sentimiento(wcoefs)

cr = calcular_cramer_rao(theta_hat)
g = cr["gaussiano"]

# Simulated grade: correlated by construction with the simulated emotional
# intensity (not with real data), only to illustrate how to read Figure 4.
simulated_grade = np.clip(
    92 - 9.0 * intensities + rng.normal(0, 6, size=N), 40, 100
)

# ---- Figure 2: histogram of simulated theta_hat ----------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.hist(theta_hat, bins=14, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6)
ax.axvline(np.mean(theta_hat), color=GOLD, linewidth=1.6,
           label=fr"$\bar\theta = {np.mean(theta_hat):.3f}$ (simulated)")
ax.set_xlabel(r"$\hat\theta_i$")
ax.set_ylabel("number of students (simulated)")
ax.set_title("Distribution of $\\hat\\theta$ -- simulated group, $N=60$", fontsize=9.5)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_theta_hist.pdf"), bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: bootstrap of the mean vs. +-sqrt(CRB) band ------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
boots = g["boots_mu"]
ax.hist(boots, bins=28, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axvline(g["mu_hat"], color=GOLD, linewidth=1.6, label=r"$\bar\theta$")
crb_sd = np.sqrt(g["crb_mu"])
ax.axvline(g["mu_hat"] - crb_sd, color="#b23b3b", linestyle="--", linewidth=1.3,
           label=r"$\bar\theta \pm \sqrt{\mathrm{CRB}_\mu}$")
ax.axvline(g["mu_hat"] + crb_sd, color="#b23b3b", linestyle="--", linewidth=1.3)
ax.set_xlabel(r"$\bar\theta$ (bootstrap resamples)")
ax.set_ylabel("frequency")
ax.set_title(
    f"Bootstrap of $\\bar\\theta$ vs. Cramér--Rao band "
    f"(simulated EEE $= {g['eficiencia']*100:.1f}\\%$)", fontsize=8.8
)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_bootstrap_crb.pdf"), bbox_inches="tight")
plt.close(fig)

# ---- Figure 4: simulated correlation theta_hat vs. grade -------------------
r = np.corrcoef(theta_hat, simulated_grade)[0, 1]
m, b = np.polyfit(theta_hat, simulated_grade, 1)
x_line = np.linspace(theta_hat.min(), theta_hat.max(), 50)

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.scatter(theta_hat, simulated_grade, color=BLUE, alpha=0.75, s=26, edgecolor="white", linewidth=0.4)
ax.plot(x_line, m * x_line + b, color=GOLD, linewidth=1.6, label=fr"$r = {r:.2f}$ (simulated)")
ax.set_xlabel(r"$\hat\theta_i$ (simulated)")
ax.set_ylabel("grade (simulated)")
ax.set_title("Simulated correlation $\\hat\\theta$ vs. grade -- illustrative, not empirical",
             fontsize=8.8)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_correlation.pdf"), bbox_inches="tight")
plt.close(fig)

print("Figures generated in:", OUT)
print(f"  N={N}, mu_hat={g['mu_hat']:.4f}, EEE={g['eficiencia']*100:.2f}%, "
      f"shapiro_p={g['shapiro_p']:.4f}, r_simulated={r:.3f}")
