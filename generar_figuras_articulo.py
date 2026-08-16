"""
Genera las figuras del artículo (articulo_galindo_villafuente.tex) EJECUTANDO
el código real de processing/ (aplicar_dwt_posicional, calcular_indice_sentimiento,
calcular_cramer_rao) sobre señales SIMULADAS, no sobre datos reales de curso.

No se usa ningún dato de alumnos reales. Las señales y calificaciones se generan
sintéticamente, con una correlación impuesta por construcción entre "intensidad
emocional" simulada y "calificación" simulada, únicamente para ilustrar cómo se
lee cada gráfica. Esto se declara explícitamente en el artículo.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from processing.nlp_wavelet import aplicar_dwt_posicional, calcular_indice_sentimiento
from processing.cramer_rao import calcular_cramer_rao

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

# ── Identidad visual: los mismos dos colores del documento (ihesblue / ihesgold) ──
AZUL  = "#193E72"   # ihesblue
ORO   = "#A07828"   # ihesgold
GRIS  = "#6b6b6b"
rng = np.random.default_rng(7)

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
# 1. Señal simulada por posición de token: dos ejemplos (estable vs. con
#    fluctuación emocional local), y su descomposición aproximación/detalle.
# ══════════════════════════════════════════════════════════════════════════

def señal_simulada(L, intensidad_emocional, rng):
    """
    Señal 1D indexada por posición de token: una tendencia global suave
    (caminata aleatoria de paso chico) más ráfagas locales en posiciones
    aleatorias, de amplitud proporcional a `intensidad_emocional`.
    """
    tendencia = np.cumsum(rng.normal(0, 0.05, size=L))
    señal = tendencia.copy()
    n_rafagas = rng.integers(1, 4)
    for _ in range(n_rafagas):
        pos = rng.integers(0, L)
        ancho = rng.integers(1, 3)
        amp = intensidad_emocional * rng.choice([-1, 1]) * rng.uniform(0.8, 1.4)
        ini, fin = max(0, pos - ancho), min(L, pos + ancho + 1)
        señal[ini:fin] += amp
    return señal


L_ejemplo = 36
s_estable  = señal_simulada(L_ejemplo, intensidad_emocional=0.15, rng=rng)
s_emocional = señal_simulada(L_ejemplo, intensidad_emocional=1.6, rng=rng)

wc_ejemplo = aplicar_dwt_posicional([s_estable, s_emocional], wavelet="db4", nivel=3)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
for ax, señal, energia_det, energia_apr, titulo in zip(
    axes, [s_estable, s_emocional],
    wc_ejemplo["energia_det"], wc_ejemplo["energia_apr"],
    [r"Respuesta simulada A: baja $E^{\mathrm{det}}$", r"Respuesta simulada B: alta $E^{\mathrm{det}}$"],
):
    ax.plot(señal, color=AZUL, linewidth=1.3)
    ax.axhline(0, color=GRIS, linewidth=0.6, linestyle=":")
    ax.set_title(titulo, fontsize=9.5)
    ax.set_xlabel("posición del token $t$")
axes[0].set_ylabel(r"$s_i(t)$")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_señal_ejemplo.pdf"), bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 2-4. Grupo simulado de N=60 "alumnos": señales -> θ̂ -> CRB -> correlación
# ══════════════════════════════════════════════════════════════════════════

N = 60
intensidades = rng.gamma(shape=2.0, scale=0.5, size=N)   # intensidad emocional simulada por alumno
longitudes = rng.integers(12, 55, size=N)

señales = [señal_simulada(L, e, rng) for L, e in zip(longitudes, intensidades)]
wcoefs = aplicar_dwt_posicional(señales, wavelet="db4", nivel=3)
theta_hat = calcular_indice_sentimiento(wcoefs)

cr = calcular_cramer_rao(theta_hat)
g = cr["gaussiano"]

# Calificación simulada: correlacionada por construcción con la intensidad
# emocional simulada (no con datos reales), solo para ilustrar la lectura
# de la Figura 4.
calificacion_sim = np.clip(
    92 - 9.0 * intensidades + rng.normal(0, 6, size=N), 40, 100
)

# ---- Figura 2: histograma de θ̂ simulado -----------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.hist(theta_hat, bins=14, color=AZUL, alpha=0.85, edgecolor="white", linewidth=0.6)
ax.axvline(np.mean(theta_hat), color=ORO, linewidth=1.6,
           label=fr"$\bar\theta = {np.mean(theta_hat):.3f}$ (simulado)")
ax.set_xlabel(r"$\hat\theta_i$")
ax.set_ylabel("núm. de alumnos (simulados)")
ax.set_title("Distribución de $\\hat\\theta$ — grupo simulado, $N=60$", fontsize=9.5)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_theta_hist.pdf"), bbox_inches="tight")
plt.close(fig)

# ---- Figura 3: bootstrap de la media vs. banda ±√CRB -----------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2))
boots = g["boots_mu"]
ax.hist(boots, bins=28, color=AZUL, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axvline(g["mu_hat"], color=ORO, linewidth=1.6, label=r"$\bar\theta$")
crb_sd = np.sqrt(g["crb_mu"])
ax.axvline(g["mu_hat"] - crb_sd, color="#b23b3b", linestyle="--", linewidth=1.3,
           label=r"$\bar\theta \pm \sqrt{\mathrm{CRB}_\mu}$")
ax.axvline(g["mu_hat"] + crb_sd, color="#b23b3b", linestyle="--", linewidth=1.3)
ax.set_xlabel(r"$\bar\theta$ (remuestreos bootstrap)")
ax.set_ylabel("frecuencia")
ax.set_title(
    f"Bootstrap de $\\bar\\theta$ vs. banda de Cramér--Rao "
    f"(EES simulada $= {g['eficiencia']*100:.1f}\\%$)", fontsize=8.8
)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_bootstrap_crb.pdf"), bbox_inches="tight")
plt.close(fig)

# ---- Figura 4: correlación simulada θ̂ vs. calificación --------------------
r = np.corrcoef(theta_hat, calificacion_sim)[0, 1]
m, b = np.polyfit(theta_hat, calificacion_sim, 1)
x_line = np.linspace(theta_hat.min(), theta_hat.max(), 50)

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.scatter(theta_hat, calificacion_sim, color=AZUL, alpha=0.75, s=26, edgecolor="white", linewidth=0.4)
ax.plot(x_line, m * x_line + b, color=ORO, linewidth=1.6, label=fr"$r = {r:.2f}$ (simulado)")
ax.set_xlabel(r"$\hat\theta_i$ (simulado)")
ax.set_ylabel("calificación (simulada)")
ax.set_title("Correlación simulada $\\hat\\theta$ vs. calificación — ilustrativa, no empírica",
             fontsize=8.8)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_correlacion.pdf"), bbox_inches="tight")
plt.close(fig)

print("Figuras generadas en:", OUT)
print(f"  N={N}, mu_hat={g['mu_hat']:.4f}, EES={g['eficiencia']*100:.2f}%, "
      f"shapiro_p={g['shapiro_p']:.4f}, r_simulado={r:.3f}")
