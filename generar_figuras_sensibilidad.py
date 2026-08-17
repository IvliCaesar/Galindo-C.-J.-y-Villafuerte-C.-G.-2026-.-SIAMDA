"""
Análisis de sensibilidad (simulación, no datos reales de curso) que EJECUTA
el código real de processing/ para explorar tres afirmaciones que el
artículo hace pero no mostraba numéricamente:

  (a) "con N<30 las estimaciones bootstrap de varianza son inestables"
      -> Figura S1: EES y ancho del intervalo bootstrap en función de N.
  (b) "el índice theta_hat depende de la familia de wavelet elegida"
      -> Figura S2: distribución de theta_hat para las 6 familias que
         ofrece la interfaz real (pages/sentimiento.py: WAVELETS_DISPONIBLES).
  (c) "...y del nivel de descomposición J"
      -> Figura S3: distribución de theta_hat para J=1..5, familia fija (db4).

Mismo generador de señales sintéticas que generar_figuras_articulo.py
(semilla fija), para que estas figuras sean comparables con las del cuerpo
principal del artículo. Ningún dato de alumnos reales se usa aquí tampoco.
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

AZUL = "#193E72"
ORO  = "#A07828"
ROJO = "#b23b3b"
GRIS = "#6b6b6b"

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


def señal_simulada(L, intensidad_emocional, rng):
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


def grupo_simulado(N, seed):
    rng = np.random.default_rng(seed)
    intensidades = rng.gamma(shape=2.0, scale=0.5, size=N)
    longitudes = rng.integers(12, 55, size=N)
    return [señal_simulada(L, e, rng) for L, e in zip(longitudes, intensidades)]


# ══════════════════════════════════════════════════════════════════════════
# S1. Sensibilidad a N: ¿qué tan estable es el bootstrap para N chico?
# ══════════════════════════════════════════════════════════════════════════

Ns = [8, 10, 15, 20, 30, 40, 60, 100, 150, 250]
n_repeticiones = 20  # repeticiones por N, con semillas distintas, para ver la dispersión

resultados_ees = {N: [] for N in Ns}
resultados_ancho_ic = {N: [] for N in Ns}  # ancho del intervalo bootstrap [mu-sqrt(CRB), mu+sqrt(CRB)]

for N in Ns:
    for rep in range(n_repeticiones):
        señales = grupo_simulado(N, seed=1000 * N + rep)
        wcoefs = aplicar_dwt_posicional(señales, wavelet="db4", nivel=3)
        theta_hat = calcular_indice_sentimiento(wcoefs)
        cr = calcular_cramer_rao(theta_hat)
        g = cr["gaussiano"]
        resultados_ees[N].append(g["eficiencia"])
        resultados_ancho_ic[N].append(2 * np.sqrt(g["var_empirica_mu"]))  # ancho del IC bootstrap

medias_ees = [np.mean(resultados_ees[N]) for N in Ns]
q25_ees = [np.percentile(resultados_ees[N], 25) for N in Ns]
q75_ees = [np.percentile(resultados_ees[N], 75) for N in Ns]

medias_ancho = [np.mean(resultados_ancho_ic[N]) for N in Ns]
sd_ancho = [np.std(resultados_ancho_ic[N]) for N in Ns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))

ax1.plot(Ns, medias_ees, color=AZUL, marker="o", markersize=4, linewidth=1.5)
ax1.fill_between(Ns, q25_ees, q75_ees, color=AZUL, alpha=0.18, label="rango intercuartil (20 réplicas)")
ax1.axvline(30, color=ROJO, linestyle="--", linewidth=1.1, label="$N=30$")
ax1.set_xlabel("$N$ (tamaño de la muestra simulada)")
ax1.set_ylabel("EES")
ax1.set_title("EES vs. $N$", fontsize=9.5)
ax1.legend(frameon=False, fontsize=7.5, loc="lower right")

ax2.plot(Ns, medias_ancho, color=ORO, marker="o", markersize=4, linewidth=1.5)
ax2.fill_between(Ns,
                  np.array(medias_ancho) - np.array(sd_ancho),
                  np.array(medias_ancho) + np.array(sd_ancho),
                  color=ORO, alpha=0.18, label="±1 d.e. entre réplicas")
ax2.axvline(30, color=ROJO, linestyle="--", linewidth=1.1, label="$N=30$")
ax2.set_xlabel("$N$ (tamaño de la muestra simulada)")
ax2.set_ylabel(r"ancho del IC bootstrap $2\sqrt{\widehat{\mathrm{Var}}(\bar\theta)}$")
ax2.set_title("Estabilidad del bootstrap vs. $N$", fontsize=9.5)
ax2.legend(frameon=False, fontsize=7.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sensibilidad_N.pdf"), bbox_inches="tight")
plt.close(fig)

print("S1 -- dispersión (d.e. entre réplicas) del ancho del IC bootstrap:")
for N in Ns:
    print(f"    N={N:>4}: d.e.={np.std(resultados_ancho_ic[N]):.4f}  "
          f"(media EES={np.mean(resultados_ees[N])*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════
# S2. Sensibilidad a la familia de wavelet (mismas señales, N=60 fijo)
# ══════════════════════════════════════════════════════════════════════════

FAMILIAS = ["db4", "db2", "haar", "sym4", "coif2", "dmey"]  # = WAVELETS_DISPONIBLES en pages/sentimiento.py
N_fijo = 60
señales_fijas = grupo_simulado(N_fijo, seed=7)  # misma semilla que generar_figuras_articulo.py

theta_por_familia = {}
for fam in FAMILIAS:
    wcoefs = aplicar_dwt_posicional(señales_fijas, wavelet=fam, nivel=3)
    theta_por_familia[fam] = calcular_indice_sentimiento(wcoefs)

fig, ax = plt.subplots(figsize=(7.5, 3.4))
datos_box = [theta_por_familia[f] for f in FAMILIAS]
bp = ax.boxplot(datos_box, tick_labels=FAMILIAS, patch_artist=True, widths=0.55,
                 medianprops=dict(color=ORO, linewidth=1.6),
                 boxprops=dict(facecolor=AZUL, alpha=0.35, edgecolor=AZUL),
                 whiskerprops=dict(color=AZUL), capprops=dict(color=AZUL),
                 flierprops=dict(markeredgecolor=AZUL, markersize=3))
ax.set_ylabel(r"$\hat\theta_i$")
ax.set_xlabel("familia de onduleta")
ax.set_title(f"Sensibilidad de $\\hat\\theta$ a la familia de onduleta (mismas {N_fijo} señales simuladas)",
             fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sensibilidad_familia.pdf"), bbox_inches="tight")
plt.close(fig)

print("\nS2 -- media y d.e. de theta_hat por familia (mismas señales, N=60):")
for fam in FAMILIAS:
    t = theta_por_familia[fam]
    print(f"    {fam:>6}: media={np.mean(t):+.4f}  d.e.={np.std(t):.4f}")

# Correlación entre pares de familias (¿qué tan de acuerdo están entre sí?)
print("\nS2b -- correlación de Pearson entre theta_hat de cada familia y db4 (referencia):")
ref = theta_por_familia["db4"]
for fam in FAMILIAS:
    r = np.corrcoef(ref, theta_por_familia[fam])[0, 1]
    print(f"    corr(db4, {fam:>6}) = {r:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# S3. Sensibilidad al nivel de descomposición J (familia db4 fija, N=60)
#
# HALLAZGO al correr esto por primera vez: con las longitudes realistas de
# fig_correlacion.pdf (12-54 tokens), el nivel efectivo queda atorado en 1
# para CUALQUIER J solicitado (1..5) -- la respuesta más corta del lote lo
# acota (pywt.dwt_max_level). Es decir, "J" no hace nada en la práctica
# cuando hay una sola respuesta corta en el grupo. Se muestra ese caso
# realista tal cual salió (panel izquierdo) junto con un caso de control de
# respuestas largas donde J sí varía el resultado (panel derecho), en vez de
# ocultar el resultado degenerado.
# ══════════════════════════════════════════════════════════════════════════

NIVELES = [1, 2, 3, 4, 5]

theta_por_nivel_corto = {}
nivel_efectivo_corto = {}
for J in NIVELES:
    wcoefs = aplicar_dwt_posicional(señales_fijas, wavelet="db4", nivel=J)
    theta_por_nivel_corto[J] = calcular_indice_sentimiento(wcoefs)
    nivel_efectivo_corto[J] = wcoefs["nivel"]

# Caso de control: mismas 60 "respuestas" pero con longitudes largas
# (150-400 tokens), para las cuales J sí alcanza a variar el nivel efectivo.
rng_largo = np.random.default_rng(7)
intensidades_largo = rng_largo.gamma(shape=2.0, scale=0.5, size=N_fijo)
longitudes_largo = rng_largo.integers(150, 400, size=N_fijo)
señales_largas = [señal_simulada(L, e, rng_largo) for L, e in zip(longitudes_largo, intensidades_largo)]

theta_por_nivel_largo = {}
nivel_efectivo_largo = {}
for J in NIVELES:
    wcoefs = aplicar_dwt_posicional(señales_largas, wavelet="db4", nivel=J)
    theta_por_nivel_largo[J] = calcular_indice_sentimiento(wcoefs)
    nivel_efectivo_largo[J] = wcoefs["nivel"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)

for ax, theta_por_nivel, nivel_efectivo, titulo, longitudes_txt in [
    (ax1, theta_por_nivel_corto, nivel_efectivo_corto,
     "Respuestas cortas (12--54 tokens)\n-- caso realista de encuesta --", None),
    (ax2, theta_por_nivel_largo, nivel_efectivo_largo,
     "Respuestas largas (150--400 tokens)\n-- caso de control --", None),
]:
    datos_box = [theta_por_nivel[J] for J in NIVELES]
    etiquetas = [f"$J={J}$\n(ef.: {nivel_efectivo[J]})" for J in NIVELES]
    ax.boxplot(datos_box, tick_labels=etiquetas, patch_artist=True, widths=0.55,
               medianprops=dict(color=ORO, linewidth=1.6),
               boxprops=dict(facecolor=AZUL, alpha=0.35, edgecolor=AZUL),
               whiskerprops=dict(color=AZUL), capprops=dict(color=AZUL),
               flierprops=dict(markeredgecolor=AZUL, markersize=3))
    ax.set_ylabel(r"$\hat\theta_i$")
    ax.set_title(titulo, fontsize=9.0)
ax1.set_xlabel("$J$ solicitado (nivel efectivo entre paréntesis)", fontsize=8.5)
ax2.set_xlabel("$J$ solicitado (nivel efectivo entre paréntesis)", fontsize=8.5)
fig.suptitle("Sensibilidad de $\\hat\\theta$ al nivel $J$: acotado por la respuesta más corta del lote",
             fontsize=9.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_sensibilidad_nivel.pdf"), bbox_inches="tight")
plt.close(fig)

print("\nS3a -- respuestas cortas (12-54 tok): nivel efectivo por J solicitado:")
for J in NIVELES:
    t = theta_por_nivel_corto[J]
    print(f"    J={J} (efectivo {nivel_efectivo_corto[J]}): media={np.mean(t):+.4f}  d.e.={np.std(t):.4f}")
print("S3b -- respuestas largas (150-400 tok): nivel efectivo por J solicitado:")
for J in NIVELES:
    t = theta_por_nivel_largo[J]
    print(f"    J={J} (efectivo {nivel_efectivo_largo[J]}): media={np.mean(t):+.4f}  d.e.={np.std(t):.4f}")

print("\nFiguras de sensibilidad generadas en:", OUT)
