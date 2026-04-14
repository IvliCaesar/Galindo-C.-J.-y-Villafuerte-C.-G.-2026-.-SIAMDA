"""
Módulo de Validación Teórica — Información de Fisher y Cota de Cramér-Rao.
RF-03: Cov(θ̂) ≥ 1/F(θ)
"""
import numpy as np
from scipy import stats


# ── Información de Fisher empírica ───────────────────────────────────────────

def fisher_gaussiano(theta_hat: np.ndarray) -> dict:
    """
    Bajo el modelo gaussiano θ ~ N(μ, σ²), el estimador de máxima
    verosimilitud es la media muestral.

    Información de Fisher para μ desconocida, σ² conocida (estimada):
        F(μ) = N / σ²

    Información de Fisher para σ² desconocida:
        F(σ²) = N / (2σ⁴)

    Retorna métricas completas para mostrar en la UI.
    """
    N = len(theta_hat)
    mu_hat    = float(np.mean(theta_hat))
    sigma_hat = float(np.std(theta_hat, ddof=1))
    var_hat   = sigma_hat ** 2

    # Información de Fisher
    F_mu    = N / (var_hat + 1e-12)
    F_sigma = N / (2 * var_hat ** 2 + 1e-12)

    # Cota de Cramér-Rao
    crb_mu    = 1.0 / F_mu
    crb_sigma = 1.0 / F_sigma

    # Varianza empírica del estimador (bootstrap ligero)
    np.random.seed(42)
    B = 500
    boots = [np.mean(np.random.choice(theta_hat, size=N, replace=True)) for _ in range(B)]
    var_empirica_mu = float(np.var(boots, ddof=1))

    # Test de normalidad (Shapiro-Wilk, máx 50 muestras)
    muestra_sw = theta_hat[:50] if N > 50 else theta_hat
    stat_sw, p_sw = stats.shapiro(muestra_sw)

    # Eficiencia del estimador (qué tan cerca está de la cota)
    eficiencia = float(crb_mu / (var_empirica_mu + 1e-12))
    eficiencia = min(eficiencia, 1.0)   # no puede superar 1 en teoría

    return {
        "N":               N,
        "mu_hat":          mu_hat,
        "sigma_hat":       sigma_hat,
        "var_hat":         var_hat,
        "F_mu":            F_mu,
        "F_sigma":         F_sigma,
        "crb_mu":          crb_mu,
        "crb_sigma":       crb_sigma,
        "var_empirica_mu": var_empirica_mu,
        "eficiencia":      eficiencia,
        "shapiro_stat":    float(stat_sw),
        "shapiro_p":       float(p_sw),
        "boots_mu":        np.array(boots),
    }


def fisher_laplaciano(theta_hat: np.ndarray) -> dict:
    """
    Bajo modelo Laplaciano θ ~ Laplace(μ, b):
        F(μ) = N / b²   (donde b = std/√2)
    Útil si los sentimientos tienen distribución de colas pesadas.
    """
    N  = len(theta_hat)
    mu = float(np.median(theta_hat))          # estimador robusto para Laplace
    b  = float(np.mean(np.abs(theta_hat - mu))) + 1e-12  # estimador de escala

    F_mu  = N / (b ** 2)
    crb   = 1.0 / F_mu

    return {
        "modelo":  "Laplaciano",
        "N":       N,
        "mu_hat":  mu,
        "b_hat":   b,
        "F_mu":    F_mu,
        "crb_mu":  crb,
    }


def calcular_cramer_rao(theta_hat: np.ndarray) -> dict:
    """
    Calcula ambos modelos y retorna un dict unificado.
    """
    gauss  = fisher_gaussiano(theta_hat)
    laplace = fisher_laplaciano(theta_hat)

    return {
        "gaussiano":  gauss,
        "laplaciano": laplace,
        "theta_hat":  theta_hat,
    }
