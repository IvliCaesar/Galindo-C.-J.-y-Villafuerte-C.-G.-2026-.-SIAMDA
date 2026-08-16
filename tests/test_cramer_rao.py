"""
Pruebas de processing/cramer_rao.py contra réplicas manuales de las fórmulas
del artículo (galindo_villafuerte.tex, §3.3): F(mu)=N/sigma^2, CRB=1/F(mu),
modelo Laplaciano F(mu)=N/b^2, y EES=min(CRB/Var_boot, 1).
"""
import numpy as np
import pytest
from scipy import stats

from processing.cramer_rao import (
    calcular_cramer_rao,
    fisher_gaussiano,
    fisher_laplaciano,
)

MUESTRA = np.array([
    0.12, -0.34, 0.05, -0.61, 0.22, -0.08, 0.41, -0.15, 0.03, -0.29,
    0.18, -0.44, 0.09, -0.02, 0.37,
])


def test_gaussiano_formulas_coinciden_con_el_articulo():
    N = len(MUESTRA)
    g = fisher_gaussiano(MUESTRA)

    mu_hat = np.mean(MUESTRA)
    sigma_hat = np.std(MUESTRA, ddof=1)
    var_hat = sigma_hat**2
    F_mu = N / var_hat
    crb_mu = 1.0 / F_mu

    assert g["mu_hat"] == pytest.approx(mu_hat)
    assert g["sigma_hat"] == pytest.approx(sigma_hat)
    assert g["var_hat"] == pytest.approx(var_hat)
    # tolerancia amplia por el epsilon de estabilidad numérica (+1e-12) del código
    assert g["F_mu"] == pytest.approx(F_mu, rel=1e-8)
    assert g["crb_mu"] == pytest.approx(crb_mu, rel=1e-8)
    assert g["crb_mu"] == pytest.approx(1.0 / g["F_mu"])


def test_laplaciano_formulas_coinciden_con_el_articulo():
    N = len(MUESTRA)
    lp = fisher_laplaciano(MUESTRA)

    mu_hat = np.median(MUESTRA)
    b_hat = np.mean(np.abs(MUESTRA - mu_hat))
    F_mu = N / b_hat**2
    crb_mu = 1.0 / F_mu

    assert lp["mu_hat"] == pytest.approx(mu_hat)
    assert lp["b_hat"] == pytest.approx(b_hat)
    assert lp["F_mu"] == pytest.approx(F_mu, rel=1e-8)
    assert lp["crb_mu"] == pytest.approx(crb_mu, rel=1e-8)


def test_shapiro_wilk_coincide_con_scipy_directo():
    g = fisher_gaussiano(MUESTRA)
    stat, p = stats.shapiro(MUESTRA)
    assert g["shapiro_stat"] == pytest.approx(stat)
    assert g["shapiro_p"] == pytest.approx(p)


def test_eficiencia_es_el_cociente_crb_sobre_varianza_empirica_topado_en_1():
    g = fisher_gaussiano(MUESTRA)
    esperado = min(g["crb_mu"] / g["var_empirica_mu"], 1.0)
    assert g["eficiencia"] == pytest.approx(esperado)
    assert 0.0 <= g["eficiencia"] <= 1.0


def test_eficiencia_es_practicamente_1_para_muestra_gaussiana_bien_comportada():
    # Bajo el modelo gaussiano, la media muestral es SIEMPRE eficiente
    # (Lehmann-Scheffe) -- documentado como limitación en el articulo, §3.3.
    rng = np.random.default_rng(0)
    muestra_normal = rng.normal(loc=0.1, scale=0.3, size=200)
    g = fisher_gaussiano(muestra_normal)
    assert g["eficiencia"] > 0.85


def test_calcular_cramer_rao_agrega_ambos_modelos():
    resultado = calcular_cramer_rao(MUESTRA)
    assert set(resultado.keys()) == {"gaussiano", "laplaciano", "theta_hat"}
    assert resultado["gaussiano"]["N"] == len(MUESTRA)
    assert resultado["laplaciano"]["N"] == len(MUESTRA)
    np.testing.assert_array_equal(resultado["theta_hat"], MUESTRA)


def test_bootstrap_produce_B_igual_a_500_remuestreos():
    g = fisher_gaussiano(MUESTRA)
    assert len(g["boots_mu"]) == 500
