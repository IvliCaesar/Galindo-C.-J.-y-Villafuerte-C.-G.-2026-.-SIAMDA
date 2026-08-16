"""
Pruebas de processing/nlp_wavelet.py contra réplicas manuales (numpy / pywt
directo) de las fórmulas del artículo (galindo_villafuerte.tex, §3.2):
s_i(t) = <u_{i,t}, x_i>, E^apr = ||c_A||^2, E^det = sum_j ||c_{D,j}||^2,
theta_hat = tanh((E^det - E^apr) / (E^det + E^apr + eps)).

No requiere descargar el modelo de sentence-transformers: se construyen
embeddings sintéticos directamente, ejercitando solo la parte matemática
del pipeline (que es la que el artículo formaliza).
"""
import numpy as np
import pywt
import pytest

from processing.nlp_wavelet import (
    señal_por_posicion,
    aplicar_dwt_posicional,
    calcular_indice_sentimiento,
)

D = 384


def _embeddings_sinteticos(rng, L):
    emb_oracion = rng.normal(size=D)
    emb_oracion /= np.linalg.norm(emb_oracion)
    emb_tokens = rng.normal(size=(L, D))
    return emb_oracion, emb_tokens


def test_señal_por_posicion_es_el_producto_punto_token_a_token():
    rng = np.random.default_rng(1)
    emb_oracion, emb_tokens = _embeddings_sinteticos(rng, L=20)

    s = señal_por_posicion(emb_oracion, emb_tokens)

    esperado = np.array([np.dot(emb_tokens[t], emb_oracion) for t in range(len(emb_tokens))])
    assert s.shape == (20,)
    np.testing.assert_allclose(s, esperado, atol=1e-10)


def test_energias_coinciden_con_pywt_wavedec_directo():
    rng = np.random.default_rng(2)
    _, emb_tokens = _embeddings_sinteticos(rng, L=25)
    señal = señal_por_posicion(rng.normal(size=D), emb_tokens)

    wcoefs = aplicar_dwt_posicional([señal], wavelet="db4", nivel=3)
    nivel = wcoefs["nivel"]

    coefs = pywt.wavedec(señal, "db4", level=nivel)
    E_apr_manual = np.sum(coefs[0] ** 2)
    E_det_manual = sum(np.sum(c ** 2) for c in coefs[1:])

    assert wcoefs["energia_apr"][0] == pytest.approx(E_apr_manual)
    assert wcoefs["energia_det"][0] == pytest.approx(E_det_manual)


def test_indice_sentimiento_es_la_formula_tanh_del_articulo():
    wcoefs = {
        "energia_det": np.array([3.0, 0.0, 5.0]),
        "energia_apr": np.array([1.0, 0.0, 5.0]),
    }
    theta = calcular_indice_sentimiento(wcoefs)

    eps = 1e-9
    esperado = np.tanh((wcoefs["energia_det"] - wcoefs["energia_apr"])
                        / (wcoefs["energia_det"] + wcoefs["energia_apr"] + eps))
    np.testing.assert_allclose(theta, esperado)
    assert np.all(np.abs(theta) < 1.0)


def test_indice_sentimiento_signo_correcto():
    # E_det > E_apr (respuesta con "carga emocional" alta) -> theta_hat > 0
    # E_det < E_apr (respuesta "estable") -> theta_hat < 0
    wcoefs = {"energia_det": np.array([10.0, 1.0]), "energia_apr": np.array([1.0, 10.0])}
    theta = calcular_indice_sentimiento(wcoefs)
    assert theta[0] > 0
    assert theta[1] < 0


def test_nivel_efectivo_acotado_por_la_respuesta_mas_corta_del_lote():
    rng = np.random.default_rng(3)
    señales = []
    for L in [8, 40, 100]:
        _, emb_tokens = _embeddings_sinteticos(rng, L)
        señales.append(señal_por_posicion(rng.normal(size=D), emb_tokens))

    wcoefs = aplicar_dwt_posicional(señales, wavelet="db4", nivel=5)
    nivel_max_para_la_mas_corta = pywt.dwt_max_level(8, "db4")
    assert wcoefs["nivel"] <= max(1, nivel_max_para_la_mas_corta)
    assert wcoefs["nivel"] <= 5


def test_no_falla_con_respuestas_muy_cortas():
    # Caso límite: respuestas de solo unos pocos tokens (p.ej. "sin respuesta")
    rng = np.random.default_rng(4)
    señales = []
    for L in [3, 4, 5]:
        _, emb_tokens = _embeddings_sinteticos(rng, L)
        señales.append(señal_por_posicion(rng.normal(size=D), emb_tokens))

    wcoefs = aplicar_dwt_posicional(señales, wavelet="db4", nivel=3)
    theta = calcular_indice_sentimiento(wcoefs)
    assert wcoefs["nivel"] >= 1
    assert np.all(np.isfinite(theta))
    assert np.all(np.abs(theta) < 1.0)
