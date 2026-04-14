"""
Módulo de Análisis Cualitativo NLP + Transformada Wavelet.
RF-02: Embeddings → DWT → índice de sentimiento por alumno.
"""
import numpy as np
import pywt
from typing import Optional


# ── Embeddings ────────────────────────────────────────────────────────────────

def obtener_embeddings(textos: list[str], modelo_nombre: str = "paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """
    Genera embeddings de oración usando sentence-transformers.
    Retorna X ∈ R^{N x d}.
    """
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(modelo_nombre)
    embeddings = modelo.encode(textos, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.astype(np.float32)


# ── Transformada Discreta de Onduleta ─────────────────────────────────────────

def aplicar_dwt(X: np.ndarray, wavelet: str = "db4", nivel: int = 3) -> dict:
    """
    Aplica DWT fila por fila (por alumno) sobre el vector de embedding.

    Parámetros
    ----------
    X       : R^{N x d} — matriz de embeddings
    wavelet : familia de onduleta (default Daubechies 4)
    nivel   : profundidad de descomposición

    Retorna
    -------
    dict con:
      'aproximacion' : coeficientes cA (contexto global) R^{N x k}
      'detalles'     : lista de arrays R^{N x k_i} por nivel
      'energia_det'  : R^{N} energía total de coeficientes de detalle (proxy emocional)
      'energia_apr'  : R^{N} energía de aproximación
    """
    N, d = X.shape
    nivel_max = pywt.dwt_max_level(d, wavelet)
    nivel = min(nivel, nivel_max)

    aprox_list = []
    det_energias = np.zeros(N)
    apr_energias = np.zeros(N)
    det_coefs_all = [[] for _ in range(nivel)]

    for i in range(N):
        señal = X[i]  # vector d-dimensional como "señal 1D"
        coefs = pywt.wavedec(señal, wavelet, level=nivel)
        # coefs[0] = aproximación, coefs[1..] = detalles nivel a nivel
        cA = coefs[0]
        aprox_list.append(cA)
        apr_energias[i] = np.sum(cA ** 2)
        e_det = 0.0
        for lvl_idx, cD in enumerate(coefs[1:]):
            det_coefs_all[lvl_idx].append(cD)
            e_det += np.sum(cD ** 2)
        det_energias[i] = e_det

    # Homogeneizar longitudes para apilar
    def pad_and_stack(lista):
        max_len = max(len(v) for v in lista)
        padded = [np.pad(v, (0, max_len - len(v))) for v in lista]
        return np.vstack(padded)

    return {
        "aproximacion": pad_and_stack(aprox_list),      # contexto global
        "detalles":     [pad_and_stack(det_coefs_all[l]) for l in range(nivel)],
        "energia_det":  det_energias,                   # carga emocional aguda
        "energia_apr":  apr_energias,
        "wavelet":      wavelet,
        "nivel":        nivel,
    }


# ── Índice de sentimiento ─────────────────────────────────────────────────────

def calcular_indice_sentimiento(wcoefs: dict) -> np.ndarray:
    """
    Índice de sentimiento θ̂ ∈ [-1, 1] para cada alumno.
    
    Estrategia:
      - Alta energía de detalle relativa → respuesta emocionalmente cargada (negativa o positiva)
      - Normalizamos respecto al total de energía (detalle + aproximación)
      - Luego re-escalamos a [-1, 1] usando tanh para interpretabilidad
    
    θ̂_i = tanh( (E_det_i - E_apr_i) / (E_det_i + E_apr_i + ε) )
    """
    eps = 1e-9
    E_d = wcoefs["energia_det"]
    E_a = wcoefs["energia_apr"]
    ratio = (E_d - E_a) / (E_d + E_a + eps)
    return np.tanh(ratio)


# ── Análisis completo ─────────────────────────────────────────────────────────

def pipeline_nlp_wavelet(
    textos: list[str],
    wavelet: str = "db4",
    nivel: int = 3,
    modelo_nombre: str = "paraphrase-multilingual-MiniLM-L12-v2",
    progress_callback=None,
) -> dict:
    """
    Pipeline completo:
      1. Embeddings (sentence-transformers)
      2. DWT sobre matriz de embeddings
      3. Índice de sentimiento

    Retorna dict con embeddings, coeficientes y θ̂.
    """
    if progress_callback:
        progress_callback(0.1, "Cargando modelo de embeddings…")

    X = obtener_embeddings(textos, modelo_nombre)

    if progress_callback:
        progress_callback(0.55, "Aplicando Transformada de Onduleta…")

    wcoefs = aplicar_dwt(X, wavelet=wavelet, nivel=nivel)

    if progress_callback:
        progress_callback(0.85, "Calculando índice de sentimiento…")

    theta_hat = calcular_indice_sentimiento(wcoefs)

    if progress_callback:
        progress_callback(1.0, "Listo.")

    return {
        "embeddings":  X,
        "wcoefs":      wcoefs,
        "theta_hat":   theta_hat,
    }
