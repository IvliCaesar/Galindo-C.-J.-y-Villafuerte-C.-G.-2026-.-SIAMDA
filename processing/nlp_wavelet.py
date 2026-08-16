"""
Módulo de Análisis Cualitativo NLP + Transformada Wavelet.
RF-02: Embeddings → DWT → índice de sentimiento por alumno.

La DWT se aplica sobre la posición del token dentro de la respuesta (un eje con
adyacencia temporal genuina), no sobre las coordenadas del embedding de oración
(que no tienen orden ni noción de frecuencia intrínsecos: permutarlas no cambia
el significado del texto, pero sí cambiaría cualquier coeficiente de una DWT
aplicada sobre ese eje). Para cada token se proyecta su embedding contextual
sobre la dirección del embedding de oración, dando una señal escalar real
indexada por posición; esa señal sí tiene la estructura local/global que la DWT
está diseñada para separar.
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


def obtener_embeddings_token(
    textos: list[str], modelo_nombre: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Genera, en una sola pasada del modelo:
      - X ∈ R^{N x d}: embeddings de oración (normalizados, para el índice de sentimiento).
      - lista de N arrays R^{L_i x d}: embeddings contextuales por token (uno por
        respuesta, longitud L_i variable = número de tokens de esa respuesta,
        sin relleno).

    Ambos provienen del mismo forward pass del modelo, así que no hay costo
    adicional de inferencia frente a solo pedir el embedding de oración.
    """
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(modelo_nombre)

    X = modelo.encode(textos, show_progress_bar=False, normalize_embeddings=True)
    X = X.astype(np.float32)

    salida_tokens = modelo.encode(
        textos, show_progress_bar=False, output_value="token_embeddings"
    )
    # sentence-transformers regresa, para output_value="token_embeddings", una lista
    # de tensores (uno por texto) ya recortados a su longitud real (sin padding).
    tokens_por_texto = []
    for t in salida_tokens:
        arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        tokens_por_texto.append(arr.astype(np.float32))

    return X, tokens_por_texto


# ── Señal 1D por posición de token ─────────────────────────────────────────────

def señal_por_posicion(embedding_oracion: np.ndarray, embeddings_token: np.ndarray) -> np.ndarray:
    """
    Proyecta cada embedding de token sobre la dirección del embedding de oración
    (que está normalizado a norma unitaria), dando una señal escalar real
    s[t] = <embeddings_token[t], embedding_oracion> indexada por la posición t
    del token en la respuesta — un eje con adyacencia temporal genuina, a
    diferencia de las coordenadas del embedding.

    Valores altos de s[t] indican que ese token está alineado con el sentido
    global de la respuesta; caídas o picos locales indican palabras que se
    desvían de ese sentido (candidatas a carga emocional puntual).
    """
    return embeddings_token @ embedding_oracion


# ── Transformada Discreta de Onduleta ─────────────────────────────────────────

def aplicar_dwt_posicional(
    señales: list[np.ndarray], wavelet: str = "db4", nivel: int = 3
) -> dict:
    """
    Aplica DWT a cada señal 1D de la lista (una por alumno, indexada por
    posición de token, longitud variable).

    Parámetros
    ----------
    señales : lista de N arrays 1D (longitud L_i variable, L_i = núm. de tokens)
    wavelet : familia de onduleta (default Daubechies 4)
    nivel   : profundidad de descomposición deseada

    Retorna
    -------
    dict con:
      'aproximacion' : coeficientes cA (tendencia global de la respuesta) R^{N x k}
      'detalles'     : lista de arrays R^{N x k_i} por nivel (fluctuación local
                        palabra a palabra, con relleno a la longitud máxima para apilar)
      'energia_det'  : R^{N} energía total de coeficientes de detalle (proxy emocional)
      'energia_apr'  : R^{N} energía de aproximación
      'nivel'        : nivel de descomposición efectivamente usado (puede ser
                        menor que el solicitado si alguna respuesta es muy corta)
    """
    N = len(señales)

    # El nivel de descomposición está acotado por la respuesta MÁS CORTA del lote,
    # no por una dimensión fija (a diferencia del caso de embeddings de dimensión
    # constante): las respuestas tienen longitudes distintas en tokens.
    niveles_max = [pywt.dwt_max_level(max(len(s), 1), wavelet) for s in señales]
    nivel_efectivo = max(1, min(nivel, min(niveles_max)))

    aprox_list = []
    det_energias = np.zeros(N)
    apr_energias = np.zeros(N)
    det_coefs_all = [[] for _ in range(nivel_efectivo)]

    for i, señal in enumerate(señales):
        coefs = pywt.wavedec(señal, wavelet, level=nivel_efectivo)
        cA = coefs[0]
        aprox_list.append(cA)
        apr_energias[i] = np.sum(cA ** 2)
        e_det = 0.0
        for lvl_idx, cD in enumerate(coefs[1:]):
            det_coefs_all[lvl_idx].append(cD)
            e_det += np.sum(cD ** 2)
        det_energias[i] = e_det

    def pad_and_stack(lista):
        max_len = max(len(v) for v in lista)
        padded = [np.pad(v, (0, max_len - len(v))) for v in lista]
        return np.vstack(padded)

    return {
        "aproximacion": pad_and_stack(aprox_list),      # tendencia global de cada respuesta
        "detalles":     [pad_and_stack(det_coefs_all[l]) for l in range(nivel_efectivo)],
        "energia_det":  det_energias,                   # carga emocional aguda (local, en tokens)
        "energia_apr":  apr_energias,
        "wavelet":      wavelet,
        "nivel":        nivel_efectivo,
    }


# ── Índice de sentimiento ─────────────────────────────────────────────────────

def calcular_indice_sentimiento(wcoefs: dict) -> np.ndarray:
    """
    Índice de sentimiento θ̂ ∈ [-1, 1] para cada alumno.

    Estrategia:
      - Alta energía de detalle relativa → respuesta con fluctuación emocional
        local marcada palabra a palabra
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
      1. Embeddings de oración y de token (sentence-transformers)
      2. Señal escalar por posición de token (proyección sobre el embedding de oración)
      3. DWT sobre esa señal posicional
      4. Índice de sentimiento

    Retorna dict con embeddings de oración, coeficientes y θ̂.
    """
    if progress_callback:
        progress_callback(0.1, "Cargando modelo de embeddings…")

    X, tokens_por_texto = obtener_embeddings_token(textos, modelo_nombre)

    if progress_callback:
        progress_callback(0.4, "Construyendo señal posicional por respuesta…")

    señales = [
        señal_por_posicion(X[i], tokens_por_texto[i])
        for i in range(len(textos))
    ]

    if progress_callback:
        progress_callback(0.55, "Aplicando Transformada de Onduleta…")

    wcoefs = aplicar_dwt_posicional(señales, wavelet=wavelet, nivel=nivel)

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
