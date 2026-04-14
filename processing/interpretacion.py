"""
Módulo de interpretación automática de resultados.
Genera textos explicativos en español para cada métrica del dashboard.
"""
import numpy as np


# ── Colores para alertas ───────────────────────────────────────────────────────
def _box(texto: str, tipo: str = "info") -> str:
    """Retorna HTML de una caja de interpretación."""
    colores = {
        "bueno":    ("#1a2f1a", "#3fb950", "#7ee787"),
        "info":     ("#1a1f2f", "#58a6ff", "#79c0ff"),
        "atención": ("#2f2a1a", "#d29922", "#ffa657"),
        "riesgo":   ("#3d1f1f", "#f85149", "#ffa198"),
    }
    bg, border, text = colores.get(tipo, colores["info"])
    return (
        f"<div style='background:{bg};border-left:4px solid {border};"
        f"border-radius:4px;padding:0.9rem 1.1rem;margin:0.5rem 0;"
        f"color:{text};font-size:0.88rem;line-height:1.6;'>{texto}</div>"
    )


# ── Interpretación de métricas del curso ──────────────────────────────────────

def interpretar_promedio(prom: float) -> str:
    if prom >= 85:
        return _box(
            f" <b>Desempeño grupal excelente</b> (promedio {prom:.1f}). "
            "El grupo en su conjunto domina los contenidos del curso. "
            "Se recomienda considerar actividades de profundización o retos adicionales.",
            "bueno",
        )
    elif prom >= 70:
        return _box(
            f" <b>Desempeño grupal aceptable</b> (promedio {prom:.1f}). "
            "La mayoría de los alumnos superan el umbral aprobatorio. "
            "Existen oportunidades de mejora en los temas con mayor dispersión.",
            "info",
        )
    elif prom >= 60:
        return _box(
            f" <b>Desempeño grupal bajo</b> (promedio {prom:.1f}). "
            "Una parte significativa del grupo está en riesgo de reprobar. "
            "Se recomienda revisar la estrategia didáctica y activar tutorías de apoyo.",
            "atención",
        )
    else:
        return _box(
            f" <b>Desempeño grupal crítico</b> (promedio {prom:.1f}). "
            "El grupo como unidad no alcanza el umbral aprobatorio. "
            "Se recomienda una intervención inmediata: diagnóstico de prerrequisitos, "
            "replanteamiento de evaluaciones y sesiones de recuperación.",
            "riesgo",
        )


def interpretar_riesgo(n_riesgo: int, n_total: int) -> str:
    pct = n_riesgo / n_total * 100 if n_total > 0 else 0
    if pct == 0:
        return _box(
            " <b>Sin alumnos en riesgo detectados.</b> "
            "Todos los estudiantes superan el umbral aprobatorio y no presentan "
            "situaciones académicas especiales (reposición, final, baja).",
            "bueno",
        )
    elif pct <= 15:
        return _box(
            f" <b>{n_riesgo} alumno(s) en riesgo</b> ({pct:.1f}% del grupo). "
            "El número es manejable. Se recomienda contacto personalizado con "
            "cada estudiante afectado para identificar las causas raíz.",
            "atención",
        )
    elif pct <= 35:
        return _box(
            f" <b>{n_riesgo} alumnos en riesgo</b> ({pct:.1f}% del grupo). "
            "Más de un tercio del grupo está en situación vulnerable. "
            "Considerar sesiones de recuperación grupales y revisión del ritmo del curso.",
            "atención",
        )
    else:
        return _box(
            f" <b>{n_riesgo} alumnos en riesgo</b> ({pct:.1f}% del grupo). "
            "Más de la mitad del grupo está en riesgo. Esta cifra requiere "
            "atención institucional urgente: reporte a coordinación académica y "
            "plan de contingencia curricular.",
            "riesgo",
        )


def interpretar_parciales(promedios: dict) -> str:
    if not promedios:
        return ""
    vals = list(promedios.values())
    tendencia = vals[-1] - vals[0] if len(vals) > 1 else 0
    min_p = min(promedios, key=promedios.get)
    if tendencia > 5:
        return _box(
            f" <b>Tendencia positiva entre parciales</b> (+{tendencia:.1f} pts). "
            f"El grupo mejora con el avance del semestre, lo que sugiere que "
            f"la curva de aprendizaje es adecuada. "
            f"Parcial con menor desempeño: <b>{min_p}</b> ({promedios[min_p]:.1f} pts).",
            "bueno",
        )
    elif tendencia < -5:
        return _box(
            f" <b>Tendencia negativa entre parciales</b> ({tendencia:.1f} pts). "
            f"El grupo pierde rendimiento conforme avanza el semestre. "
            f"Revisar la dificultad progresiva del temario y la carga de trabajo. "
            f"Parcial más débil: <b>{min_p}</b> ({promedios[min_p]:.1f} pts).",
            "riesgo",
        )
    else:
        return _box(
            f" <b>Desempeño estable entre parciales</b> (variación {tendencia:+.1f} pts). "
            f"No hay mejora ni deterioro significativo. "
            f"Parcial con menor promedio: <b>{min_p}</b> ({promedios[min_p]:.1f} pts). "
            "Considerar estrategias diferenciadas para ese tema.",
            "info",
        )


# ── Interpretación de sentimiento ─────────────────────────────────────────────

def interpretar_distribucion_sentimiento(theta: np.ndarray) -> str:
    media  = float(np.mean(theta))
    pct_neg = float(np.mean(theta < -0.2) * 100)
    pct_pos = float(np.mean(theta >  0.2) * 100)
    pct_neu = 100 - pct_neg - pct_pos

    if media < -0.15:
        return _box(
            f" <b>Grupo predominantemente tranquilo</b> (θ̂ medio = {media:.3f}). "
            f"{pct_neg:.0f}% de los alumnos tienen respuestas afectivamente neutras o positivas. "
            "Las respuestas de la encuesta reflejan principalmente contenido reflexivo y contextual, "
            "con poca carga emocional aguda. Esto puede indicar alto nivel de confort con el curso.",
            "bueno",
        )
    elif media < 0.15:
        return _box(
            f" <b>Grupo con sentimiento mixto</b> (θ̂ medio = {media:.3f}). "
            f"{pct_neu:.0f}% de los alumnos se ubican en la zona neutra. "
            f"{pct_pos:.0f}% presentan carga emocional elevada. "
            "El grupo en promedio no muestra señales de alarma, pero existen casos individuales "
            "que merecen atención. Revisar la correlación con calificaciones.",
            "info",
        )
    else:
        return _box(
            f" <b>Grupo con alta carga emocional</b> (θ̂ medio = {media:.3f}). "
            f"{pct_pos:.0f}% de los alumnos presentan respuestas con carga emocional aguda alta. "
            "Esto puede indicar estrés, desmotivación, o dificultad percibida con los contenidos. "
            "Se recomienda revisar los comentarios cualitativos de los alumnos con θ̂ > 0.4 "
            "e implementar acciones de apoyo emocional y académico.",
            "riesgo",
        )


def interpretar_correlacion(r: float, p: float, n: int) -> str:
    sig = p < 0.05
    abs_r = abs(r)
    direccion = "negativa" if r < 0 else "positiva"
    interp_dir = (
        "los alumnos con mayor carga emocional en sus respuestas tienden a obtener calificaciones más bajas"
        if r < 0 else
        "los alumnos con mayor carga emocional tienden a obtener calificaciones más altas (posible motivación extrínseca)"
    )

    if abs_r >= 0.6 and sig:
        return _box(
            f" <b>Correlación {direccion} fuerte</b> (r = {r:.3f}, p = {p:.4f}, N = {n}). "
            f"El índice de sentimiento wavelet es un predictor significativo del desempeño: "
            f"{interp_dir}. "
            "Esta relación es estadísticamente robusta y puede usarse como señal de alerta temprana.",
            "riesgo" if r < 0 else "bueno",
        )
    elif abs_r >= 0.3 and sig:
        return _box(
            f" <b>Correlación {direccion} moderada</b> (r = {r:.3f}, p = {p:.4f}, N = {n}). "
            f"Existe una asociación estadísticamente significativa: {interp_dir}. "
            "La relación es real pero moderada; otros factores también explican el desempeño. "
            "El sentimiento aporta información complementaria valiosa para la toma de decisiones.",
            "atención",
        )
    elif sig:
        return _box(
            f" <b>Correlación {direccion} débil pero significativa</b> (r = {r:.3f}, p = {p:.4f}, N = {n}). "
            "El sentimiento y las calificaciones están relacionados, aunque la asociación es débil. "
            "Considerar ampliar la muestra o agregar más preguntas a la encuesta para capturar "
            "mejor la dimensión cualitativa.",
            "info",
        )
    else:
        return _box(
            f" <b>Correlación no significativa</b> (r = {r:.3f}, p = {p:.4f}, N = {n}). "
            "No se encontró evidencia estadística de asociación entre el índice de sentimiento "
            "y las calificaciones. Esto puede deberse a tamaño de muestra pequeño, "
            "preguntas de encuesta poco discriminativas, o que el desempeño es explicado "
            "por factores no capturados en el texto.",
            "info",
        )


def interpretar_energia_wavelet(media_det: float, media_apr: float) -> str:
    ratio = media_det / (media_apr + 1e-9)
    if ratio < 0.5:
        return _box(
            f" <b>Energía dominada por aproximación</b> (ratio det/apr = {ratio:.2f}). "
            "Las respuestas contienen principalmente contenido semántico estable y contextual. "
            "El vocabulario es consistente y los alumnos expresan ideas organizadas.",
            "bueno",
        )
    elif ratio < 1.2:
        return _box(
            f" <b>Balance entre aproximación y detalle</b> (ratio det/apr = {ratio:.2f}). "
            "Las respuestas combinan contenido contextual con fluctuaciones emocionales locales. "
            "El grupo expresa tanto ideas estructuradas como reacciones emocionales específicas.",
            "info",
        )
    else:
        return _box(
            f" <b>Energía dominada por coeficientes de detalle</b> (ratio det/apr = {ratio:.2f}). "
            "Las respuestas presentan alta variabilidad emocional localizada. "
            "Esto puede indicar que los alumnos expresan reacciones fuertes ante temas concretos "
            "del curso (exámenes, temas difíciles, carga de trabajo).",
            "riesgo",
        )


# ── Interpretación de Cramér-Rao ──────────────────────────────────────────────

def interpretar_crb(eficiencia: float, shapiro_p: float, N: int) -> str:
    msgs = []

    # Eficiencia
    if eficiencia >= 0.95:
        msgs.append(
            f" <b>Estimador altamente eficiente</b> (EES = {eficiencia:.1%}). "
            "La varianza empírica del estimador es prácticamente igual a la cota teórica "
            "de Cramér-Rao. No es posible reducir significativamente el error con el mismo N."
        )
        tipo = "bueno"
    elif eficiencia >= 0.80:
        msgs.append(
            f" <b>Eficiencia moderada</b> (EES = {eficiencia:.1%}). "
            "El estimador está razonablemente cerca de la cota, pero existe margen de mejora. "
            "Considera aumentar N, cambiar el modelo de embeddings, o ajustar el nivel wavelet J."
        )
        tipo = "atención"
    else:
        msgs.append(
            f" <b>Baja eficiencia</b> (EES = {eficiencia:.1%}). "
            "El estimador supera notablemente la cota de Cramér-Rao. "
            "El pipeline NLP-Wavelet actual no extrae toda la información disponible en los textos. "
            "Se recomienda: (1) aumentar N, (2) usar un modelo de embeddings más potente, "
            "(3) aumentar el nivel de descomposición wavelet."
        )
        tipo = "riesgo"

    # Normalidad
    if shapiro_p > 0.05:
        msgs.append(
            f" <b>Supuesto gaussiano verificado</b> (Shapiro-Wilk p = {shapiro_p:.4f} > 0.05). "
            "La distribución de θ̂ es compatible con una normal, lo que valida el modelo "
            "estadístico subyacente y la aplicabilidad directa de la cota de Cramér-Rao."
        )
    else:
        msgs.append(
            f" <b>Supuesto gaussiano no verificado</b> (Shapiro-Wilk p = {shapiro_p:.4f} < 0.05). "
            "La distribución de θ̂ se desvía de la normalidad. Considera el modelo Laplaciano "
            "(cuya cota se muestra arriba) o un estimador robusto basado en la mediana."
        )

    # Tamaño de muestra
    if N < 30:
        msgs.append(
            f" <b>Muestra pequeña</b> (N = {N}). "
            "Con menos de 30 observaciones, las estimaciones de varianza bootstrap son inestables. "
            "Los resultados deben interpretarse con cautela; ampliar la encuesta fortalecería las conclusiones."
        )

    return _box(" &nbsp;|&nbsp; ".join(msgs), tipo)


def interpretar_wordcloud(top_palabras: list[tuple]) -> str:
    if not top_palabras:
        return ""
    top5 = ", ".join([f"<b>{p}</b> ({f}x)" for p, f in top_palabras[:5]])
    return _box(
        f" <b>Palabras más frecuentes en las encuestas:</b> {top5}. "
        "El tamaño de cada palabra en la nube es proporcional a su frecuencia de aparición. "
        "Las palabras más grandes representan los conceptos o temas que los alumnos mencionan "
        "con mayor recurrencia en sus respuestas abiertas. "
        "Compara estos términos con el índice de sentimiento para identificar qué temas "
        "generan mayor carga emocional.",
        "info",
    )
