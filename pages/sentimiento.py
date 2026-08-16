"""
Página: Análisis de Sentimiento (Wavelets)
RF-02 / RF-04: NLP → Embeddings → DWT → Correlación con calificaciones.
Incluye: nube de palabras clave + interpretaciones automáticas.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import state
from processing.nlp_wavelet import pipeline_nlp_wavelet
from processing.cramer_rao import calcular_cramer_rao
from processing.wordcloud_gen import (
    calcular_frecuencias, generar_wordcloud_imagen, imagen_a_bytes
)
from processing.interpretacion import (
    interpretar_distribucion_sentimiento,
    interpretar_correlacion,
    interpretar_energia_wavelet,
    interpretar_wordcloud,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
)

WAVELETS_DISPONIBLES = ["db4", "db2", "haar", "sym4", "coif2", "dmey"]


def render():
    state.init()
    st.title(" Análisis de Sentimiento (Wavelets)")

    if not state.listo_encuesta():
        st.warning(" Primero carga el archivo de encuesta en **Carga de Datos**.")
        return

    df_enc = state.get("df_encuesta")
    textos = df_enc["_texto_concat"].tolist()
    ids    = df_enc["_id"].tolist()

    n_textos = len(textos)
    n_vacios = sum(1 for t in textos if not t.strip())
    st.info(f" {n_textos} respuestas cargadas · {n_vacios} vacías · "
            f"{n_textos - n_vacios} listas para análisis")

    # ── Parámetros ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Parámetros del análisis</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        wavelet = st.selectbox("Familia de onduleta:", WAVELETS_DISPONIBLES, index=0,
                               help="db4 (Daubechies-4) es la más recomendada para señales de lenguaje natural.")
    with c2:
        nivel = st.slider("Nivel de descomposición J:", 1, 5, 3,
                          help="Mayor J = más resolución temporal sobre la señal por posición de token. "
                               "El nivel efectivo queda acotado por la respuesta más corta del lote.")
    with c3:
        modelo_emb = st.selectbox(
            "Modelo de embeddings:",
            ["paraphrase-multilingual-MiniLM-L12-v2",
             "distiluse-base-multilingual-cased-v2"],
            index=0,
            help="MiniLM es el más eficiente para español. distiluse es más lento pero más preciso.",
        )

    st.markdown(
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;"
        "padding:0.7rem 1rem;font-size:0.82rem;color:#8b949e;margin-bottom:0.5rem;'>"
        "ℹ <b>¿Cómo funciona?</b> Cada respuesta de texto se convierte en un embedding "
        "contextual por palabra (token) usando un modelo de lenguaje, y cada palabra se "
        "proyecta sobre el sentido global de la respuesta, dando una señal numérica indexada "
        "por posición en el texto. Esa señal se descompone con la Transformada Discreta de "
        "Onduleta (DWT): los <b>coeficientes de aproximación</b> capturan la tendencia global "
        "de la respuesta y los <b>coeficientes de detalle</b> capturan fluctuaciones locales "
        "palabra a palabra (candidatas a carga emocional puntual). El ratio entre ambas "
        "energías define el índice θ̂."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Ejecutar pipeline ─────────────────────────────────────────────────────
    if st.button(" Ejecutar análisis NLP + Wavelet", type="primary"):
        barra  = st.progress(0.0)
        status = st.empty()

        def cb(p, msg):
            barra.progress(p)
            status.text(msg)

        try:
            resultado = pipeline_nlp_wavelet(
                textos=[t if t.strip() else "sin respuesta" for t in textos],
                wavelet=wavelet, nivel=nivel,
                modelo_nombre=modelo_emb, progress_callback=cb,
            )
        except Exception as e:
            st.error(f"Error en el pipeline: {e}")
            return

        theta_hat = resultado["theta_hat"]
        wcoefs    = resultado["wcoefs"]

        df_sent = pd.DataFrame({
            "_id":       ids,
            "theta_hat": theta_hat,
            "E_det":     wcoefs["energia_det"],
            "E_apr":     wcoefs["energia_apr"],
        })
        state.set("df_sentimiento", df_sent)
        state.set("wavelet_coefs", wcoefs)
        state.set("embeddings",    resultado["embeddings"])
        state.set("cramer_rao",    calcular_cramer_rao(theta_hat))

        # Nube de palabras
        freqs = calcular_frecuencias(
            [t for t in textos if t.strip()], top_n=80
        )
        state.set("word_frequencies", freqs)

        barra.empty(); status.empty()
        st.success("✔ Análisis completado.")

    if not state.listo_sentimiento():
        return

    df_sent = state.get("df_sentimiento")
    wcoefs  = state.get("wavelet_coefs")
    theta   = df_sent["theta_hat"].values
    freqs   = state.get("word_frequencies") or {}

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 1: NUBE DE PALABRAS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'> Palabras clave del cuestionario</div>",
                unsafe_allow_html=True)

    if freqs:
        top_palabras = list(freqs.items())[:20]

        col_cloud, col_freq = st.columns([3, 2])

        with col_cloud:
            st.markdown("**Nube de palabras**")
            with st.spinner("Generando nube de palabras..."):
                img_pil  = generar_wordcloud_imagen(freqs, width=860, height=440)
                img_bytes = imagen_a_bytes(img_pil)
            st.image(img_bytes, use_container_width=True,
                     caption="Tamaño ∝ frecuencia de aparición en las encuestas")

        with col_freq:
            st.markdown("**Top 15 palabras más frecuentes**")
            df_freq = pd.DataFrame(top_palabras[:15], columns=["Palabra", "Frecuencia"])
            df_freq["Barra"] = df_freq["Frecuencia"] / df_freq["Frecuencia"].max()

            fig_freq = go.Figure(go.Bar(
                x=df_freq["Frecuencia"],
                y=df_freq["Palabra"],
                orientation="h",
                marker=dict(
                    color=df_freq["Barra"].values,
                    colorscale="Blues",
                    showscale=False,
                ),
                text=df_freq["Frecuencia"],
                textposition="outside",
            ))
            fig_freq.update_layout(
                **PLOTLY_LAYOUT,
                height=400,
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis_title="Frecuencia",
            )
            fig_freq.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_freq, use_container_width=True)

        # Interpretación de la nube
        st.markdown(interpretar_wordcloud(top_palabras), unsafe_allow_html=True)
    else:
        st.info("No hay datos de frecuencia disponibles. Ejecuta primero el análisis.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 2: DISTRIBUCIÓN DE SENTIMIENTO
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-title'>Distribución del índice de sentimiento θ̂</div>",
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_hist = px.histogram(
            df_sent, x="theta_hat", nbins=15,
            color_discrete_sequence=["#58a6ff"],
            title="Distribución de θ̂ (índice de sentimiento)",
            labels={"theta_hat": "θ̂ ∈ [-1, 1]"},
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#f85149",
                           annotation_text="Neutro (0)")
        fig_hist.add_vline(x=float(np.mean(theta)), line_dash="dot",
                           line_color="#3fb950",
                           annotation_text=f"Media ({np.mean(theta):.3f})")
        fig_hist.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_e = px.scatter(
            df_sent, x="E_apr", y="E_det",
            color="theta_hat", color_continuous_scale="RdYlGn",
            title="Energía: Aproximación vs Detalle",
            labels={"E_apr": "E_apr (contexto global)",
                    "E_det": "E_det (carga emocional)"},
            hover_data={"_id": True},
        )
        # Línea diagonal de referencia (E_det = E_apr → θ̂ ≈ 0)
        max_e = max(df_sent["E_apr"].max(), df_sent["E_det"].max())
        fig_e.add_trace(go.Scatter(
            x=[0, max_e], y=[0, max_e], mode="lines",
            line=dict(dash="dash", color="#8b949e", width=1),
            name="E_det = E_apr (θ̂=0)",
        ))
        fig_e.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_e, use_container_width=True)

    # Interpretaciones de sentimiento
    st.markdown(interpretar_distribucion_sentimiento(theta), unsafe_allow_html=True)

    media_det = float(df_sent["E_det"].mean())
    media_apr = float(df_sent["E_apr"].mean())
    st.markdown(interpretar_energia_wavelet(media_det, media_apr), unsafe_allow_html=True)

    # ── Heatmap de coeficientes ────────────────────────────────────────────────
    if wcoefs and "detalles" in wcoefs:
        st.divider()
        st.markdown(
            "<div class='section-title'>Coeficientes de onduleta — Nivel 1 "
            "(primeros 30 alumnos)</div>", unsafe_allow_html=True)
        det1 = wcoefs["detalles"][0][:30]
        fig_heat = go.Figure(data=go.Heatmap(
            z=det1, colorscale="RdBu", zmid=0,
            colorbar=dict(title="Coef."),
        ))
        fig_heat.update_layout(
            **PLOTLY_LAYOUT,
            title="Coeficientes de detalle nivel 1 (carga emocional aguda)",
            xaxis_title="Posición en la respuesta (tokens)", yaxis_title="Alumno",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown(
            "<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;"
            "padding:0.7rem 1rem;font-size:0.82rem;color:#8b949e;'>"
            " <b>Lectura del mapa de calor:</b> Cada fila representa un alumno y cada "
            "columna una dimensión del embedding. Los colores <b style='color:#f85149'>rojos</b> "
            "indican coeficientes positivos (activaciones emocionales altas), los "
            "<b style='color:#58a6ff'>azules</b> indican activaciones negativas (ausencia de "
            "esa característica emocional). Filas con alta saturación roja corresponden a "
            "alumnos con mayor carga emocional en sus respuestas.</div>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 3: CORRELACIÓN CON CALIFICACIONES
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("<div class='section-title'>RF-04 · Correlación sentimiento ↔ calificación</div>",
                unsafe_allow_html=True)

    if not state.listo_calificaciones():
        st.info("Carga el archivo de calificaciones para ver la correlación.")
        return

    df_cal     = state.get("df_calificaciones")
    col_id_enc = state.get("col_id_enc")
    col_calif  = ("Calificacion" if "Calificacion" in df_cal.columns
                  else ("TOTAL" if "TOTAL" in df_cal.columns else None))

    df_m = None
    if col_id_enc and col_id_enc != "— ninguna —" and "No-Cuenta" in df_cal.columns:
        df_sent_m = df_sent.rename(columns={"_id": "No-Cuenta"}).copy()
        df_cal_c  = df_cal.copy()
        df_cal_c["No-Cuenta"]   = df_cal_c["No-Cuenta"].astype(str).str.strip()
        df_sent_m["No-Cuenta"]  = df_sent_m["No-Cuenta"].astype(str).str.strip()
        df_m = df_cal_c[["No-Cuenta"] + ([col_calif] if col_calif else [])].merge(
            df_sent_m[["No-Cuenta", "theta_hat"]], on="No-Cuenta", how="inner"
        )
    elif col_calif:
        n    = min(len(df_cal), len(df_sent))
        df_m = df_cal[[col_calif]].iloc[:n].reset_index(drop=True)
        df_m["theta_hat"] = df_sent["theta_hat"].values[:n]

    if df_m is not None and col_calif and len(df_m) > 3:
        df_plot = df_m.dropna(subset=[col_calif, "theta_hat"])
        r, p    = stats.pearsonr(df_plot[col_calif], df_plot["theta_hat"])

        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(
                f"<div class='metric-card'><div class='val'>{round(r,3)}</div>"
                f"<div class='lbl'>Correlación de Pearson (r)</div></div><br>"
                f"<div class='metric-card'><div class='val'>{round(p,4)}</div>"
                f"<div class='lbl'>p-valor</div></div>",
                unsafe_allow_html=True,
            )

        with c2:
            nombre_col = "Nombre" if "Nombre" in df_plot.columns else None
            x_vals = df_plot["theta_hat"].values
            y_vals = df_plot[col_calif].values
            m_reg, b_reg = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

            fig_scatter = px.scatter(
                df_plot, x="theta_hat", y=col_calif,
                color=col_calif, color_continuous_scale="Blues",
                hover_name=nombre_col,
                labels={"theta_hat": "Índice de sentimiento θ̂",
                        col_calif: "Calificación"},
                title=f"Sentimiento vs Calificación (r = {round(r,3)})",
            )
            fig_scatter.add_trace(go.Scatter(
                x=x_line, y=m_reg * x_line + b_reg,
                mode="lines",
                line=dict(color="#f85149", width=2, dash="dash"),
                name=f"Regresión (y={round(m_reg,2)}x+{round(b_reg,2)})",
            ))
            fig_scatter.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Interpretación de la correlación
        st.markdown(interpretar_correlacion(r, p, len(df_plot)), unsafe_allow_html=True)

        # Tabla de alumnos con sentimiento alto + calificación baja (zona de alarma)
        alarma = df_plot[(df_plot["theta_hat"] > 0.3) & (df_plot[col_calif] < 70)]
        if len(alarma) > 0:
            st.markdown("<div class='section-title'> Alumnos en zona de doble riesgo "
                        "(sentimiento alto + calificación baja)</div>",
                        unsafe_allow_html=True)
            st.markdown(
                "<div style='background:#3d1f1f;border-left:4px solid #f85149;"
                "border-radius:4px;padding:0.8rem 1rem;margin-bottom:0.5rem;"
                "color:#ffa198;font-size:0.85rem;'>"
                " Estos alumnos combinan <b>alta carga emocional en sus respuestas</b> "
                "(θ̂ > 0.3) con <b>calificación por debajo del umbral aprobatorio</b> (&lt; 70). "
                "Son los candidatos prioritarios para intervención de tutoría y apoyo psicopedagógico."
                "</div>",
                unsafe_allow_html=True,
            )
            cols_alarm = (["No-Cuenta"] if "No-Cuenta" in alarma.columns else []) + \
                         (["Nombre"] if "Nombre" in alarma.columns else []) + \
                         [col_calif, "theta_hat"]
            cols_alarm = [c for c in cols_alarm if c in alarma.columns]
            st.dataframe(alarma[cols_alarm].rename(columns={"theta_hat": "Sentimiento θ̂"}),
                         use_container_width=True)

        state.set("df_merged", df_m)
    else:
        st.info("No se pudo hacer el cruce de datos. Verifica que los IDs coincidan.")

    # ── Tabla completa de sentimientos ────────────────────────────────────────
    st.divider()
    with st.expander(" Ver tabla completa de índices de sentimiento"):
        st.dataframe(
            df_sent.rename(columns={
                "theta_hat": "Sentimiento θ̂",
                "E_det": "Energía detalle",
                "E_apr": "Energía aproximación",
            }),
            use_container_width=True,
        )