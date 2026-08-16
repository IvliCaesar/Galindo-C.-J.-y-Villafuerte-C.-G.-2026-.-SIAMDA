"""
Página: Validación Teórica — Información de Fisher y Cota de Cramér-Rao.
RF-03: Cov(θ̂) ≥ 1/F(θ)  + interpretaciones automáticas completas.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import state
from processing.interpretacion import interpretar_crb

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
)


def render():
    state.init()
    st.title(" Validación Teórica")
    st.markdown("<div class='section-title'>Información de Fisher y Cota de Cramér-Rao</div>",
                unsafe_allow_html=True)

    if not state.listo_sentimiento():
        st.warning(" Primero ejecuta el Análisis de Sentimiento (Wavelets).")
        return

    cr = state.get("cramer_rao")
    if cr is None:
        st.warning(" No se encontraron datos de Cramér-Rao. Re-ejecuta el análisis.")
        return

    g  = cr["gaussiano"]
    lp = cr["laplaciano"]
    theta_hat = cr["theta_hat"]

    # ── Marco teórico ──────────────────────────────────────────────────────────
    st.markdown("###  Marco Teórico")

    with st.expander("Ver fundamento matemático completo", expanded=False):
        st.markdown(r"""
**¿Qué es la Cota de Cramér-Rao?**

La Cota de Cramér-Rao establece el **límite teórico inferior** de la varianza
de cualquier estimador estadístico insesgado. En términos simples: ningún estimador puede
ser más preciso que este límite, independientemente del método utilizado.

$$\text{Var}(\hat{\theta}) \geq \frac{1}{F(\theta)} \quad \Longleftrightarrow \quad \text{Cov}(\hat{\theta}) \geq \mathbf{F}^{-1}(\theta)$$

**Información de Fisher** $F(\theta)$: mide cuánta información contiene la muestra sobre el parámetro $\theta$.

$$F(\theta) = -\mathbb{E}\!\left[\frac{\partial^2}{\partial \theta^2} \ln p(X;\theta)\right]$$

**Bajo modelo Gaussiano** $\theta \sim \mathcal{N}(\mu, \sigma^2)$:

$$F(\mu) = \frac{N}{\sigma^2}, \qquad \text{CRB} = \frac{\sigma^2}{N}$$

Un estimador es **eficiente** si alcanza la cota exactamente. El estimador de media muestral
$\bar{\theta} = \frac{1}{N}\sum_i \hat{\theta}_i$ es **siempre** eficiente bajo el modelo
gaussiano con $\sigma^2$ conocida (Lehmann-Scheffé) — esto no depende de qué tan bien el
pipeline NLP-Wavelet capture sentimiento real, es una propiedad de la media muestral como
estimador de sí misma.

**Relevancia para SIAMDA:** esta comparación certifica que la media reportada $\bar\theta$
y su intervalo de confianza son estadísticamente coherentes bajo el modelo gaussiano
(bootstrap ≈ cota teórica). **No** certifica que el pipeline NLP-Wavelet extraiga la máxima
información posible del texto de las encuestas — verificar eso exigiría comparar $\hat\theta$
contra una medida de sentimiento alternativa (léxica o de anotación humana) sobre el mismo
texto, algo que este módulo no hace.
        """)

    st.divider()

    # ── KPIs modelo gaussiano ─────────────────────────────────────────────────
    st.markdown("### Modelo Gaussiano")

    for col, lbl, val in zip(
        st.columns(4),
        ["N (muestras)", "μ̂ (media θ̂)", "σ̂ (desv. est.)", "Var(θ̂) empírica"],
        [g["N"], round(g["mu_hat"],4), round(g["sigma_hat"],4), round(g["var_empirica_mu"],6)],
    ):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='val'>{val}</div>"
                f"<div class='lbl'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    efic_pct = round(g["eficiencia"] * 100, 1)
    color_ef  = "#3fb950" if g["eficiencia"] > 0.85 else ("#d29922" if g["eficiencia"] > 0.6 else "#f85149")

    for col, lbl, val, extra in zip(
        st.columns(3),
        ["Información de Fisher F(μ)", "Cota Cramér-Rao (CRB)", "Eficiencia del estimador"],
        [round(g["F_mu"],4), round(g["crb_mu"],6), f"{efic_pct}%"],
        ["", "", color_ef],
    ):
        with col:
            color_str = f"style='color:{extra}'" if extra else ""
            st.markdown(
                f"<div class='metric-card'><div class='val' {color_str}>{val}</div>"
                f"<div class='lbl'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )

    # Interpretación principal de la CRB
    st.write("")
    st.markdown(
        interpretar_crb(g["eficiencia"], g["shapiro_p"], g["N"]),
        unsafe_allow_html=True,
    )

    # ── Modelo Laplaciano ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Modelo Laplaciano (distribuciones de cola pesada)")
    st.markdown(
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;"
        "padding:0.7rem 1rem;font-size:0.83rem;color:#8b949e;margin-bottom:0.6rem;'>"
        " <b>¿Cuándo usar el modelo Laplaciano?</b> Si el test Shapiro-Wilk rechaza "
        "la normalidad (p &lt; 0.05) o si la distribución de θ̂ muestra colas pesadas "
        "(alumnos muy extremos), el modelo Laplaciano es más robusto. Su estimador óptimo "
        "es la <b>mediana muestral</b> en lugar de la media, y la información de Fisher "
        "es F(μ) = N/b², donde b es la escala de la distribución."
        "</div>",
        unsafe_allow_html=True,
    )

    for col, lbl, val in zip(
        st.columns(3),
        ["Mediana (μ̂ robusto)", "Escala b̂", "CRB Laplaciano"],
        [round(lp["mu_hat"],4), round(lp["b_hat"],4), round(lp["crb_mu"],6)],
    ):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='val'>{val}</div>"
                f"<div class='lbl'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )

    # ── Gráficas ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Visualizaciones diagnósticas")

    col_a, col_b = st.columns(2)

    with col_a:
        boots = g["boots_mu"]
        fig_b = go.Figure()
        fig_b.add_trace(go.Histogram(
            x=boots, nbinsx=30, marker_color="#58a6ff", opacity=0.8, name="Bootstrap μ̂",
        ))
        fig_b.add_vline(x=g["mu_hat"], line_color="#3fb950", line_dash="dash",
                        annotation_text="μ̂")
        fig_b.add_vline(x=g["mu_hat"] - np.sqrt(g["crb_mu"]), line_color="#f85149",
                        line_dash="dot", annotation_text="-√CRB")
        fig_b.add_vline(x=g["mu_hat"] + np.sqrt(g["crb_mu"]), line_color="#f85149",
                        line_dash="dot", annotation_text="+√CRB")
        fig_b.update_layout(**PLOTLY_LAYOUT, title="Distribución Bootstrap del estimador μ̂",
                            xaxis_title="μ̂", yaxis_title="Frecuencia")
        st.plotly_chart(fig_b, use_container_width=True)
        st.markdown(
            "<div style='font-size:0.8rem;color:#8b949e;padding:0.4rem 0;'>"
            " El histograma muestra la distribución de 500 estimaciones bootstrap de la media. "
            "Las líneas rojas punteadas marcan ±√CRB (la desviación estándar mínima teórica). "
            "Si la distribución bootstrap cabe dentro de esas líneas, el estimador es eficiente."
            "</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        categorias = ["Var(θ̂) empírica", "CRB Gaussiano", "CRB Laplaciano"]
        valores    = [g["var_empirica_mu"], g["crb_mu"], lp["crb_mu"]]
        colores    = ["#58a6ff", "#3fb950", "#d29922"]
        fig_comp   = go.Figure(go.Bar(
            x=categorias, y=valores, marker_color=colores,
            text=[f"{v:.2e}" for v in valores], textposition="outside",
        ))
        fig_comp.update_layout(**PLOTLY_LAYOUT,
                               title="Var(θ̂) empírica vs Cotas teóricas de Cramér-Rao",
                               yaxis_title="Varianza")
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown(
            "<div style='font-size:0.8rem;color:#8b949e;padding:0.4rem 0;'>"
            " La barra azul debe ser lo más parecida posible a la barra verde (CRB gaussiano). "
            "Si la barra azul es mucho mayor, el estimador tiene margen de mejora. "
            "Si fuera menor, indicaría un estimador sesgado (imposible para estimadores insesgados)."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Q-Q Plot ──────────────────────────────────────────────────────────────
    from scipy import stats as scipy_stats
    (osm, osr), (slope, intercept, r_qq) = scipy_stats.probplot(theta_hat, dist="norm")
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=list(osm), y=list(osr), mode="markers",
        marker=dict(color="#58a6ff", size=5), name="θ̂",
    ))
    fig_qq.add_trace(go.Scatter(
        x=[min(osm), max(osm)],
        y=[slope * min(osm) + intercept, slope * max(osm) + intercept],
        mode="lines", line=dict(color="#f85149", dash="dash"), name="Normal ref.",
    ))
    fig_qq.update_layout(**PLOTLY_LAYOUT, title="Q-Q Plot de θ̂ vs Distribución Normal",
                         xaxis_title="Cuantiles teóricos", yaxis_title="Cuantiles muestrales")
    st.plotly_chart(fig_qq, use_container_width=True)
    st.markdown(
        "<div style='font-size:0.8rem;color:#8b949e;padding:0.4rem 0;'>"
        " <b>Lectura del Q-Q Plot:</b> Si los puntos se alinean sobre la línea roja diagonal, "
        "θ̂ sigue una distribución normal (supuesto gaussiano válido). "
        "Desviaciones en las colas indican distribución de colas pesadas → considerar modelo Laplaciano. "
        f"Coeficiente de determinación del ajuste: R² = {round(r_qq**2, 4)}."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Tabla resumen exportable ───────────────────────────────────────────────
    st.divider()
    st.markdown("###  Tabla resumen de métricas teóricas")
    tabla = pd.DataFrame({
        "Parámetro": ["N", "μ̂ (media sentimiento)", "σ̂ (desv. estándar)",
                      "Var(θ̂) empírica (bootstrap)", "F(μ) — Información de Fisher",
                      "CRB Gaussiano (1/F)", "Eficiencia del estimador (EES)",
                      "Shapiro-Wilk p-valor", "F(μ) Laplaciano", "CRB Laplaciano"],
        "Valor": [
            g["N"], f"{g['mu_hat']:.6f}", f"{g['sigma_hat']:.6f}",
            f"{g['var_empirica_mu']:.2e}", f"{g['F_mu']:.4f}",
            f"{g['crb_mu']:.2e}", f"{g['eficiencia']*100:.2f}%",
            f"{g['shapiro_p']:.4f}", f"{lp['F_mu']:.4f}", f"{lp['crb_mu']:.2e}",
        ],
        "Interpretación": [
            "Tamaño de la muestra analizada",
            "Sentimiento promedio del grupo (−1 = neutro, +1 = alta carga)",
            "Dispersión del sentimiento entre alumnos",
            "Varianza real del estimador (500 resamples)",
            "Información total del cuestionario sobre el parámetro",
            "Mínimo teórico de la varianza — nadie puede superar esto",
            "Qué tan cerca está el estimador de su límite óptimo",
            "> 0.05 = normalidad aceptada, < 0.05 = usar modelo Laplaciano",
            "Información Fisher bajo distribución de colas pesadas",
            "CRB bajo supuesto Laplaciano (más conservador)",
        ],
    })
    st.dataframe(tabla, use_container_width=True, height=380)
