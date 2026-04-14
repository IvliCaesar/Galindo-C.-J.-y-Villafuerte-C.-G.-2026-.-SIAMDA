"""
Página: Métricas del Curso
RF-01 / RF-04: Dashboard cuantitativo + interpretaciones automáticas.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import state
from processing.interpretacion import (
    interpretar_promedio, interpretar_riesgo, interpretar_parciales
)

COLORS = {"azul": "#58a6ff", "verde": "#3fb950", "rojo": "#f85149",
          "naranja": "#d29922"}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
)


def _metric_card(label, valor, unidad=""):
    return (f"<div class='metric-card'><div class='val'>{valor}{unidad}</div>"
            f"<div class='lbl'>{label}</div></div>")


def render():
    state.init()
    st.title(" Métricas del Curso")

    if not state.listo_calificaciones():
        st.warning(" Primero carga el archivo de calificaciones en **Carga de Datos**.")
        return

    df = state.get("df_calificaciones")

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Resumen general</div>", unsafe_allow_html=True)

    n_total    = len(df)
    n_riesgo   = int(df["en_riesgo"].sum()) if "en_riesgo" in df.columns else 0
    n_especial = int((df["estado_especial"] != "Normal").sum()) if "estado_especial" in df.columns else 0
    col_calif  = ("Calificacion" if "Calificacion" in df.columns
                  else ("TOTAL" if "TOTAL" in df.columns else None))
    prom       = round(df[col_calif].mean(), 1) if col_calif else None
    aprobados  = int((df[col_calif] >= 70).sum()) if col_calif else None
    pct_aprobados = f"{round(aprobados/n_total*100,1)}%" if isinstance(aprobados, int) else "—"

    for col, lbl, val in zip(
        st.columns(5),
        ["Alumnos", "Promedio", "Aprobados", "En riesgo", "Est. especial"],
        [n_total, prom if prom else "—", pct_aprobados, n_riesgo, n_especial],
    ):
        with col:
            st.markdown(_metric_card(lbl, val), unsafe_allow_html=True)

    # Interpretaciones automáticas
    st.write("")
    if prom is not None:
        st.markdown(interpretar_promedio(prom), unsafe_allow_html=True)
    st.markdown(interpretar_riesgo(n_riesgo, n_total), unsafe_allow_html=True)

    st.divider()

    # ── Distribución de calificaciones ────────────────────────────────────────
    if col_calif:
        st.markdown("<div class='section-title'>Distribución de calificaciones</div>",
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            fig_hist = px.histogram(df, x=col_calif, nbins=20,
                                    color_discrete_sequence=[COLORS["azul"]],
                                    title="Histograma de calificaciones")
            fig_hist.add_vline(x=70, line_dash="dash", line_color=COLORS["rojo"],
                               annotation_text="Min. aprobatorio (70)")
            fig_hist.update_layout(**PLOTLY_TEMPLATE)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            bins   = [0, 60, 70, 80, 90, 101]
            labels = ["< 60", "60-69", "70-79", "80-89", "≥ 90"]
            df["rango"] = pd.cut(df[col_calif], bins=bins, labels=labels, right=False)
            conteo = df["rango"].value_counts().reindex(labels).reset_index()
            conteo.columns = ["Rango", "Alumnos"]
            fig_bar = px.bar(conteo, x="Rango", y="Alumnos", color="Rango",
                             color_discrete_sequence=[COLORS["rojo"], COLORS["naranja"],
                                                      COLORS["azul"], COLORS["verde"], "#8b949e"],
                             title="Alumnos por rango de calificación")
            fig_bar.update_layout(**PLOTLY_TEMPLATE, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        reprobados_n = int((df[col_calif] < 70).sum())
        excelentes_n = int((df[col_calif] >= 90).sum())
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:6px;"
            f"padding:0.8rem 1rem;font-size:0.85rem;color:#8b949e;'>"
            f" <b>Lectura del histograma:</b> De {n_total} alumnos, "
            f"<b style='color:#f85149'>{reprobados_n} reprueban</b> (calificación &lt; 70) y "
            f"<b style='color:#3fb950'>{excelentes_n} tienen excelencia</b> (≥ 90). "
            f"La forma de la distribución indica si el nivel de dificultad del curso "
            f"está bien calibrado para el grupo.</div>",
            unsafe_allow_html=True,
        )

    # ── Parciales ─────────────────────────────────────────────────────────────
    parciales = [c for c in df.columns if c.startswith("Parcial_")]
    if parciales:
        st.divider()
        st.markdown("<div class='section-title'>Rendimiento por parcial</div>",
                    unsafe_allow_html=True)
        prom_parc = df[parciales].dropna(how="all").mean()
        fig_line  = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=prom_parc.index.tolist(), y=prom_parc.values.tolist(),
            mode="lines+markers",
            line=dict(color=COLORS["azul"], width=2), marker=dict(size=8),
            name="Promedio grupal",
        ))
        fig_line.add_hline(y=70, line_dash="dash", line_color=COLORS["rojo"],
                           annotation_text="70 pts")
        fig_line.update_layout(**PLOTLY_TEMPLATE, title="Promedio grupal por parcial")
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown(interpretar_parciales({p: float(prom_parc[p]) for p in parciales}),
                    unsafe_allow_html=True)

    # ── Alumnos en riesgo ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("<div class='section-title'>Alumnos en riesgo / situación especial</div>",
                unsafe_allow_html=True)

    df_riesgo = df[df.get("en_riesgo", pd.Series(False, index=df.index)) == True].copy()
    if len(df_riesgo) > 0:
        cols_m = ([c for c in ["No-Cuenta", "Nombre"] if c in df_riesgo.columns]
                  + ([col_calif] if col_calif else [])
                  + (["estado_especial"] if "estado_especial" in df_riesgo.columns else []))
        for _, row in df_riesgo[cols_m].iterrows():
            nombre = row.get("Nombre", row.get("No-Cuenta", "—"))
            calif  = row.get(col_calif, "—") if col_calif else "—"
            estado = row.get("estado_especial", "")
            tag    = f" · {estado}" if estado and estado != "Normal" else ""
            emoji  = "🔴" if isinstance(calif, float) and calif < 60 else "🟡"
            accion = ("Tutoría personalizada urgente."
                      if isinstance(calif, float) and calif < 60
                      else "Monitoreo continuo y apoyo en temas débiles.")
            st.markdown(
                f"<div class='alert-riesgo'>{emoji} <b>{nombre}</b> "
                f"— Calificación: <b>{calif}</b>{tag} · {accion}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<div class='alert-ok'>✔ Sin alumnos en riesgo detectados.</div>",
                    unsafe_allow_html=True)

    st.divider()
    with st.expander(" Ver tabla completa de alumnos"):
        st.dataframe(df, use_container_width=True, height=400)
