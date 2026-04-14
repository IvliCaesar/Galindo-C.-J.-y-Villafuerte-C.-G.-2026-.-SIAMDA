"""
Página: Carga de Datos
"""
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import state
from processing.datos import (
    leer_calificaciones, leer_encuesta,
    detectar_columnas, procesar_calificaciones,
    detectar_col_texto,
)


def render():
    state.init()
    st.title(" Carga de Datos")
    st.markdown("<div class='section-title'>Ingesta de archivos académicos</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── CSV Calificaciones ─────────────────────────────────────────────────────
    with col1:
        st.subheader(" Calificaciones")
        st.caption("Excel/CSV con columnas: E1, E2, Tareas, Firmas, Calificación")
        archivo_calif = st.file_uploader(
            "Subir archivo de calificaciones",
            type=["csv", "xlsx", "xls"],
            key="uploader_calif",
        )

        if archivo_calif:
            try:
                hojas = leer_calificaciones(archivo_calif)
                nombres_hojas = list(hojas.keys())

                if len(nombres_hojas) > 1:
                    hoja_sel = st.selectbox("Selecciona la hoja a usar:", nombres_hojas)
                else:
                    hoja_sel = nombres_hojas[0]

                df_raw = hojas[hoja_sel]
                st.markdown(f"**{len(df_raw)} registros · {len(df_raw.columns)} columnas detectadas**")

                with st.expander("Vista previa (primeras 5 filas)"):
                    st.dataframe(df_raw.head(), use_container_width=True)

                mapa = detectar_columnas(df_raw)

                with st.expander(" Mapeo de columnas detectado"):
                    for k, v in mapa.items():
                        if k == "parciales":
                            st.write(f"• **Parciales**: {v}")
                        elif v:
                            st.write(f"• **{k}** → `{v}`")
                        else:
                            st.write(f"• **{k}** →  no encontrada")

                if st.button(" Confirmar y procesar calificaciones", type="primary"):
                    df_proc = procesar_calificaciones(df_raw, mapa)
                    state.set("df_calificaciones", df_proc)
                    st.success(f" {len(df_proc)} alumnos cargados correctamente.")

                    # Mostrar alertas de riesgo
                    en_riesgo = df_proc[df_proc.get("en_riesgo", pd.Series(False, index=df_proc.index))]
                    if len(en_riesgo) > 0:
                        st.warning(f" {len(en_riesgo)} alumnos en situación de riesgo detectados.")

            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

    # ── CSV Encuesta ───────────────────────────────────────────────────────────
    with col2:
        st.subheader(" Encuesta")
        st.caption("CSV con respuestas abiertas de los alumnos")
        archivo_enc = st.file_uploader(
            "Subir archivo de encuesta",
            type=["csv", "xlsx", "xls"],
            key="uploader_enc",
        )

        if archivo_enc:
            try:
                df_enc = leer_encuesta(archivo_enc)
                st.markdown(f"**{len(df_enc)} respuestas · {len(df_enc.columns)} columnas**")

                with st.expander("Vista previa"):
                    st.dataframe(df_enc.head(), use_container_width=True)

                cols_texto = detectar_col_texto(df_enc)
                if not cols_texto:
                    st.warning("No se detectaron columnas de texto largo. Selecciona manualmente:")
                    cols_texto = list(df_enc.columns)

                col_texto_sel = st.multiselect(
                    "Columnas de respuesta abierta a analizar:",
                    options=list(df_enc.columns),
                    default=cols_texto[:3],
                )

                col_id_enc = st.selectbox(
                    "Columna identificadora (No-Cuenta / Nombre):",
                    options=["— ninguna —"] + list(df_enc.columns),
                )

                if st.button(" Confirmar encuesta", type="primary"):
                    df_enc_proc = df_enc.copy()
                    df_enc_proc["_texto_concat"] = (
                        df_enc[col_texto_sel]
                        .fillna("")
                        .astype(str)
                        .agg(" ".join, axis=1)
                        .str.strip()
                    )
                    if col_id_enc != "— ninguna —":
                        df_enc_proc["_id"] = df_enc[col_id_enc].astype(str)
                    else:
                        df_enc_proc["_id"] = [f"alumno_{i}" for i in range(len(df_enc_proc))]

                    state.set("df_encuesta", df_enc_proc)
                    state.set("col_texto", col_texto_sel)
                    state.set("col_id_enc", col_id_enc)
                    st.success(f"✔ Encuesta cargada. {len(df_enc_proc)} respuestas listas.")

            except Exception as e:
                st.error(f"Error al leer el archivo de encuesta: {e}")

    # ── Estado del sistema ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("<div class='section-title'>Estado del sistema</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ok = state.listo_calificaciones()
        st.markdown(
            f"<div class='alert-ok'>✔ Calificaciones cargadas</div>" if ok
            else "<div class='alert-riesgo'>✗ Calificaciones no cargadas</div>",
            unsafe_allow_html=True,
        )
    with c2:
        ok = state.listo_encuesta()
        st.markdown(
            f"<div class='alert-ok'>✔ Encuesta cargada</div>" if ok
            else "<div class='alert-riesgo'>✗ Encuesta no cargada</div>",
            unsafe_allow_html=True,
        )
    with c3:
        ok = state.listo_sentimiento()
        st.markdown(
            f"<div class='alert-ok'>✔ Sentimiento calculado</div>" if ok
            else "<div class='alert-riesgo'>✗ Sentimiento pendiente</div>",
            unsafe_allow_html=True,
        )
