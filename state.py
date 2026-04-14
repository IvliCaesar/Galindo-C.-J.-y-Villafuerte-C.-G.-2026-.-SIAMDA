"""
Gestión centralizada del estado de sesión de Streamlit.
"""
import streamlit as st


def init():
    defaults = {
        "df_calificaciones": None,   # DataFrame principal de calificaciones
        "df_encuesta": None,         # DataFrame de respuestas de encuesta
        "df_sentimiento": None,      # DataFrame con índices de sentimiento + wavelet
        "df_merged": None,           # Merge calificaciones + sentimiento
        "embeddings": None,          # Matriz numpy de embeddings
        "wavelet_coefs": None,       # Dict con coeficientes wavelet
        "cramer_rao": None,          # Dict con info de Fisher y cota
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get(key):
    return st.session_state.get(key)


def set(key, value):
    st.session_state[key] = value


def listo_calificaciones():
    return st.session_state.get("df_calificaciones") is not None


def listo_encuesta():
    return st.session_state.get("df_encuesta") is not None


def listo_sentimiento():
    return st.session_state.get("df_sentimiento") is not None
