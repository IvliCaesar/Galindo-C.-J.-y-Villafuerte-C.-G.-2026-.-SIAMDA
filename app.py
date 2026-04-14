import streamlit as st

st.set_page_config(
    page_title="SIAMDA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos globales ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stRadio label { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .val { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #58a6ff; }
.metric-card .lbl { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

.alert-riesgo {
    background: #3d1f1f;
    border-left: 4px solid #f85149;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #ffa198;
}
.alert-ok {
    background: #1a2f1a;
    border-left: 4px solid #3fb950;
    border-radius: 4px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: #7ee787;
}
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  SIAMDA")
    st.markdown("<div class='section-title'>Navegación</div>", unsafe_allow_html=True)
    pagina = st.radio(
        "",
        [" Carga de Datos",
         " Métricas del Curso",
         " Análisis de Sentimiento (Wavelets)",
         " Validación Teórica"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("<span style='font-size:0.72rem;color:#484f58;'>Procesos Estocásticos · SIAMDA v1.0</span>", unsafe_allow_html=True)

# ── Enrutamiento de páginas ───────────────────────────────────────────────────
if pagina == " Carga de Datos":
    from pages import carga
    carga.render()
elif pagina == " Métricas del Curso":
    from pages import metricas
    metricas.render()
elif pagina == " Análisis de Sentimiento (Wavelets)":
    from pages import sentimiento
    sentimiento.render()
elif pagina == " Validación Teórica":
    from pages import validacion
    validacion.render()
