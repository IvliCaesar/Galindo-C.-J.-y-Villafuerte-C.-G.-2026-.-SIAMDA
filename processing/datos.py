"""
Módulo de ingesta y procesamiento de datos académicos.
RF-01: Lectura y estructuración de columnas con manejo de texto mixto.
"""
import pandas as pd
import numpy as np
import re


# ── Constantes de columnas esperadas ──────────────────────────────────────────
COLS_ID = ["No-Cuenta", "Nombre"]

COLS_EXAMENES = ["E1", "E2", "E3"]          # parciales detectados dinámicamente
COL_EXAM_POND = "EXAMENES (70%)"
COL_TAREAS    = "TAREAS (20%)"
COL_EXPO      = "EXPO (10%)"
COL_FIRMA     = "Firmas"
COL_VALIDACION = "EXAMENES VALIDACION"
COL_TOTAL     = "TOTAL"
COL_CALIF     = "Calificacion"

# Texto mixto que indica estado especial (no numérico)
TEXTOS_ESPECIALES = [
    "te presentas a final",
    "está en reposición",
    "esta en reposicion",
    "reposicion",
    "final",
    "baja",
    "no presentó",
    "no presento",
]


def _es_especial(val) -> bool:
    if pd.isna(val):
        return False
    return isinstance(val, str) and any(t in str(val).lower() for t in TEXTOS_ESPECIALES)


def _to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convierte a numérico ignorando texto mixto (devuelve NaN para texto)."""
    def _conv(v):
        if _es_especial(v):
            return np.nan
        try:
            return float(str(v).replace(",", "."))
        except (ValueError, TypeError):
            return np.nan
    return series.map(_conv)


def leer_calificaciones(file) -> dict[str, pd.DataFrame]:
    """
    Lee el archivo Excel/CSV de calificaciones.
    Retorna dict {nombre_hoja: DataFrame} para que el usuario seleccione.
    """
    nombre = file.name.lower()
    if nombre.endswith(".csv"):
        df = pd.read_csv(file, encoding="utf-8-sig")
        return {"Hoja1": df}
    else:
        xl = pd.ExcelFile(file)
        return {hoja: xl.parse(hoja) for hoja in xl.sheet_names}


def leer_encuesta(file) -> pd.DataFrame:
    """Lee el CSV/Excel de encuesta."""
    nombre = file.name.lower()
    if nombre.endswith(".csv"):
        return pd.read_csv(file, encoding="utf-8-sig")
    else:
        return pd.read_excel(file)


def detectar_columnas(df: pd.DataFrame) -> dict:
    """
    Detecta automáticamente qué columnas del DataFrame corresponden
    a cada categoría esperada.
    """
    cols = {c.strip(): c for c in df.columns}
    cols_lower = {c.strip().lower(): c for c in df.columns}

    def find(posibles):
        for p in posibles:
            if p in cols:
                return cols[p]
            if p.lower() in cols_lower:
                return cols_lower[p.lower()]
        return None

    # Buscar parciales dinámicamente (E1, E2, ExmP1, Parcial1, etc.)
    parciales = []
    for c in df.columns:
        if re.match(r"^(e\d+|exmp?\d+|parcial\s*\d+|p\d+)$", c.strip().lower()):
            parciales.append(c)

    return {
        "no_cuenta":   find(["No-Cuenta", "NoCuenta", "Cuenta", "ID", "Matricula"]),
        "nombre":      find(["Nombre", "Alumno", "Estudiante", "Name"]),
        "parciales":   parciales,
        "exam_pond":   find(["EXAMENES (70%)", "EXAMENES(70%)", "Examenes", "EXAMENES"]),
        "tareas":      find(["TAREAS (20%)", "TAREAS(20%)", "Tareas", "TAREAS"]),
        "expo":        find(["EXPO (10%)", "EXPO(10%)", "Expo", "EXPO", "Exposicion"]),
        "firmas":      find(["Firmas", "Firma", "FIRMAS"]),
        "validacion":  find(["EXAMENES VALIDACION", "Validacion", "VALIDACION"]),
        "total":       find(["TOTAL", "Total", "Puntaje"]),
        "calificacion":find(["Calificacion", "Calificación", "CALIFICACION", "Calif"]),
    }


def procesar_calificaciones(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    """
    Normaliza el DataFrame de calificaciones:
    - Renombra columnas detectadas a nombres canónicos
    - Convierte valores a numérico (maneja texto mixto)
    - Agrega columna 'estado_especial'
    """
    rename = {}
    if mapa["no_cuenta"]:  rename[mapa["no_cuenta"]]  = "No-Cuenta"
    if mapa["nombre"]:     rename[mapa["nombre"]]      = "Nombre"
    if mapa["exam_pond"]:  rename[mapa["exam_pond"]]   = "EXAMENES_POND"
    if mapa["tareas"]:     rename[mapa["tareas"]]      = "TAREAS_POND"
    if mapa["expo"]:       rename[mapa["expo"]]        = "EXPO_POND"
    if mapa["firmas"]:     rename[mapa["firmas"]]      = "Firmas"
    if mapa["validacion"]: rename[mapa["validacion"]]  = "Validacion"
    if mapa["total"]:      rename[mapa["total"]]       = "TOTAL"
    if mapa["calificacion"]:rename[mapa["calificacion"]]= "Calificacion"
    for i, p in enumerate(mapa["parciales"]):
        rename[p] = f"Parcial_{i+1}"

    out = df.rename(columns=rename).copy()

    # Normalizar No-Cuenta a string para evitar conflictos de tipo en merges
    if "No-Cuenta" in out.columns:
        out["No-Cuenta"] = out["No-Cuenta"].astype(str).str.strip()

    # Convertir numéricas
    num_cols = ["EXAMENES_POND", "TAREAS_POND", "EXPO_POND", "Firmas", "TOTAL", "Calificacion"]
    num_cols += [f"Parcial_{i+1}" for i in range(len(mapa["parciales"]))]
    for col in num_cols:
        if col in out.columns:
            out[col] = _to_numeric_safe(out[col])

    # Estado especial desde validación
    if "Validacion" in out.columns:
        out["estado_especial"] = out["Validacion"].apply(
            lambda v: str(v) if _es_especial(v) else "Normal"
        )
    else:
        out["estado_especial"] = "Normal"

    # Calcular TOTAL si no existe o recalcular
    componentes = ["EXAMENES_POND", "TAREAS_POND", "EXPO_POND"]
    componentes_presentes = [c for c in componentes if c in out.columns]
    if componentes_presentes:
        out["TOTAL_CALC"] = out[componentes_presentes].sum(axis=1, skipna=False)

    # Flag de riesgo: calificación < 70 o estado especial
    if "Calificacion" in out.columns:
        out["en_riesgo"] = (out["Calificacion"] < 70) | (out["estado_especial"] != "Normal")
    elif "TOTAL" in out.columns:
        out["en_riesgo"] = (out["TOTAL"] < 70) | (out["estado_especial"] != "Normal")
    else:
        out["en_riesgo"] = False

    return out


def detectar_col_texto(df: pd.DataFrame) -> list[str]:
    """Detecta columnas de texto (respuestas abiertas) en el DataFrame de encuesta."""
    candidatas = []
    for col in df.columns:
        # pd.api.types.is_string_dtype cubre tanto el dtype 'object' clásico
        # como el dtype de string dedicado que pandas >= 2.x/3.x puede inferir
        # por defecto (pd.options.future.infer_string); comparar contra
        # `== object` deja de detectar cualquier columna de texto en ese caso.
        if pd.api.types.is_string_dtype(df[col]):
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len and avg_len > 20:   # respuestas con más de 20 chars en promedio
                candidatas.append(col)
    return candidatas
