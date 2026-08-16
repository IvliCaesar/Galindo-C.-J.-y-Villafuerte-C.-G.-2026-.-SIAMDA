"""
Pruebas de processing/datos.py, en particular la bandera de riesgo académico
formalizada en el artículo (galindo_villafuerte.tex, §3.1):
r_i = 1{c_i < 70} OR 1{e_i != "Normal"}.
"""
import numpy as np
import pandas as pd
import pytest

from processing.datos import (
    _es_especial,
    _to_numeric_safe,
    detectar_columnas,
    procesar_calificaciones,
    detectar_col_texto,
)


def test_es_especial_detecta_textos_conocidos():
    assert _es_especial("Te presentas a Final")
    assert _es_especial("está en reposición")
    assert _es_especial("BAJA")
    assert not _es_especial("85")
    assert not _es_especial(np.nan)


def test_to_numeric_safe_convierte_y_deja_nan_en_texto_especial():
    serie = pd.Series(["85", "70.5", "te presentas a final", "60,5", None])
    out = _to_numeric_safe(serie)
    assert out.iloc[0] == 85.0
    assert out.iloc[1] == 70.5
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] == 60.5  # coma decimal
    assert pd.isna(out.iloc[4])


def test_detectar_columnas_encuentra_parciales_dinamicamente():
    df = pd.DataFrame(columns=[
        "No-Cuenta", "Nombre", "E1", "E2", "E3",
        "EXAMENES (70%)", "TAREAS (20%)", "EXPO (10%)",
        "Firmas", "EXAMENES VALIDACION", "TOTAL", "Calificacion",
    ])
    mapa = detectar_columnas(df)
    assert mapa["no_cuenta"] == "No-Cuenta"
    assert mapa["nombre"] == "Nombre"
    assert set(mapa["parciales"]) == {"E1", "E2", "E3"}
    assert mapa["calificacion"] == "Calificacion"


@pytest.mark.parametrize(
    "calificacion,estado,en_riesgo_esperado",
    [
        (85, "Normal", False),   # aprueba, sin estado especial
        (69.9, "Normal", True),  # reprueba por calificación
        (95, "Está en reposición", True),  # aprueba pero con estado especial
        (50, "Baja", True),      # ambas condiciones
        (70, "Normal", False),   # frontera: 70 no es < 70
    ],
)
def test_bandera_de_riesgo_r_i_del_articulo(calificacion, estado, en_riesgo_esperado):
    df = pd.DataFrame({
        "No-Cuenta": ["1"],
        "Nombre": ["Alumno"],
        "Calificacion": [calificacion],
        "EXAMENES VALIDACION": [estado],
    })
    mapa = detectar_columnas(df)
    out = procesar_calificaciones(df, mapa)
    assert bool(out["en_riesgo"].iloc[0]) == en_riesgo_esperado


def test_detectar_col_texto_ignora_columnas_cortas():
    df = pd.DataFrame({
        "id": ["1", "2"],
        "comentario": [
            "Esta es una respuesta larga con más de veinte caracteres de verdad.",
            "Otra respuesta abierta igual de larga para pasar el umbral fijado.",
        ],
    })
    candidatas = detectar_col_texto(df)
    assert "comentario" in candidatas
    assert "id" not in candidatas
