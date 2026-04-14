# SIAMDA — Sistema Integrado de Análisis Multiresolución y Desempeño Académico

## Instalación

```bash
# 1. Clonar / descomprimir el proyecto
cd siamda/

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la app
streamlit run app.py
```

La app abre automáticamente en http://localhost:8501

---

## Estructura del proyecto

```
siamda/
├── app.py                      # Punto de entrada Streamlit
├── state.py                    # Gestión centralizada del estado de sesión
├── requirements.txt
├── processing/
│   ├── datos.py                # RF-01: Ingesta y normalización de datos
│   ├── nlp_wavelet.py          # RF-02: Embeddings + DWT
│   └── cramer_rao.py           # RF-03: Información de Fisher y Cota de Cramér-Rao
├── pages/
│   ├── carga.py                # Página: Carga de Datos
│   ├── metricas.py             # Página: Métricas del Curso
│   ├── sentimiento.py          # Página: Análisis de Sentimiento (Wavelets)
│   └── validacion.py           # Página: Validación Teórica
└── datos_ejemplo/
    ├── calificaciones_ejemplo.csv
    └── encuesta_ejemplo.csv
```

---

## Formato esperado de los CSVs

### Calificaciones (puede ser Excel con múltiples hojas)
| No-Cuenta | Nombre | E1 | E2 | E3 | EXAMENES (70%) | TAREAS (20%) | EXPO (10%) | Firmas | EXAMENES VALIDACION | TOTAL | Calificacion |
|-----------|--------|----|----|----|----|----|----|----|----|----|----|

- Las columnas de parciales son detectadas automáticamente (E1, E2, ExmP1, Parcial1…)
- La columna `EXAMENES VALIDACION` puede contener texto como "Te presentas a final" o "Está en reposición"

### Encuesta (CSV)
| No-Cuenta | Nombre | Pregunta1 | Pregunta2 | ... |
|-----------|--------|-----------|-----------|-----|

- El sistema detecta automáticamente columnas de respuesta abierta (texto > 20 chars en promedio)
- Se puede vincular con calificaciones mediante `No-Cuenta`

---

## Flujo de uso

1. **Carga de Datos** → Subir calificaciones y encuesta, confirmar columnas
2. **Métricas del Curso** → Dashboard cuantitativo, alertas de riesgo
3. **Análisis de Sentimiento** → Seleccionar wavelet y nivel, ejecutar pipeline NLP
4. **Validación Teórica** → Revisar Información de Fisher, CRB y eficiencia del estimador

---

## Dependencias principales

| Librería | Uso |
|----------|-----|
| `streamlit` | Framework de la app web |
| `sentence-transformers` | Embeddings multilingüe |
| `PyWavelets` | Transformada Discreta de Onduleta |
| `plotly` | Gráficos interactivos |
| `scipy` | Test de normalidad, estadística |
| `pandas / numpy` | Procesamiento de datos |

> ⚠️ La primera ejecución descarga el modelo de sentence-transformers (~90 MB). Se cachea localmente.
