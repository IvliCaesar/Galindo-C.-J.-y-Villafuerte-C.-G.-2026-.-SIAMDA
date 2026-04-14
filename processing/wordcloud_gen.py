"""
Módulo de análisis de texto: nube de palabras y frecuencias.
Genera imagen PIL de wordcloud y estadísticas de texto.
"""
import re
import io
from collections import Counter

import numpy as np
from PIL import Image

# ── Stopwords en español (lista compacta) ─────────────────────────────────────
STOPWORDS_ES = {
    "a","al","ante","bajo","con","contra","de","del","desde","durante","e",
    "el","ella","ellas","ellos","en","entre","es","esa","esas","ese","esos",
    "esta","estas","este","estos","fue","han","has","hay","him","hizo","i",
    "its","la","las","le","les","lo","los","me","mi","mis","más","muy",
    "me","mi","mis","no","nos","o","on","para","pero","por","que","quien",
    "se","si","sin","so","son","su","sus","también","tan","te","the","to",
    "todo","todos","tu","un","una","unas","uno","unos","y","ya","yo",
    "como","cuando","aunque","sobre","esto","eso","aquí","allí","qué",
    "cómo","cuál","cuáles","dónde","quién","era","ser","fue","han","sido",
    "así","hay","bien","mal","más","menos","mucho","poco","algo","nada",
    "cada","otro","otra","otros","otras","mismo","misma","puede","pueden",
    "porque","pues","entonces","luego","después","antes","ahora","siempre",
    "nunca","solo","sólo","además","sin","hasta","hacia","entre","durante",
    "creo","siento","hacer","haber","tener","ir","dar","ver","saber",
}


def _limpiar_texto(textos: list[str]) -> list[str]:
    """Tokeniza y limpia lista de textos."""
    tokens = []
    for t in textos:
        t = t.lower()
        t = re.sub(r"[^a-záéíóúüñ\s]", " ", t)
        palabras = t.split()
        tokens.extend([p for p in palabras
                       if len(p) > 3 and p not in STOPWORDS_ES])
    return tokens


def calcular_frecuencias(textos: list[str], top_n: int = 60) -> dict[str, int]:
    """Retorna las top_n palabras más frecuentes."""
    tokens = _limpiar_texto(textos)
    return dict(Counter(tokens).most_common(top_n))


def generar_wordcloud_imagen(
    frecuencias: dict[str, int],
    width: int = 900,
    height: int = 480,
    bg_color: str = "#0d1117",
    paleta: list[str] | None = None,
) -> Image.Image:
    """
    Genera una nube de palabras como imagen PIL sin depender de la librería
    `wordcloud` (que requiere freetype nativa).

    Estrategia:
      1. Ordena palabras por frecuencia.
      2. Asigna tamaños proporcionales a la frecuencia (log-escalado).
      3. Coloca las palabras en espiral desde el centro usando PIL.ImageDraw.
    """
    from PIL import ImageDraw, ImageFont

    if paleta is None:
        paleta = [
            "#58a6ff", "#3fb950", "#d29922", "#f85149",
            "#bc8cff", "#79c0ff", "#56d364", "#ffa657",
        ]

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    if not frecuencias:
        return img

    # Escalar tamaños: entre 14 y 72 px
    max_freq = max(frecuencias.values())
    min_freq = min(frecuencias.values())

    def tamanio(freq):
        if max_freq == min_freq:
            return 36
        ratio = (np.log1p(freq) - np.log1p(min_freq)) / (
            np.log1p(max_freq) - np.log1p(min_freq)
        )
        return int(14 + ratio * 58)

    # Intentar cargar fuente truetype; si no, usar default
    try:
        font_cache: dict[int, ImageFont.FreeTypeFont] = {}
        def get_font(size):
            if size not in font_cache:
                font_cache[size] = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
                )
            return font_cache[size]
        get_font(16)   # test
    except Exception:
        # Fallback: fuente bitmap (sin bold, tamaño fijo)
        def get_font(size):
            return ImageFont.load_default()

    # Orden: mayor frecuencia primero
    palabras_ord = sorted(frecuencias.items(), key=lambda x: x[1], reverse=True)

    # Posicionamiento en espiral de Arquímedes
    cx, cy = width // 2, height // 2
    ocupadas: list[tuple[int, int, int, int]] = []   # (x1,y1,x2,y2)

    rng = np.random.default_rng(42)

    for idx, (palabra, freq) in enumerate(palabras_ord):
        size = tamanio(freq)
        font = get_font(size)
        try:
            bbox = font.getbbox(palabra)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(palabra) * size // 2, size

        color = paleta[idx % len(paleta)]
        colocada = False

        for radio in range(0, max(width, height), 4):
            angulo_inicio = rng.uniform(0, 2 * np.pi)
            for dtheta in np.linspace(0, 2 * np.pi, max(8, radio)):
                theta = angulo_inicio + dtheta
                x = int(cx + radio * np.cos(theta) - tw / 2)
                y = int(cy + radio * np.sin(theta) - th / 2)

                # Verificar límites
                if x < 4 or y < 4 or x + tw > width - 4 or y + th > height - 4:
                    continue

                # Verificar superposición
                rect = (x, y, x + tw, y + th)
                superpone = any(
                    not (rect[2] < r[0] or rect[0] > r[2] or
                         rect[3] < r[1] or rect[1] > r[3])
                    for r in ocupadas
                )
                if not superpone:
                    draw.text((x, y), palabra, fill=color, font=font)
                    ocupadas.append(rect)
                    colocada = True
                    break
            if colocada:
                break

    return img


def imagen_a_bytes(img: Image.Image) -> bytes:
    """Convierte imagen PIL a bytes PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
