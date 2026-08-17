#  SIAMDA: Multiresolution Analytical & Performance Framework

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentence_Transformers-009688?style=for-the-badge)
![Wavelets](https://img.shields.io/badge/Math-PyWavelets-5C3EE8?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

##  Project Overview
**SIAMDA** (Sistema Integrado de Análisis Multiresolución y Desempeño Académico) is an advanced stochastic and analytical framework. It integrates Natural Language Processing (NLP) with Discrete Wavelet Transforms (DWT) to model and evaluate performance metrics and textual sentiment.

Developed as part of a formal research initiative (co-authored with Dr. Julio César Galindo López, 2026), this tool goes beyond simple dashboards by incorporating rigorous statistical validation, calculating Fisher Information and the Cramér-Rao Bound (CRB) to prove the efficiency of its estimators.

---

##  Core Architecture & Mathematical Pipeline

The system is modularized into distinct analytical pipelines:

1.  **Data Ingestion & Normalization (`datos.py`):** Automated parsing, cleaning, and standardization of complex matrix inputs (CSV/Excel), detecting multi-dimensional performance vectors.
2.  **NLP & Wavelet Extraction (`nlp_wavelet.py`):** 
    *   Generates semantic embeddings from open-ended text data using `sentence-transformers`.
    *   Applies Multiresolution Analysis via Discrete Wavelet Transforms (DWT) to filter noise and extract underlying sentiment trends.
3.  **Theoretical Validation (`cramer_rao.py`):** 
    *   Evaluates the mathematical robustness of the models.
    *   Computes the Fisher Information matrix and the Cramér-Rao lower bound to ensure estimator efficiency.
4.  **Interactive Visualization:** An end-to-end Streamlit web interface with modular pages for data loading, quantitative metrics, stochastic sentiment analysis, and theoretical validation.

---

##  Tech Stack

| Component | Library / Framework |
| :--- | :--- |
| **Web Framework** | `streamlit` |
| **Machine Learning / NLP** | `sentence-transformers` (Multilingual Embeddings) |
| **Applied Mathematics** | `PyWavelets` (DWT), `scipy` (Statistical Testing) |
| **Data Engineering** | `pandas`, `numpy` |
| **Data Visualization** | `plotly` |

---

##  Setup & Local Execution

### Prerequisites
* Python 3.10+
* Git

### Installation Guide

```bash
# 1. Clone the repository
git clone https://github.com/Gabriel44svg/Galindo-C.-J.-y-Villafuerte-C.-G.-2026-.-SIAMDA.git
cd SIAMDA

# 2. Create and activate a virtual environment (Recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```

---

##  Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

22 tests cover the risk flag, the wavelet pipeline, and the Fisher-information/Cramér-Rao
formulas against hand-derived references (`tests/`). They run automatically on every push/PR via
GitHub Actions (`.github/workflows/tests.yml`) and do not require downloading the
`sentence-transformers` model.

---

##  Paper

Two companion technical notes document the system's mathematics, a worked simulated example, and
a sensitivity analysis (sample size, wavelet family, decomposition level) — all clearly labeled
as simulation, not real course data:

* Spanish: [`galindo_villafuerte.pdf`](galindo_villafuerte.pdf) / [`.tex`](galindo_villafuerte.tex)
* English: [`galindo_villafuerte_en.pdf`](galindo_villafuerte_en.pdf) / [`.tex`](galindo_villafuerte_en.tex)

A short [JOSS](https://joss.theoj.org/)-format submission draft is in [`paper.md`](paper.md) /
[`paper.bib`](paper.bib).

##  License

[MIT](LICENSE).

