---
title: 'SIAMDA: a wavelet-based sentiment index for academic early warning, with an explicit statistical efficiency check'
tags:
  - Python
  - education
  - learning analytics
  - natural language processing
  - wavelet transform
  - Fisher information
authors:
  - name: César Galindo López
    affiliation: 1
  - name: Gabriel Villafuerte C.
    affiliation: "1, TODO — confirm Gabriel's affiliation before submitting"
affiliations:
  - name: Universidad Panamericana, Facultad de Empresariales, Ciudad UP, Mexico
    index: 1
date: 16 August 2026
bibliography: paper.bib
---

# Summary

`SIAMDA` is an open-source Python/Streamlit application that helps instructors read the
qualitative side of a course -- open-ended feedback survey responses -- with the same rigor
normally reserved for grades. For each response, the software builds a per-word contextual
embedding, projects it onto the response's overall sentence embedding to obtain a scalar signal
indexed by word position, and decomposes that signal with the discrete wavelet transform
[@Mallat1989]. The
resulting split between a global-trend component and a local-fluctuation component yields a
per-student sentiment index $\hat\theta\in(-1,1)$. `SIAMDA` reports this index alongside an
explicit statistical check of its own group-level estimator: Fisher information and the
Cramér--Rao bound under Gaussian and Laplacian models, a Shapiro--Wilk normality test, and a
bootstrap variance estimate. Crucially, the software's interface and its accompanying paper
[@GalindoVillafuerte2026] both state plainly what that statistical check certifies and what it
does not -- a distinction that is easy to overclaim and that an earlier version of this same
software did overclaim, as documented in the paper.

# Statement of need

Learning-analytics dashboards routinely turn survey text into a single number -- a sentiment
score, a satisfaction index -- without stating what, if anything, guarantees that the number is
statistically meaningful. `SIAMDA` targets that gap directly for one specific, common
educational-technology use case: a course instructor who already collects grades and an
open-ended feedback survey, and wants an early-warning signal from the text without training in
NLP or estimation theory.

Two design choices distinguish `SIAMDA` from a typical "run text through a sentiment model and
plot it" tool. First, the wavelet decomposition operates on a signal indexed by **token
position** -- built by projecting each word's contextual embedding onto the response's sentence
embedding -- rather than on the embedding's own feature coordinates, which have no intrinsic
ordering and for which a wavelet transform's approximation/detail split has no principled
interpretation. Second, every efficiency claim the software makes about its own estimator is
paired with an explicit statement of scope: the Cramér--Rao check certifies that the *reported
group mean* is a statistically coherent summary under an assumed parametric model, not that the
underlying pipeline extracts "the maximum information available in the text" -- a claim the
software's own interface used to make and that the accompanying paper shows, with a concrete
simulated example, is close to tautological under the Gaussian model (efficiency $\approx 100\%$
even when the sample demonstrably fails a normality test). Making that distinction explicit, in
both the code's interpretive text and the paper, is the software's main methodological
contribution.

# Software design

`SIAMDA` has four modules: (1) data ingestion and normalization for grade files, including a
binary academic-risk flag; (2) NLP and wavelet extraction, producing the per-student
$\hat\theta$ index described above; (3) theoretical validation, computing Fisher information
[@Fisher1922], the Cramér--Rao bound [@Rao1945; @Cramer1946] under Gaussian and Laplacian
models, a Shapiro--Wilk normality test [@ShapiroWilk1965], a bootstrap variance estimate
[@Efron1979], and a Pearson correlation [@Pearson1896] against grades; and (4) an interactive
Streamlit interface across four pages (data loading,
course metrics, sentiment analysis, theoretical validation) with automatically generated
natural-language interpretations for each metric. Sentence and token embeddings are obtained
from a single `sentence-transformers` [@ReimersGurevych2019] forward pass; the wavelet
decomposition uses `PyWavelets`; statistical tests use `scipy.stats`.

# Mathematics

The sentiment index for student $i$, with token-position signal $s_i(t)=\langle u_{i,t},
x_i\rangle$ (each token embedding $u_{i,t}$ projected onto the sentence embedding $x_i$) and DWT
approximation/detail energies $E^{\mathrm{apr}}_i$, $E^{\mathrm{det}}_i$, is
$$
\hat\theta_i = \tanh\!\left(\frac{E^{\mathrm{det}}_i - E^{\mathrm{apr}}_i}
{E^{\mathrm{det}}_i + E^{\mathrm{apr}}_i + \varepsilon}\right).
$$
Group-level validation compares the bootstrap variance of $\bar\theta$ against the Cramér--Rao
bound $\mathrm{CRB}_\mu=\hat\sigma^2/N$ (Gaussian) or $\hat b^2/N$ (Laplacian); full derivations,
formulas, and a worked simulated example are in the accompanying paper
[@GalindoVillafuerte2026].

# Research impact statement

`SIAMDA` is, to our knowledge, the first open-source tool to pair a wavelet-based text sentiment
index with an explicit, honestly-scoped statistical efficiency certificate for its own aggregate
estimator, rather than reporting a bare score. Its intended impact is methodological: lowering
the barrier for instructors to use a reproducible, quantifiable reading of qualitative feedback,
while making explicit -- in both the software and the paper -- what such a check does and does
not establish, so it is not mistaken for a validated measure of "true sentiment" without further,
external comparison against human annotation.

# AI usage disclosure

*[Note to the authors -- this section is a draft and needs your review before submission: JOSS
now asks papers to disclose the use of AI tools. Large portions of this repository's code,
tests, and both paper drafts (Spanish and English) were produced with the assistance of an AI
coding assistant (Claude), under the direction and review of the human authors, across an
iterative process that included the AI finding and fixing several real bugs (a broken pandas
dtype check, a redundant double model forward-pass, and others documented in the paper's
"Design fix" boxes and git history). Please edit this paragraph to whatever level of detail and
phrasing you're comfortable disclosing before submitting.]*

# Acknowledgements

We acknowledge the open-source maintainers of `sentence-transformers`, `PyWavelets`, `SciPy`,
and `Streamlit`, on which `SIAMDA` is built.

# References
