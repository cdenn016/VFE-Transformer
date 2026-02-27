# Manuscript Review: "Attention, Transformers, and Backpropagation are Degenerate Limits of the Variational Free Energy Principle"

**Reviewed file:** `Attention/GL(K)_attention.tex`
**Date:** 2026-02-27

---

## CRITICAL ERRORS (must fix before submission)

### 1. Placeholder Figure Captions (Lines 1941, 1986)

Two figures have `\caption{XXXXX}` placeholder text instead of real captions:

- **Line 1941** (`fig:glk_training`): GL(K) training curves figure — caption is literally `XXXXX`
- **Line 1986** (`fig:glk_semantic`): Belief/gauge frame clustering figure — caption is literally `XXXXX`

### 2. Placeholder Body Text with Missing Data (Lines 1990–1994)

Section 5.3.2 "Emergent Semantic Structure" contains extensive placeholder text:

- **Line 1990:** `"In the $\mathrm{GL}(10)$ XXX exploratory run"` — "XXX" placeholder
- **Line 1990:** `"$\mu \in \mathbb{R}^{60}$ XXX"` — "XXX" placeholder
- **Line 1990:** `"approximately XX\% of the variance, indicating XXX"` — percentage and explanation missing
- **Line 1992:** `"$\sim$XXX\% of the total variance"` — percentage missing
- **Line 1994:** `"$\sim$XX\% of the variance in the belief space"` — percentage missing

### 3. Broken Cross-References (will produce "??" in compiled PDF)

| Line(s) | Reference | Issue |
|---------|-----------|-------|
| 1287 | `\ref{sec:formulation}` | Label `sec:formulation` does not exist anywhere in the document |
| 1287 | `\ref{sec:derivation}` | Label `sec:derivation` does not exist anywhere in the document |
| 1742, 2051 | `\ref{tab:tau_sweep}` | Label `tab:tau_sweep` does not exist; no temperature sweep table is labeled |

**Likely fix for line 1287:** The text says "Sections~\ref{sec:formulation}--\ref{sec:derivation}" — these are probably remnants from an earlier draft. They likely should point to `sec:free_energy_section` (Section 3) and `sec:mixture_derivation` (Section 3.3), or similar.

**Likely fix for `tab:tau_sweep`:** A temperature sweep table exists conceptually (the data is discussed in the text) but was never formally labeled. Either create the table and label it, or change references to point to the temperature sweep figure (`fig:temperature_sweep`).

### 4. Missing Figure Files (9 figures referenced but not found)

The following `\includegraphics` paths reference files that do not exist in the repository:

| Line | Path | Issue |
|------|------|-------|
| 1893 | `/figs/mu_q_center_norm_vacuum.png` | Absolute path starting with `/figs/` — will fail |
| 1895 | `/figs/mu_q_center_norm_ranked_vacuum.png` | Same absolute path issue |
| 1910 | `/figs/mu_q_center_norm_summary.png` | Same absolute path issue |
| 1912 | `/figs/mu_q_center_norm_ranked.png` | Same absolute path issue |
| 1933 | `attention/figs/K=60_GL(10)_training_curves.png` | File not found in repository |
| 1938 | `attention/figs/K=60_GL(10)_training_curves_non_diagonal.png` | File not found in repository |
| 1978 | `attention/figs/belief_clustering_non_diag.png` | File not found in repository |
| 1983 | `attention/figs/gauge_frame_clustering_non_diag.png` | File not found in repository |

The `/figs/` paths (lines 1893, 1895, 1910, 1912) use an absolute path that won't resolve. These should be relative paths, likely `figs/mu_q_center_norm_vacuum.png` etc., and the actual image files need to be generated and added to the repository.

### 5. Dimensional Mismatch Between Figure Captions and Body Text

- **Figure captions** (lines 1979, 1984) say: `$\mu \in \mathbb{R}^{30}$` and `$\phi \in \mathfrak{gl}(30) \cong \mathbb{R}^{900}$`
- **Body text** (lines 1990, 1992) says: `$\mu \in \mathbb{R}^{60}$` and `$\phi \in \mathfrak{gl}(10) \cong \mathbb{R}^{100}$`

These are contradictory. The captions and body text describe different model configurations. One set needs to be corrected to match the actual figures being shown.

---

## MODERATE ERRORS

### 6. Grammatical Error: Double "is" (Line 561)

> "The free energy for the alignment component **is** the KL divergence between the variational posterior and the generative model **is:**"

Should be one of:
- "The free energy for the alignment component — the KL divergence between the variational posterior and the generative model — is:"
- "The alignment free energy is the KL divergence between the variational posterior and the generative model:"

### 7. Grammatical Error: "training updating" (Line 1132)

> "gradient descent on this free energy recovers standard transformer training **updating** thereby providing..."

Should be: "training **update**" (noun, not gerund).

### 8. Spelling: "apriori" (Line 745)

> "the prior precision is **apriori** known and fixed"

Should be "**a priori**" (two words; standard Latin phrase, conventionally italicized in English text).

### 9. Notation Table Inconsistency (Line 411)

The notation table lists `$\phi_i$` as having type `$\mathfrak{g}\!\cong\!\mathbb{R}^3$`. This is only correct for the `SO(3)` case used in the agent-based simulations. For the general `GL(K)` case used throughout most of the paper and experiments, the dimension is `K^2`, not 3. The table entry should reflect the general case or note the specialization.

### 10. JMLR Heading Year (Line 32)

`\jmlrheading{}{2025}{}{}{}{}` — The year 2025 should be updated to 2026 if submitting in 2026 (experimental data directories contain dates from February 2026).

### 11. Editor Field (Line 41)

`\editor{TBD}` — Needs to be filled in or handled per JMLR submission guidelines.

### 12. Missing Cover Letter Preamble

The cover letter (`jmlr_coverletter.tex`, line 2) includes `\input{jmlr_coverletter_preamble.tex}`, but this file does not appear to exist in the repository.

### 13. Inconsistent Model Specifications in Results

The text transitions between multiple GL(K) configurations without clear delineation:
- Table 2 (line 1955): Reports **GL(20), K=80** as the primary result
- Figure captions (lines 1933–1938): Show **GL(10), K=60** training curves
- Body text (line 1990): Discusses **GL(10)** exploratory run with `$\mu \in \mathbb{R}^{60}$`
- Figure captions (lines 1979, 1984): Show **GL(30)** with `$\mu \in \mathbb{R}^{30}$`

The narrative should clearly distinguish which configuration each figure/discussion refers to, ideally stating explicitly when transitioning between exploratory and primary results.

---

## BIBLIOGRAPHY ISSUES

### 14. All 59 Manuscript Citations Resolve (No Missing Keys)

Every `\citep{}` and `\citet{}` key in the manuscript has a corresponding entry in `references.bib`. No compilation errors expected from missing citations.

### 15. Extensive Duplicate Entries in references.bib

The bibliography file contains significant duplication:

**Exact duplicate keys** (BibTeX will silently use only one, risking wrong metadata):
- `Rovelli1996` — defined **4 times** (lines 424, 800, 1532, 1578)
- `Wheeler1990` — defined twice (different entry types: `@book` vs `@incollection`)
- `Jacobson1995` — defined twice (different author name formatting)
- `Verlinde2011` — defined twice
- `Clark2013` — defined twice
- `Arndt2014`, `Aspelmeyer2014`, `Hoffman2019`, `Putnam1981`, `Ladyman2014`, `Chentsov1982` — each defined twice

**Semantic duplicates** (same work under different keys; ~24 pairs):
The bib file contains both Title-Case keys (e.g., `Vaswani2017`) and lowercase descriptive keys (e.g., `vaswani2017attention`) for the same works. The manuscript uses only the lowercase versions. The Title-Case duplicates are dead weight.

**Year mismatches:**
- `shen2008coarse`: Key says 2008 but `year` field says 2011
- `anderson1984basic`: Key says 1984 but `year` field says 1988

### 16. Recommendation

Clean `references.bib` by removing:
- All 11+ exact duplicate entries (keeping the most complete version of each)
- All ~24 unused Title-Case aliases
- Fix the year mismatches in `shen2008coarse` and `anderson1984basic`

---

## REDUNDANCIES

### 17. Repeated Definition of Gauge Action on Gaussians

- **Eq. (4), line 247:** Defines `$\rho_q(\Omega) \cdot (\mu_q, \Sigma_q) = (\rho_q(\Omega)\mu_q, \rho_q(\Omega)\Sigma_q\rho_q(\Omega)^\top)$`
- **Eq. (6), line 337:** Defines `$\Omega_{ij}(c) \cdot (\mu_j, \Sigma_j) = (\Omega_{ij}(c)\mu_j, \Omega_{ij}(c)\Sigma_j\Omega_{ij}(c)^\top)$`

These are the same expression. Consider referencing Eq. (4) at line 337 instead of restating.

### 18. Redundant Summary in Section 3.6 (Lines 844–848)

The Summary paragraph at the end of Section 3 closely restates derivations from Sections 3.3–3.5 nearly verbatim. Key phrases are repeated word-for-word:
- "The forward KL divergence emerges exactly as the alignment energy from the mixture generative model"
- "the softmax form of attention weights follows from Lagrange optimization"

This paragraph could be shortened to 2–3 sentences referencing the key results without repeating the derivation steps.

### 19. Symmetry Breaking Discussed in Four Places

The concept of observation-induced symmetry breaking appears in:
1. **Section 3.7** (line 839): Theoretical introduction
2. **Section 4.3.1** (line 1134): As loss function interpretation
3. **Section 5.3** (line 1883): Empirical demonstration
4. **Discussion 6.1** (line 2047): Re-summary

While some repetition is expected in a journal paper, the theoretical treatment in Section 3.7 substantially overlaps with the discussion in Section 4.3.1. Consider consolidating.

### 20. Flat Bundle Limit Discussed Redundantly

- **Line 324** (Section 2.3.3): Explains the 0-dimensional case and its implications
- **Section 4.2** (lines 928–936): Introduces Limit 2 (flat bundle)
- **Section 6.3** (lines 2069–2093): Extended treatment of flat bundle implications, conjecture

The content at line 324 and Section 4.2 cover substantially the same ground. The Discussion section (6.3) then revisits this with the holonomy calculation, which is new content, but the setup repeats previous material.

### 21. Inconsistent Use of "Gauge-Covariant Variational Free Energy"

The full phrase "gauge-covariant variational free energy" (or close variants like "gauge-equivariant variational free energy") appears 15+ times. The abbreviation "gauge VFE" is used occasionally but inconsistently. Recommend defining "gauge VFE" after first use and using it consistently.

---

## MINOR STYLE/FORMATTING ISSUES

### 22. Double Space Before "standard" (Line 1132)

> "recovers **  **standard transformer training"

Extra space character.

### 23. Dense Validation Protocol Paragraph (Line 1609)

The seven-phase validation protocol is described in a single very long paragraph. Consider using a numbered list or breaking into sub-paragraphs for readability.

### 24. Acknowledgments: AI Model Name (Line 2151)

"Claude Sonnet 4.5" — Verify this is the exact model name. Ensure the disclosure meets JMLR's current policy on AI tool usage.

### 25. Extra Blank Lines in Cover Letter

The cover letter file (`jmlr_coverletter.tex`) has ~40 trailing blank lines after `\end{document}`.

### 26. The $GL(K)$ vs $\mathrm{GL}(K)$ Inconsistency (Line 1128)

> "Our $GL(K)$ experiments"

Should be `$\mathrm{GL}(K)$` for consistency with the rest of the paper (roman font for group names).

---

## SUMMARY OF ACTION ITEMS

| Priority | Count | Category |
|----------|-------|----------|
| **Critical** | 5 | Placeholder text, broken refs, missing figures, dimension mismatch |
| **Moderate** | 8 | Grammar, notation, year, missing files, consistency |
| **Bibliography** | 3 | Duplicates, year mismatches, cleanup needed |
| **Redundancy** | 5 | Repeated content that could be consolidated |
| **Minor** | 5 | Formatting, style |
