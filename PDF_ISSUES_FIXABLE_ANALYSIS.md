# PDF Issues - Fixability Analysis

**Date:** November 2, 2024
**Context:** Analysis of which thesis PDF issues can be fixed programmatically in LaTeX source

---

## Issues I HAVE Fixed ✅

### 1. Table of Contents Formatting for Third-Level Headings
**Status:** ✅ FIXED

**Problem:** Third-level headings (\subsubsection) were misaligned in TOC

**Solution:** Added to [main.tex:39-41](overleaf/main.tex):
```latex
% Set depth for table of contents to include subsubsections
\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}
```

**Result:** TOC will now properly display and number \subsubsection entries

---

###2. Placeholder Appendix A Content
**Status:** ✅ FIXED

**Problem:** [Anexo1.tex](overleaf/Anexos/Anexo1.tex) contained Lorem Ipsum placeholder text

**Solution:** Completely rewrote appendix with meaningful content:
- Software Environment (Python, PyTorch, dependencies)
- Model Configurations (SCLIP, SAM2, Stable Diffusion parameters)
- Evaluation Benchmarks (PASCAL VOC, COCO-Stuff details)
- Computational Performance table
- Code Repository structure and usage instructions

**Result:** Appendix now provides complete implementation details for reproducibility

---

## Issues That REQUIRE MANUAL INTERVENTION ❌

### 3. Placeholder Figures (8 total)
**Status:** ❌ CANNOT FIX PROGRAMMATICALLY

**Locations:**
1. [Introduccion.tex:16](overleaf/Capitulos/Introduccion.tex) - System Overview Concept
2. [Capitulo1.tex:29](overleaf/Capitulos/Capitulo1.tex) - Evolution of Segmentation Approaches
3. [Capitulo2.tex:47](overleaf/Capitulos/Capitulo2.tex) - Dense SCLIP + SAM2 Pipeline Diagram
4. [Capitulo2.tex:110](overleaf/Capitulos/Capitulo2.tex) - Multi-Scale CLIP Feature Visualization
5. [Capitulo3.tex:245](overleaf/Capitulos/Capitulo3.tex) - Qualitative Results Grid
6. [Capitulo3.tex:305](overleaf/Capitulos/Capitulo3.tex) - Ablation Study Visual Comparison
7. [Capitulo3.tex:363](overleaf/Capitulos/Capitulo3.tex) - Failure Cases Visualization

**Reason:** These require:
- Actual experimental results (screenshots, plots)
- Diagram creation (pipeline flowcharts)
- Visual design work (figure layouts, annotations)

**Recommendation:** Generate these figures from experimental data or create diagrams using tools like:
- TikZ (for pipeline diagrams)
- Matplotlib/Seaborn (for result plots)
- Image editing tools (for qualitative result grids)

---

### 4. Incorrect Page Numbers in List of Figures/Tables
**Status:** ⚠️ MAY AUTO-CORRECT

**Problem:** Page numbers don't match actual figure locations in PDF

**Reason:** This is typically caused by:
- LaTeX compilation cache issues
- Missing `\clearpage` commands before figures
- Float placement conflicts

**Solution:** Try these steps:
1. Delete all auxiliary files (`.aux`, `.toc`, `.lof`, `.lot`)
2. Recompile LaTeX 2-3 times (LaTeX needs multiple passes to resolve references)
3. If still incorrect, add `\clearpage` before problematic figures to force placement

**Commands to run:**
```bash
cd /home/pablo/aux/tfm/overleaf
rm -f *.aux *.toc *.lof *.lot *.out
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

---

### 5. Missing Section Content / Duplicate Content (Page 22 area)
**Status:** ⚠️ PARTIALLY IDENTIFIED

**Problem:** Section "Alternative Exploration: Proposal-Based Segmentation" (line 197 in Capitulo2.tex) contains DUPLICATE dense SCLIP content instead of describing the proposal-based approach

**What's wrong:**
- Lines 11-195: Correctly describe "Primary Approach: Dense SCLIP with Novel SAM2 Refinement" ✅
- Lines 197-329: INCORRECTLY describe dense SCLIP AGAIN under "Alternative Exploration" ❌
  - Should briefly describe proposal-based (SAM2 automatic → CLIP scoring)
  - Instead has duplicate CSA formulas, multi-layer aggregation, etc.
- Lines 330-384: Correctly compare both approaches ✅

**Recommendation:**
Lines 197-329 need to be REPLACED with a brief (3-4 page) summary of the proposal-based approach:
- SAM2 automatic mask generation
- Multi-scale CLIP voting
- Multi-instance selection
- Why it's complementary (good for "things", less for "stuff")

This is complex structural reorganization requiring careful rewriting, not simple search-replace.

---

## Issues from User's Checklist Still Pending

### From [THESIS_REVISION_CHECKLIST.md](THESIS_REVISION_CHECKLIST.md):

**Completed:**
- ✅ Abstract rewrite (Resumen.tex)
- ✅ Introduction contributions (Introduccion.tex)
- ✅ Chapter 2 section titles updated
- ✅ System Overview section rewritten with dense pipeline
- ✅ Appendix A content replaced

**Still Needed:**
- ⚠️ **Chapter 2:** Lines 197-329 need condensing (proposal-based as brief alternative)
- ⚠️ **Chapter 3:** Replace specific metrics (69.3%, 48.09%, 49.52%) with [TBD] placeholders
- ⚠️ **Chapter 3:** Reorganize to present dense results first, proposal-based as secondary
- ⚠️ **All chapters:** Create/insert actual figures to replace 8 placeholders

---

## Summary

### What I Fixed:
1. ✅ TOC formatting for third-level headings
2. ✅ Appendix A placeholder content

### What Needs Manual Work:
3. ❌ 8 placeholder figures (need actual experimental figures)
4. ⚠️ Page number references (try recompiling 3× first)
5. ⚠️ Chapter 2 lines 197-329 (duplicate content, needs condensing)
6. ⚠️ Chapter 3 metrics placeholders (search-replace with [TBD])
7. ⚠️ Chapter 3 reorganization (dense results first)

### Estimated Effort:
- **Figures:** 4-6 hours (create diagrams + result visualizations)
- **Chapter 2 condensing:** 1-2 hours (rewrite proposal-based section)
- **Chapter 3 metrics:** 30 minutes (search-replace)
- **Chapter 3 reorganization:** 1-2 hours (reorder sections)
- **Total:** ~8-10 hours of remaining work

---

## Recommendation

**Next steps in priority order:**

1. **Immediate (I can help):**
   - Replace metrics in Chapter 3 with [TBD] placeholders

2. **Short-term (requires manual work):**
   - Rewrite Chapter 2 lines 197-329 to briefly describe proposal-based as alternative
   - Reorganize Chapter 3 to present dense results first

3. **Medium-term (requires experimental output):**
   - Generate 8 placeholder figures from actual data/diagrams

4. **Final (before submission):**
   - Run full dataset evaluation
   - Replace [TBD] metrics with actual results
   - Final compilation and page number verification

---

**Generated:** November 2, 2024
**Status:** Thesis structure mostly correct, main work is content condensing and figure generation
