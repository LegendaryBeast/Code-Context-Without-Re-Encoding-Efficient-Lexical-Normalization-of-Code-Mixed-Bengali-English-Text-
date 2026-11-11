UFAL-style per-word re-encoding baseline (nb6). One folder per Kaggle version of `kaggle/notebooks/nb6_ufal_baseline.ipynb`
(the contents of its `ctt/` output folder; the copies of ctt.py and banglish_align.py are identical to kaggle/src/ and are not kept):

- run1_ufal_s42_s13/              Version 1: ufal_plain on splits 42 and 13 (Kaggle output, 2026-11-04)
- run2_ufal_s2026_ctxplain_s42/   Version 2: ufal_plain on split 2026 + ctx_plain rerun on split 42 (Kaggle output, 2026-11-04)

Analyse with `python analysis/analyze_ufal.py` from the repository root.
