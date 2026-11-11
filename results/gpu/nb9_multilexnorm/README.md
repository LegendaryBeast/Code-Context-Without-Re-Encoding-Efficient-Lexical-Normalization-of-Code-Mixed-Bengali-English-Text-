# nb9: CTT on MultiLexNorm (Indonesian–English `iden`, Turkish–German `trde`)

Outputs of `kaggle/notebooks/nb9_multilexnorm.ipynb` (two Kaggle versions, one per language), 2026-11-05.
Per job `<lang>_<system>_s<seed>`: `result_*.json` (scores, history, efficiency), `log_*.txt`, `test_spans_*.json`
(per-token predictions, used for ERR) and `test_text_*.json`. `mln_<lang>.csv` (the converted MultiLexNorm data) is not redistributed; recreate it as described in the top-level README (section "MultiLexNorm");
`nb9_err_<lang>.csv` is the notebook's own summary table.

Analysis: `python analysis/analyze_mln.py` (Table 13 trained rows, bootstrap tests, cost ratios) and
`python analysis/score_mln_submissions.py --repo <multilexnorm checkout>` (Table 13 published rows).
