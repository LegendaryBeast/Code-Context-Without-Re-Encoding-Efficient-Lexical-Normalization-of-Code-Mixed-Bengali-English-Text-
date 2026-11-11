Oracle-tag / word x context check of the LID ablation (nb8). Contents of the `ctt/` output folder of one Kaggle run of
`kaggle/notebooks/nb8_oracle_tags.ipynb` (2026-11-05, split 42, GPU T4 x2; the copies of ctt.py and banglish_align.py
in the download were identical to kaggle/src/ and are not kept):

- ctx_crf_rerun          context, trained with silver tags; tested with CRF tags and, same weights, gold tags (`*_goldtags_*`)
- word_crf_align_rerun   word level, same as above
- word_plain_align       word level, no tags

Analyse with `python analysis/analyze_oracle.py` from the repository root.
