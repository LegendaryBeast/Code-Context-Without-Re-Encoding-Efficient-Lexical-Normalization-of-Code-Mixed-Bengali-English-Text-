# Context Without Re-Encoding: code, splits, annotations, and outputs

This is the release for the article *Context Without Re-Encoding: Efficient Lexical Normalization of Code-Mixed
Bengali–English Text* (T. H. Prappo, N. Gazi, N. Islam, M. S. Rahman; Shahjalal University of Science and
Technology). It contains:

- the **Context-aware Token Transducer (CTT)** and every controlled system in the article, all from one training script;
- the **near-duplicate-grouped splits** of BanglaMix, as comment IDs only;
- the **annotation sheets**: the two-annotator LID audit and the two-annotator analysis of 200 CTT errors;
- the **saved outputs**: test predictions, per-run configurations, training histories, and timings for every trained system;
- the **analysis scripts** that produce the tables and statistical tests from those outputs.

Every number in the article's controlled comparisons can be recomputed from the saved outputs on a CPU, without
retraining (see [Reproducing the tables](#reproducing-the-tables-from-saved-outputs-cpu)).

## Contents

```
data/splits/                 split_seed{13,42,2026}.csv: comment id -> train / val / test (no comment text)
experiments/                 CPU experiments: corpus audit, classical LID, lexicon normalization, noise audit
  common.py                    data loading, splits, metrics, comment-clustered bootstrap
  01_data_audit.py             corpus statistics, near-duplicate groups, leakage checks
  02_lid_baselines.py          script rule, lexicon, logistic regression, CRF (Table 7, classical rows)
  03_translit_baselines.py     identity and lexicon normalization baselines (Table 8, lexicon rows)
  04_noise_audit.py            confident-learning flags and the 400-token audit sheet (Section 5)
  05_extra_analyses.py         CRF confusion matrix and learning curve
kaggle/                      GPU experiments, run as Kaggle notebooks (one NVIDIA T4, 16 GB, fp32)
  src/ctt.py                   CTT and all controlled normalization systems (--mode focus | word | ufal)
  src/banglish_align.py        monotonic dynamic-programming token-to-span aligner
  src/joint_mtl.py             shared helpers imported by ctt.py
  src/mln_convert.py           converts MultiLexNorm to the same span format
  src/nb*.py                   notebook sources (the cells are split on "# %%")
  notebooks/*.ipynb            the same notebooks, ready to import into Kaggle
  build_notebooks.py           rebuilds notebooks/ from src/ and the Kaggle dataset zip from data/
analysis/                    CPU analyses of the saved outputs
  review_analyses.py           unguarded vs guarded CER, bootstrap tests, chunk boundaries, error taxonomy
  analyze_seeds.py             three training runs on split 42
  analyze_stages.py, analyze_oracle.py, analyze_ufal.py, ablations_and_copy_gate.py, repetition_guard.py
  randomization_tests.py       two-sided paired randomization tests (p-values of Tables 9-11, 13-14 and the text)
  randomization_tests_output.txt   output of randomization_tests.py
  analyze_mln.py, score_mln_submissions.py, mln_copy_err.py   MultiLexNorm scoring
  src/nb7_ctt_error_analysis.py   samples the 200 CTT errors for human labelling
  error_annotation/            ctt_errors_key.csv (sample), ctt_errors_annotator{1,2}_filled.csv (labels)
results/cpu/                 outputs of experiments/ (including the LID audit sheet and key, CRF predictions)
results/gpu/                 outputs of the Kaggle notebooks, one folder per notebook (see the table below)
```

## Getting the data

BanglaMix and MultiLexNorm are **not redistributed** here; obtain them from their original sources.

**BanglaMix.** Afridi, Partha, and Khan, *BanglaMix: A dataset of token-level annotated Bengali-English code-mixed
social media comments with Bengali transliteration*, Data in Brief 69 (2026) 113317,
https://doi.org/10.1016/j.dib.2026.113317 (CC BY 4.0). Place the CSV at `data/BanglaMix.csv`. The file used for the
article has SHA-256
`8108215f738e7daa06b2082c9945f03ae5ab5d08e76c869e2368233dee9fcd89`; check with `shasum -a 256 data/BanglaMix.csv`.

**MultiLexNorm.** Clone the shared-task repository at the commit used by the notebook
(`kaggle/src/nb9_multilexnorm.py`):

```bash
git clone https://bitbucket.org/robvanderg/multilexnorm.git /tmp/multilexnorm
git -C /tmp/multilexnorm checkout e58e946375299a04355fa2d7fa9e1c769234e68a
python kaggle/src/mln_convert.py --repo /tmp/multilexnorm --out /tmp/mln --langs iden trde
cp /tmp/mln/iden/mln.csv results/gpu/nb9_multilexnorm/mln_iden.csv
cp /tmp/mln/trde/mln.csv results/gpu/nb9_multilexnorm/mln_trde.csv
```

The converted files used for the article have SHA-256 `fb9679e2f4185b2b68bd82d96230e4b272b74ceeaf1df3cdff6f1aaa25ad07dc`
(`mln_iden.csv`) and `9cc0d11bfb5b6a1d3771b9ea91035494be44e0cc8e5620796205605f61acba38` (`mln_trde.csv`).

## Installation

```bash
python -m pip install -r requirements.txt
```

The CPU scripts were run with Python 3.13; `requirements.txt` lists the versions used locally where they are known.
The GPU notebooks ran in Kaggle's standard Python GPU image, which provides PyTorch and Hugging Face Transformers.

## System names

The article's names map to the variant names used in the code and in the output file names:

| Article | Variant | Notebook | Configuration |
|---|---|---|---|
| CTT | `ctx_plain_aks_heads` (`ctt` in nb11) | nb5, nb11 | `--mode focus --align 1 --pretrain_aksharantar 200000 --lam 0.4 --copy_mu 0.5 --detach_heads 1` |
| Context-only CTT | `ctx_plain` (`ctx_plain_rerun` in nb6) | nb5, nb6, nb11 | `--mode focus --align 1` |
| ÚFAL-style re-encoding | `ufal_plain` | nb6 | `--mode ufal --align 1 --max_enc_tokens 4096` |
| Word level (matched) | `word_plain_align` | nb8, nb11 | `--mode word --align 1` |
| `word_crf` | `word_crf` | nb5 | `--mode word --ext_lid crf --align 0` |
| Word level with aligner, CRF tag | `word_crf_align` (`_rerun` in nb8) | nb5, nb8 | `--mode word --ext_lid crf --align 1` |
| Context with CRF tag | `ctx_crf` (`_rerun` in nb8) | nb5, nb8 | `--mode focus --ext_lid crf --align 1` |
| Context with joint LID | `ctx_joint` | nb5 | `--mode focus --align 1 --lam 0.4 --inject 1 --copy_mu 0.5` |
| CTT, 8 epochs / ByT5-base | `ctx_plain_aks_heads_e8` / `..._byt5-base` | nb5 | as CTT, `--epochs 8` / `--model google/byt5-base` |
| MultiLexNorm: word level, context-only, ÚFAL-style, CTT | `word`, `ctx`, `ufal`, `ctx_copy` | nb9 | `ctx_copy` = `ctx` + `--copy_mu 0.5 --detach_heads 1` |

Shared settings (all normalization runs): `google/byt5-small`, AdamW, learning rate 3e-4, 5 epochs on BanglaMix and
10 on MultiLexNorm, up to 48 decoded words and 8 chunks per micro-batch, 2-step gradient accumulation, chunks of at
most 160 bytes, copy threshold 0.9, greedy decoding, best epoch by CER on 400 validation comments. Every
`result_*.json` stores the complete argument set of its run under `args`, the per-epoch loss, validation CER,
training time, and peak memory under `history`, and the timings under `efficiency`.

## Saved outputs

| Folder | Notebook | Contents | Article |
|---|---|---|---|
| `results/gpu/nb1_lid_transformers/` | nb1 | test predictions of mBERT, MuRIL, BanglaBERT, BanglishBERT, XLM-R (three splits) | Table 7 |
| `results/gpu/nb2_byt5/` | nb2 | earlier ByT5-word and ByT5-sentence recipes | Table 8 |
| `results/gpu/nb3_llm_qwen/` | nb3 | prompted Qwen2.5-7B: subset IDs, raw outputs, scores | Tables 7-8 |
| `results/gpu/nb5_ctt/stage1_seed42/` | nb5 | component ablations on split 42 | Table 11 |
| `results/gpu/nb5_ctt/stage2_three_splits/` | nb5 | CTT, context-only CTT, `word_crf` on three splits | Tables 8-9 |
| `results/gpu/nb5_ctt/stage3a_8_epochs/`, `stage3b_byt5_base/` | nb5 | training length and model size | Section 10.3 |
| `results/gpu/nb6_ufal/` | nb6 | ÚFAL-style re-encoding (three splits), context-only rerun | Tables 8-9, 12 |
| `results/gpu/nb8_oracle_tags/` | nb8 | tag-source runs (CRF and annotated tags), matched word level | Tables 10-11 |
| `results/gpu/nb9_multilexnorm/` | nb9 | four systems x two languages x three seeds | Tables 13-14 |
| `results/gpu/nb11_train_seeds/` | nb11 | training seeds 1 and 2 on split 42 (CTT, context-only, word level) | Tables 9-10, 12 |

File types: `test_spans_*.json` (one output string per source token), `test_text_*_{free,copy,lid}.json` (decoded
comments; `free` = free decoding, `copy` = copy gate; `+guard` files apply the exploratory repetition guard, which is
not used in any primary result), `test_lid_*.json` (LID-head predictions), `result_*.json`, `log_*.txt`.

## Reproducing the tables from saved outputs (CPU)

Run from this folder after placing `data/BanglaMix.csv` (and, for MultiLexNorm, the two `mln_*.csv` files):

```bash
python experiments/01_data_audit.py
python experiments/02_lid_baselines.py
python experiments/03_translit_baselines.py
python experiments/04_noise_audit.py
python experiments/05_extra_analyses.py
python analysis/review_analyses.py
python analysis/analyze_seeds.py
python analysis/randomization_tests.py
python analysis/analyze_mln.py
python analysis/mln_copy_err.py
```

`score_mln_submissions.py --repo <multilexnorm checkout>` rescores the published shared-task submissions.

## Retraining (GPU)

1. Run `python kaggle/build_notebooks.py` after placing `data/BanglaMix.csv`; it writes `kaggle/kaggle_dataset.zip`
   (BanglaMix, the splits, CRF predictions, and the noise matrix).
2. Upload the zip as a private Kaggle dataset and attach it to each notebook in `kaggle/notebooks/`.
3. Run the notebooks with a T4 GPU and copy each notebook's output folder into the matching `results/gpu/` folder.

Seeds: split seeds 13, 42, and 2026; training seeds 1 and 2 for the additional split-42 runs (nb11); training seeds 13,
42, and 2026 on MultiLexNorm (nb9). One CTT epoch on BanglaMix takes about 22 minutes on a T4, and the one-epoch
Aksharantar warm start 40-46 minutes.

## Annotations

- `results/cpu/04_audit_sheet.csv`: 400 tokens (confident-learning flags and a random sample) with both annotators'
  LID labels; `04_audit_key_DO_NOT_SHOW_ANNOTATORS.csv` gives the corpus label, the confident-learning suggestion, and the flag
  for each token (it was hidden from the annotators during labelling).
- `analysis/error_annotation/ctt_errors_key.csv`: the 200 sampled CTT errors (split, comment ID, token, CTT output,
  reference, automatic category); `ctt_errors_annotator{1,2}_filled.csv`: the two annotators' labels
  (M1-M5 model errors, R1 reference wrong, R2 both acceptable).

All annotators are authors of the article; the error labels were made by the first and second authors.

## Citation

If you use this release, please cite the article (bibliographic details to be added on publication) and BanglaMix.

## License

This repository is released under the MIT License (see `LICENSE`). BanglaMix and MultiLexNorm are not
included and remain under their own licenses; saved outputs and annotation sheets that quote BanglaMix tokens or
references carry its CC BY 4.0 attribution requirement.
