"""Oracle-tag check of the LID ablation (nb8), after its outputs are in results/gpu/nb8_oracle_tags/.

Split 42, guarded CER (same guard and bootstrap as repetition_guard.py / analyze_ufal.py). Prints
  1. the word/context x none/predicted/oracle table (every run listed, so run-to-run spread is visible);
  2. paired comment-clustered bootstrap tests (2,000 resamples) for the questions the review asked:
     - CRF vs. gold tags with the SAME weights (the train-test tag mismatch, free of training noise);
     - oracle tags vs. no tags, at word level and with context (do tags help at all?);
     - predicted tags vs. no tags at word level (the missing factorial cell);
     - reruns vs. the original nb5 runs (run-to-run variation).
Missing runs are skipped, so the script also works before nb8 is downloaded.

Run from the repo root:  python analysis/analyze_oracle.py
"""
import json, os, re, sys, unicodedata
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "src"))
from banglish_align import join_spans  # noqa: E402

GPU = os.path.join(ROOT, "results", "gpu")
DIRS = [os.path.join(GPU, "nb8_oracle_tags"), os.path.join(GPU, "nb5_ctt", "stage1_seed42"),
        os.path.join(GPU, "nb6_ufal", "run2_ufal_s2026_ctxplain_s42")]
S, N_BOOT = 42, 2000
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
RUN = re.compile(r"(.{1,4}?)\1+", re.S)
lev = Levenshtein.distance


def bn(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


def max_run(s):
    return max([len(m.group(0)) // len(m.group(1)) for m in RUN.finditer(s)] or [1])


def guard(span, src):   # identical to rep_guard in kaggle/src/ctt.py
    allowed = max(3, max_run(bn(src).lower()) + 2)
    return RUN.sub(lambda m: m.group(1) * min(allowed, len(m.group(0)) // len(m.group(1)))
                   + m.group(0)[len(m.group(0)) // len(m.group(1)) * len(m.group(1)):], span)


df = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv"))
ref = {i: " ".join(bn(t).split()) for i, t in zip(df.id, df.bengali_transliteration)}
toks = {i: [t.rsplit("/", 1)[0] for t in tl.split(" ")] for i, tl in zip(df.id, df.token_labels)}


def errors(v):
    """Per-comment guarded and raw character errors of system v on split 42, or None if its spans are missing."""
    for d in DIRS:
        p = os.path.join(d, f"test_spans_{v}_s{S}.json")
        if os.path.exists(p):
            sp = json.load(open(p, encoding="utf-8"))
            ids = sorted(sp)
            return dict(ids=ids, ln=np.array([len(ref[i]) for i in ids]),
                        g=np.array([lev(join_spans([guard(x, w) for x, w in zip(sp[i], toks[i])]), ref[i]) for i in ids]),
                        r=np.array([lev(join_spans(sp[i]), ref[i]) for i in ids]))
    return None


#            cell (level, tags)       runs (variant names; "_goldtags" = same weights tested with gold tags)
CELLS = [(("word", "none"),      ["word_plain_align"]),
         (("word", "predicted"), ["word_crf_align", "word_crf_align_rerun"]),
         (("word", "oracle"),    ["word_crf_align_rerun_goldtags"]),
         (("context", "none"),      ["ctx_plain", "ctx_plain_rerun"]),
         (("context", "predicted"), ["ctx_crf", "ctx_crf_rerun"]),
         (("context", "oracle"),    ["ctx_crf_rerun_goldtags"])]
E = {v: errors(v) for _, vs in CELLS for v in vs}
missing = [v for v, e in E.items() if e is None]
if missing:
    print("missing (skipped):", ", ".join(missing))
cer = lambda e, k="g": 100 * e[k].sum() / e["ln"].sum()

print(f"\n== split {S}: CER per run (guarded / raw)")
rows = []
for (level, tags), vs in CELLS:
    for v in vs:
        if E[v] is not None:
            rows.append(dict(level=level, tags=tags, run=v, CER_guard=cer(E[v]), CER_raw=cer(E[v], "r")))
T = pd.DataFrame(rows)
if len(T):
    print(T.round(2).to_string(index=False))
    print("\nmean guarded CER per cell (over available runs)")
    print(T.pivot_table(index="level", columns="tags", values="CER_guard", aggfunc="mean")
           .reindex(columns=["none", "predicted", "oracle"]).round(2).to_string())

TESTS = [  # (question, a, b): dCER = CER(a) - CER(b); < 0 means a is better
    ("same weights: gold vs CRF tags (context)", "ctx_crf_rerun_goldtags", "ctx_crf_rerun"),
    ("same weights: gold vs CRF tags (word)", "word_crf_align_rerun_goldtags", "word_crf_align_rerun"),
    ("oracle tags vs no tags (context)", "ctx_crf_rerun_goldtags", "ctx_plain"),
    ("oracle tags vs no tags (context, rerun)", "ctx_crf_rerun_goldtags", "ctx_plain_rerun"),
    ("oracle tags vs no tags (word)", "word_crf_align_rerun_goldtags", "word_plain_align"),
    ("predicted tags vs no tags (word)", "word_crf_align", "word_plain_align"),
    ("predicted tags vs no tags (word, rerun)", "word_crf_align_rerun", "word_plain_align"),
    ("predicted tags vs no tags (context, rerun)", "ctx_crf_rerun", "ctx_plain"),
    ("context vs word, no tags", "ctx_plain", "word_plain_align"),
    ("run-to-run: ctx_crf rerun vs original", "ctx_crf_rerun", "ctx_crf"),
    ("run-to-run: word_crf_align rerun vs original", "word_crf_align_rerun", "word_crf_align"),
    ("run-to-run: ctx_plain rerun vs original", "ctx_plain_rerun", "ctx_plain"),
]
print(f"\n== paired comment-clustered bootstrap on guarded CER ({N_BOOT} resamples; dCER < 0: first is better;"
      " p = share of resamples where the first is not better)")
rng = np.random.RandomState(0)
for q, a, b in TESTS:
    if E[a] is None or E[b] is None:
        continue
    assert E[a]["ids"] == E[b]["ids"], (a, b)
    ea, eb, ln = E[a]["g"], E[b]["g"], E[a]["ln"]
    n = len(ln)
    d = np.array([100 * (ea[k].sum() - eb[k].sum()) / ln[k].sum() for k in (rng.randint(0, n, n) for _ in range(N_BOOT))])
    print(f"  {q:46s} dCER {100 * (ea.sum() - eb.sum()) / ln.sum():+.2f}  "
          f"CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]  p={np.mean(d >= 0):.3f}")
