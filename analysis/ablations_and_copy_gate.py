"""Paper Table 6 (component ablations, split 42) and the copy-gate rows of Tables 4-5, on guarded outputs.

1. Ablations: paired comment-clustered bootstrap (2,000 resamples) of raw and guarded CER between Stage-1 variants
   on split 42 (word_crf -> word_crf_align -> ctx_crf -> ctx_plain; ctx_joint -> ctx_plain / ctx_joint_aks; ctx_plain -> CTT).
2. Copy gate (5-epoch CTT, 3 splits): guarded CER of the copy-mode outputs and the pooled bootstrap against free decoding.
   Copy mode copies some words instead of decoding them; every decoded word is identical to free decoding, so the
   guard is applied to those words by replacing each runaway free-decoding span in the copy-mode text.

Run from the repo root:  python analysis/ablations_and_copy_gate.py
"""
import json, os, re, sys, unicodedata
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "src"))
from banglish_align import join_spans  # noqa: E402

NB5 = os.path.join(ROOT, "results", "gpu", "nb5_ctt")
DIRS = [os.path.join(NB5, d) for d in ("stage1_seed42", "stage2_three_splits")]
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


def find(name):
    for d in DIRS:
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    raise FileNotFoundError(name)


df = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv"))
ref = {i: " ".join(bn(t).split()) for i, t in zip(df.id, df.bengali_transliteration)}
toks = {i: [t.rsplit("/", 1)[0] for t in tl.split(" ")] for i, tl in zip(df.id, df.token_labels)}


def bootstrap(groups, rng, n=2000):
    """groups: list of (errors_a, errors_b, ref_lengths) per split; resample comments within each split, pool."""
    ea = np.concatenate([g[0] for g in groups]); eb = np.concatenate([g[1] for g in groups])
    ln = np.concatenate([g[2] for g in groups]); sizes = [len(g[2]) for g in groups]; off = np.cumsum([0] + sizes)
    d = np.array([100 * (ea[k].sum() - eb[k].sum()) / ln[k].sum() for k in
                  (np.concatenate([o + rng.randint(0, m, m) for o, m in zip(off, sizes)]) for _ in range(n))])
    return 100 * (ea.sum() - eb.sum()) / ln.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5), float(np.mean(d >= 0))


# ------------------------------------------------------------------ 1. ablations on split 42
S = 42
ids = sorted(json.load(open(find(f"test_spans_word_crf_s{S}.json"), encoding="utf-8")))
LN = np.array([len(ref[i]) for i in ids])
VARIANTS = ["word_crf", "word_crf_align", "ctx_crf", "ctx_plain", "ctx_joint", "ctx_joint_aks", "ctx_plain_aks_heads"]
STEPS = [("word_crf", "word_crf_align"), ("word_crf_align", "ctx_crf"), ("ctx_crf", "ctx_plain"),
         ("ctx_joint", "ctx_plain"), ("ctx_joint", "ctx_joint_aks"), ("ctx_plain", "ctx_plain_aks_heads")]


def errors(v, guarded):
    sp = json.load(open(find(f"test_spans_{v}_s{S}.json"), encoding="utf-8"))
    return np.array([lev(join_spans([guard(x, w) for x, w in zip(sp[i], toks[i])] if guarded else sp[i]), ref[i]) for i in ids])


print(f"== Table 6: ablations on split {S} (dCER < 0: the change helps; p = share of resamples where it does not)")
for guarded in (True, False):
    rng = np.random.RandomState(0)
    E = {v: errors(v, guarded) for v in VARIANTS}
    print("\n" + ("guarded" if guarded else "raw") + " CER:", {v: round(100 * E[v].sum() / LN.sum(), 2) for v in VARIANTS})
    for a, b in STEPS:
        d, lo, hi, p = bootstrap([(E[b], E[a], LN)], rng)
        print(f"  {a:15s} -> {b:20s} dCER {d:+.2f}  CI [{lo:+.2f}, {hi:+.2f}]  p={p:.3f}")

# ------------------------------------------------------------------ 2. learned copy gate, 3 splits
print("\n== copy gate (CTT, 5 epochs): guarded CER of copy mode vs. free decoding")
groups, cers = [], []
for s in (13, 42, 2026):
    sp = json.load(open(find(f"test_spans_ctx_plain_aks_heads_s{s}.json"), encoding="utf-8"))
    cp = json.load(open(find(f"test_text_ctx_plain_aks_heads_s{s}_copy.json"), encoding="utf-8"))
    cids = sorted(sp)
    ef, ec = [], []
    for i in cids:
        g = [guard(x, w) for x, w in zip(sp[i], toks[i])]
        ef.append(lev(join_spans(g), ref[i]))
        t = cp[i]
        for a, b in zip(sp[i], g):
            if a != b and a in t:        # the word was decoded in copy mode too: apply the same clamp
                t = t.replace(a, b, 1)
        ec.append(lev(t, ref[i]))
    ln = np.array([len(ref[i]) for i in cids])
    groups.append((np.array(ec), np.array(ef), ln))
    cers.append(100 * sum(ec) / ln.sum())
    print(f"  split {s}: free+guard {100 * sum(ef) / ln.sum():.3f}  copy+guard {cers[-1]:.3f}")
print(f"  copy+guard mean {np.mean(cers):.2f} ± {np.std(cers, ddof=1):.2f}")
d, lo, hi, p = bootstrap(groups, np.random.RandomState(0))
print(f"  pooled copy vs. free: dCER {d:+.2f}  CI [{lo:+.2f}, {hi:+.2f}]  p={p:.3f}")
