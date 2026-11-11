"""Training-seed replicates on split 42 (after nb11): three runs per system, pooled comparisons.

Systems (three training runs each on split 42: the existing seed-42 run + nb11 training seeds 1 and 2):
  CTT              ctx_plain_aks_heads (nb5 stage 2)  + ctt_ts1, ctt_ts2
  Context only     ctx_plain (nb5 stage 1)            + ctx_plain_ts1, ctx_plain_ts2
  Word level       word_plain_align (nb8)             + word_plain_align_ts1, word_plain_align_ts2  (aligned, no tags)
Prints raw and guarded test CER per run, mean +- std over the three runs, and a paired comment-clustered bootstrap
that pools the runs (resampling comments within each training seed), so training variance enters the test.

Run from the repo root:  python analysis/analyze_seeds.py
"""
import glob, json, os, re, sys, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "src"))
from banglish_align import join_spans  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

GPU = os.path.join(ROOT, "results", "gpu")
RUNS = {
    "CTT": [("nb5_ctt/stage2_three_splits", "ctx_plain_aks_heads"), ("nb11_train_seeds", "ctt_ts1"), ("nb11_train_seeds", "ctt_ts2")],
    "Context only": [("nb5_ctt/stage1_seed42", "ctx_plain"), ("nb11_train_seeds", "ctx_plain_ts1"), ("nb11_train_seeds", "ctx_plain_ts2")],
    "Word level": [("nb8_oracle_tags", "word_plain_align"), ("nb11_train_seeds", "word_plain_align_ts1"),
                   ("nb11_train_seeds", "word_plain_align_ts2")],
}
N_BOOT = 2000
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
RUN = re.compile(r"(.{1,4}?)\1+", re.S)


def bn(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


def guard(span, src):   # identical to rep_guard in kaggle/src/ctt.py
    mr = max([len(m.group(0)) // len(m.group(1)) for m in RUN.finditer(bn(src).lower())] or [1])
    allowed = max(3, mr + 2)
    return RUN.sub(lambda m: m.group(1) * min(allowed, len(m.group(0)) // len(m.group(1)))
                   + m.group(0)[len(m.group(0)) // len(m.group(1)) * len(m.group(1)):], span)


df = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv"))
toks = {i: [t.rsplit("/", 1)[0] for t in tl.split(" ")] for i, tl in zip(df.id, df.token_labels)}
ref = {i: " ".join(bn(t).split()) for i, t in zip(df.id, df.bengali_transliteration)}

E, ids = {}, None
for sys_, runs in RUNS.items():
    for k, (d, v) in enumerate(runs):
        p = os.path.join(GPU, d, f"test_spans_{v}_s42.json")
        if not os.path.exists(p):
            print(f"missing (skipped): {d}/test_spans_{v}_s42.json")
            continue
        sp = json.load(open(p, encoding="utf-8"))
        ids = ids or sorted(sp)
        assert sorted(sp) == ids, f"{v}: different test comments"
        for g in (False, True):
            ce = np.array([Levenshtein.distance(join_spans([guard(x, w) for x, w in zip(sp[c], toks[c])] if g else sp[c]),
                                                ref[c]) for c in ids])
            E[(sys_, k, g)] = ce
ln = np.array([len(ref[c]) for c in ids])

rows = []
for sys_, runs in RUNS.items():
    for k, (d, v) in enumerate(runs):
        if (sys_, k, True) in E:
            rows.append(dict(system=sys_, run=v, raw=100 * E[(sys_, k, False)].sum() / ln.sum(),
                             guarded=100 * E[(sys_, k, True)].sum() / ln.sum()))
R = pd.DataFrame(rows)
print("== split 42 test CER per training run")
print(R.round(2).to_string(index=False))
g = R.groupby("system", sort=False)[["raw", "guarded"]]
print("\nmean ± std over training runs")
print((g.mean().round(2).astype(str) + " ± " + g.std().round(2).astype(str)).assign(n=g.size()).to_string())

print(f"\n== paired bootstrap pooling training runs ({N_BOOT} resamples; dCER < 0: first better)")
rng = np.random.RandomState(0)
for a, b in [("CTT", "Word level"), ("Context only", "Word level"), ("CTT", "Context only")]:
    ks = [k for k in range(3) if (a, k, True) in E and (b, k, True) in E]
    if not ks:
        continue
    out = []
    for gd in (False, True):
        ea = np.concatenate([E[(a, k, gd)] for k in ks]); eb = np.concatenate([E[(b, k, gd)] for k in ks])
        L = np.tile(ln, len(ks)); n = len(ln)
        d = np.array([100 * (ea[s].sum() - eb[s].sum()) / L[s].sum() for s in
                      (np.concatenate([k * n + rng.randint(0, n, n) for k in range(len(ks))]) for _ in range(N_BOOT))])
        out.append(f"{'guarded' if gd else 'raw'} {100 * (ea.sum() - eb.sum()) / L.sum():+.2f} "
                   f"[{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}] p={np.mean(d >= 0):.3f}")
    print(f"  {a:12s} vs {b:12s} runs={len(ks)}  " + "   ".join(out))
