"""Merge the CTT stages in results/gpu/nb5_ctt/ (stage 1, 2 and 3a): 3-split tables, paired bootstrap tests on raw
outputs, and the comments that contribute most to CER differences.  Run from the repo root:  python analysis/analyze_stages.py"""
import json, os, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB5 = os.path.join(ROOT, "results", "gpu", "nb5_ctt")
DIRS = [os.path.join(NB5, d) for d in ("stage1_seed42", "stage2_three_splits", "stage3a_8_epochs")]
SEEDS = [13, 42, 2026]
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}


def bn(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


def lev(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def find(name):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


df = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv"))
ref = {i: " ".join(bn(t).split()) for i, t in zip(df.id, df.bengali_transliteration)}

# ---------------------------------------------------------------- per-seed metrics from result files
SYSTEMS = ["word_crf", "ctx_plain", "ctx_plain_aks_heads", "ctx_plain_aks_heads_e8"]
rows = []
for v in SYSTEMS:
    for s in SEEDS:
        p = find(f"result_{v}_s{s}.json")
        if not p:
            continue
        r = json.load(open(p))
        for mode, m in r["norm"].items():
            e = r["efficiency"][mode]
            rows.append(dict(system=v, mode=mode, seed=s, CER=m["CER"], WER=m["WER"], EM=m["Exact"], TokAcc=m["TokAcc"],
                             ENG=m["TokAcc_ENG"], ROM=m["TokAcc_ROM_BAN"], OTH=m["TokAcc_OTHER"], UNK=m["TokAcc_UNK"],
                             OOV=m["TokAcc_OOV"], decoded=e["decoded_word_share"], wps=e["words_per_sec"]))
        if "lid" in r:
            rows[-1]["LID_F1"] = r["lid"]["MacroF1"]
R = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("== per seed")
print(R.round(2).to_string(index=False))
print("\n== mean ± std over seeds")
g = R.groupby(["system", "mode"], sort=False)
cols = ["CER", "WER", "EM", "TokAcc", "ENG", "ROM", "OTH", "UNK", "OOV", "decoded", "wps"]
print((g[cols].mean().round(2).astype(str) + " ± " + g[cols].std().round(2).astype(str)).assign(n=g.size()).to_string())

# ---------------------------------------------------------------- comment-level errors
E, LN, IDS = {}, {}, {}
for s in SEEDS:
    ids = sorted(json.load(open(find(f"test_text_word_crf_s{s}_free.json"), encoding="utf-8")))
    IDS[s], LN[s] = ids, np.array([len(ref[i]) for i in ids])
    for v, mode in [(x, "free") for x in SYSTEMS] + [("ctx_plain_aks_heads", "copy")]:
        p = find(f"test_text_{v}_s{s}_{mode}.json")
        if p:
            A = json.load(open(p, encoding="utf-8"))
            E[(v, mode, s)] = np.array([lev(A[i], ref[i]) for i in ids])

rng = np.random.RandomState(0)


def boot(a, b, seeds, n=2000):
    """Paired comment-clustered bootstrap; with several seeds, resample within each split and pool."""
    ea = np.concatenate([E[a + (s,)] for s in seeds]); eb = np.concatenate([E[b + (s,)] for s in seeds])
    ln = np.concatenate([LN[s] for s in seeds])
    off = np.cumsum([0] + [len(LN[s]) for s in seeds])
    d = []
    for _ in range(n):
        k = np.concatenate([o + rng.randint(0, len(LN[s]), len(LN[s])) for o, s in zip(off, seeds)])
        d.append(100 * (ea[k].sum() - eb[k].sum()) / ln[k].sum())
    d = np.array(d)
    return 100 * (ea.sum() - eb.sum()) / ln.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5), float(np.mean(d >= 0))


print("\n== paired bootstrap (dCER < 0: first system better)")
pairs = [(("ctx_plain", "free"), ("word_crf", "free")), (("ctx_plain_aks_heads", "free"), ("word_crf", "free")),
         (("ctx_plain_aks_heads", "free"), ("ctx_plain", "free")), (("ctx_plain_aks_heads", "copy"), ("ctx_plain_aks_heads", "free")),
         (("ctx_plain_aks_heads_e8", "free"), ("ctx_plain_aks_heads", "free"))]
for a, b in pairs:
    for seeds in [[s] for s in SEEDS] + [SEEDS]:
        dd, lo, hi, p = boot(a, b, seeds)
        tag = "pooled" if len(seeds) > 1 else f"s{seeds[0]}"
        print(f"{a[0]}[{a[1]}] vs {b[0]}[{b[1]}] {tag:7s} dCER {dd:+.2f} CI [{lo:+.2f}, {hi:+.2f}] p={p:.3f}")

# ---------------------------------------------------------------- outliers: comments driving a CER gap
print("\n== largest per-comment error differences, ctx_plain_aks_heads - ctx_plain (seed 42)")
s = 42
diff = E[("ctx_plain_aks_heads", "free", s)] - E[("ctx_plain", "free", s)]
tot = LN[s].sum()
order = np.argsort(-diff)
print(f"total extra errors {diff.sum()} chars = {100 * diff.sum() / tot:+.2f} CER; top-5 comments contribute "
      f"{100 * diff[order[:5]].sum() / tot:+.2f} CER")
A = json.load(open(find(f"test_text_ctx_plain_aks_heads_s{s}_free.json"), encoding="utf-8"))
B = json.load(open(find(f"test_text_ctx_plain_s{s}_free.json"), encoding="utf-8"))
for k in order[:5]:
    i = IDS[s][k]
    print(f"\n  +{diff[k]} chars | ref len {len(ref[i])}")
    print("  REF :", ref[i][:220]); print("  AKS :", A[i][:220]); print("  PLN :", B[i][:220])
without = np.delete(diff, order[:5])
print(f"\nseed 42 dCER without those 5 comments: {100 * without.sum() / np.delete(LN[s], order[:5]).sum():+.2f}")
