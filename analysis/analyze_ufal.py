"""Compare the UFAL-style per-word re-encoding baseline (nb6) with CTT, after the nb6 outputs are downloaded.

Each Kaggle version's `ctt/` output folder goes into its own results/gpu/nb6_ufal/run*/ folder, then run
from the repo root:
    python analysis/analyze_ufal.py
Prints, per split and pooled: guarded CER/WER/EM, paired comment-clustered bootstrap tests on guarded outputs
(same procedure as repetition_guard.py), end-to-end throughput, and the repetition guard on validation outputs.

Generic use (any two systems with saved test spans):  python analysis/analyze_ufal.py --a ctx_plain_aks_heads --b ctx_plain
"""
import argparse, glob, json, os, re, sys, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "src"))
from banglish_align import join_spans  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein
    lev = Levenshtein.distance
except ImportError:                                            # slower pure-Python fallback
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

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="ufal_plain", help="system under test")
ap.add_argument("--b", nargs="+", default=["ctx_plain", "ctx_plain_aks_heads", "word_crf"], help="systems to compare with")
ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 2026])
ap.add_argument("--n_boot", type=int, default=2000)
args = ap.parse_args()

GPU = os.path.join(ROOT, "results", "gpu")
NB6 = sorted(glob.glob(os.path.join(GPU, "nb6_ufal", "run*")))   # run1_ufal_s42_s13, run2_ufal_s2026_ctxplain_s42
DIRS = NB6 + [os.path.join(GPU, "nb5_ctt", d) for d in
                                          ("stage1_seed42", "stage2_three_splits", "stage3a_8_epochs", "stage3b_byt5_base")]
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
RUN = re.compile(r"(.{1,4}?)\1+", re.S)


def bn(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


def max_run(s):
    return max([len(m.group(0)) // len(m.group(1)) for m in RUN.finditer(s)] or [1])


def guard(span, src):   # identical to rep_guard in ctt.py and repetition_guard.py
    allowed = max(3, max_run(bn(src).lower()) + 2)
    return RUN.sub(lambda m: m.group(1) * min(allowed, len(m.group(0)) // len(m.group(1)))
                   + m.group(0)[len(m.group(0)) // len(m.group(1)) * len(m.group(1)):], span)


def find(name):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p


df = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv"))
ref = {i: " ".join(bn(t).split()) for i, t in zip(df.id, df.bengali_transliteration)}
toks = {i: [t.rsplit("/", 1)[0] for t in tl.split(" ")] for i, tl in zip(df.id, df.token_labels)}


def comment_errors(v, s):
    p = find(f"test_spans_{v}_s{s}.json")
    if not p:
        return None
    spans = json.load(open(p, encoding="utf-8"))
    ids = sorted(spans)
    hyp = [join_spans([guard(x, w) for x, w in zip(spans[i], toks[i])]) for i in ids]
    refs = [ref[i] for i in ids]
    return dict(ids=ids, ce=np.array([lev(h, r) for h, r in zip(hyp, refs)]), ln=np.array([len(r) for r in refs]),
                we=np.array([lev(h.split(), r.split()) for h, r in zip(hyp, refs)]), wn=np.array([len(r.split()) for r in refs]),
                em=np.array([h == r for h, r in zip(hyp, refs)]))


systems = [args.a] + args.b
E = {(v, s): comment_errors(v, s) for v in systems for s in args.seeds}
missing = [f"{v}/s{s}" for (v, s), e in E.items() if e is None]
if missing:
    print("missing test spans (skipped):", ", ".join(missing))

print("\n== guarded test metrics")
rows = []
for (v, s), e in E.items():
    if e is not None:
        rows.append(dict(system=v, seed=s, CER=100 * e["ce"].sum() / e["ln"].sum(), WER=100 * e["we"].sum() / e["wn"].sum(),
                         EM=100 * e["em"].mean()))
R = pd.DataFrame(rows)
if len(R):
    print(R.round(2).to_string(index=False))
    g = R.groupby("system", sort=False)
    print("\nmean ± std over available splits")
    print((g[["CER", "WER", "EM"]].mean().round(2).astype(str) + " ± " + g[["CER", "WER", "EM"]].std().round(2).astype(str))
          .assign(n=g.size()).to_string())

print(f"\n== paired comment-clustered bootstrap on guarded CER ({args.n_boot} resamples; dCER < 0: {args.a} better)")
rng = np.random.RandomState(0)
for b in args.b:
    for seeds in [[s] for s in args.seeds] + ([args.seeds] if len(args.seeds) > 1 else []):
        pooled = len(seeds) > 1
        seeds = [s for s in seeds if E[(args.a, s)] is not None and E[(b, s)] is not None]
        if not seeds or (pooled and len(seeds) < 2):   # nothing to pool beyond the single split already shown
            continue
        for s in seeds:
            assert E[(args.a, s)]["ids"] == E[(b, s)]["ids"], f"different test comments for split {s}"
        ea = np.concatenate([E[(args.a, s)]["ce"] for s in seeds]); eb = np.concatenate([E[(b, s)]["ce"] for s in seeds])
        ln = np.concatenate([E[(args.a, s)]["ln"] for s in seeds])
        sizes = [len(E[(args.a, s)]["ln"]) for s in seeds]; off = np.cumsum([0] + sizes)
        d = np.array([100 * (ea[k].sum() - eb[k].sum()) / ln[k].sum() for k in
                      (np.concatenate([o + rng.randint(0, n, n) for o, n in zip(off, sizes)]) for _ in range(args.n_boot))])
        tag = "pooled" if len(seeds) > 1 else f"s{seeds[0]}"
        print(f"{args.a} vs {b:22s} {tag:7s} dCER {100 * (ea.sum() - eb.sum()) / ln.sum():+.2f} "
              f"CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}] p(a not better)={np.mean(d >= 0):.3f}")

print("\n== efficiency (free decoding, test split) and the repetition guard on validation outputs")
eff, val = [], []
for f in sorted(glob.glob(os.path.join(GPU, "nb6_ufal", "run*", "result_*.json"))):
    r = json.load(open(f))
    e = r["efficiency"]["free"]
    eff.append(dict(variant=r["args"]["variant"], seed=r["args"]["seed"], test_words=r["n_test_words"],
                    encoder_bytes=e.get("encoder_bytes"), enc_s=e.get("enc_seconds"), gen_s=e["gen_seconds"],
                    gen_words_per_s=e["words_per_sec"], end_to_end_words_per_s=e.get("total_words_per_sec")))
    if "val" in r:
        val.append(dict(variant=r["args"]["variant"], seed=r["args"]["seed"], val_CER_raw=r["val"]["free"]["CER"],
                        val_CER_guard=r["val"]["free+guard"]["CER"], words_changed=r["val"]["guard_changed_words"]))
if eff:
    print(pd.DataFrame(eff).round(2).to_string(index=False))
else:
    print("no nb6 result files yet in results/gpu/nb6_ufal/")
if val:
    V = pd.DataFrame(val); V["delta"] = V.val_CER_guard - V.val_CER_raw
    print("\nrepetition guard on the full validation split:")
    print(V.round(3).to_string(index=False))
