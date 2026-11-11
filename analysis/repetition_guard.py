"""Post-hoc test of a repetition guard on saved CTT word spans (no GPU).
Rule, fixed before looking at its effect: a unit of 1-4 characters repeated consecutively in an output span is
clamped to max(3, longest consecutive repeat run of any 1-4-char unit in the SOURCE token + 2).
Run from the repo root:  python analysis/repetition_guard.py"""
import json, os, re, sys, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "src"))
from banglish_align import join_spans  # noqa: E402

NB5 = os.path.join(ROOT, "results", "gpu", "nb5_ctt")
DIRS = [os.path.join(NB5, d) for d in ("stage1_seed42", "stage2_three_splits", "stage3a_8_epochs", "stage3b_byt5_base")]
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
RUN = re.compile(r"(.{1,4}?)\1+", re.S)


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


def max_run(s):
    return max([len(m.group(0)) // len(m.group(1)) for m in RUN.finditer(s)] or [1])


def guard(span, src):
    allowed = max(3, max_run(bn(src).lower()) + 2)
    return RUN.sub(lambda m: m.group(1) * min(allowed, len(m.group(0)) // len(m.group(1))) + m.group(0)[len(m.group(0)) // len(m.group(1)) * len(m.group(1)):], span)


def find(name):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p


df = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv"))
ref = {i: " ".join(bn(t).split()) for i, t in zip(df.id, df.bengali_transliteration)}
toks = {i: [t.rsplit("/", 1)[0] for t in tl.split(" ")] for i, tl in zip(df.id, df.token_labels)}

rows = []
for v in ["word_crf", "word_crf_align", "ctx_crf", "ctx_plain", "ctx_joint", "ctx_joint_aks", "ctx_plain_aks_heads", "ctx_plain_aks_heads_e8", "ctx_plain_aks_heads_byt5-base"]:
    for s in (13, 42, 2026):
        p = find(f"test_spans_{v}_s{s}.json")
        if not p:
            continue
        spans = json.load(open(p, encoding="utf-8"))
        ids = sorted(spans)
        ln = sum(len(ref[i]) for i in ids)
        e0 = e1 = changed = runaway = 0
        for i in ids:
            sp = spans[i]
            g = [guard(x, w) for x, w in zip(sp, toks[i])]
            changed += sum(a != b for a, b in zip(sp, g))
            runaway += sum(len(x.encode("utf-8")) >= 150 for x in sp)       # near the 160-byte generation cap
            e0 += lev(join_spans(sp), ref[i]); e1 += lev(join_spans(g), ref[i])
        rows.append(dict(system=v, seed=s, CER=100 * e0 / ln, CER_guard=100 * e1 / ln, spans_changed=changed, runaway_spans=runaway))
R = pd.DataFrame(rows)
R["delta"] = R.CER_guard - R.CER
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
g = R.groupby("system", sort=False)
print("\nmean ± std over available seeds")
print(pd.DataFrame({"CER": g.CER.mean().round(2).astype(str) + " ± " + g.CER.std().round(2).astype(str),
                    "CER_guard": g.CER_guard.mean().round(2).astype(str) + " ± " + g.CER_guard.std().round(2).astype(str),
                    "n": g.size()}).to_string())


# ---------------------------------------------------------------- paired bootstrap on guarded outputs (3 splits pooled)
def comment_errors(v, s):
    spans = json.load(open(find(f"test_spans_{v}_s{s}.json"), encoding="utf-8"))
    ids = sorted(spans)
    return ids, np.array([lev(join_spans([guard(x, w) for x, w in zip(spans[i], toks[i])]), ref[i]) for i in ids])


rng = np.random.RandomState(0)
E, LN = {}, {}
for v in ["word_crf", "ctx_plain", "ctx_plain_aks_heads", "ctx_plain_aks_heads_e8"]:
    for s in (13, 42, 2026):
        ids, E[(v, s)] = comment_errors(v, s)
        LN[s] = np.array([len(ref[i]) for i in ids])
print("\npaired bootstrap, guarded outputs (dCER < 0: first better)")
for a, b in [("ctx_plain", "word_crf"), ("ctx_plain_aks_heads", "word_crf"), ("ctx_plain_aks_heads", "ctx_plain"),
             ("ctx_plain_aks_heads_e8", "ctx_plain_aks_heads"), ("ctx_plain_aks_heads_e8", "word_crf")]:
    for seeds in [(13,), (42,), (2026,), (13, 42, 2026)]:
        ea = np.concatenate([E[(a, s)] for s in seeds]); eb = np.concatenate([E[(b, s)] for s in seeds])
        ln = np.concatenate([LN[s] for s in seeds]); off = np.cumsum([0] + [len(LN[s]) for s in seeds])
        d = np.array([100 * (ea[k].sum() - eb[k].sum()) / ln[k].sum() for k in
                      (np.concatenate([o + rng.randint(0, len(LN[s]), len(LN[s])) for o, s in zip(off, seeds)]) for _ in range(2000))])
        tag = "pooled" if len(seeds) > 1 else f"s{seeds[0]}"
        print(f"{a:20s} vs {b:10s} {tag:7s} dCER {100 * (ea.sum() - eb.sum()) / ln.sum():+.2f} "
              f"CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}] p={np.mean(d >= 0):.3f}")


# ---------------------------------------------------------------- Stage 3b: byt5-base vs byt5-small (seeds 13 and 42 only)
for s in (13, 42):
    ids, E[("ctx_plain_aks_heads_byt5-base", s)] = comment_errors("ctx_plain_aks_heads_byt5-base", s)
for a, b in [("ctx_plain_aks_heads_byt5-base", "ctx_plain_aks_heads"), ("ctx_plain_aks_heads_byt5-base", "word_crf")]:
    for seeds in [(13,), (42,), (13, 42)]:
        ea = np.concatenate([E[(a, s)] for s in seeds]); eb = np.concatenate([E[(b, s)] for s in seeds])
        ln = np.concatenate([LN[s] for s in seeds]); off = np.cumsum([0] + [len(LN[s]) for s in seeds])
        d = np.array([100 * (ea[k].sum() - eb[k].sum()) / ln[k].sum() for k in
                      (np.concatenate([o + rng.randint(0, len(LN[s]), len(LN[s])) for o, s in zip(off, seeds)]) for _ in range(2000))])
        tag = "pooled" if len(seeds) > 1 else f"s{seeds[0]}"
        print(f"{a:30s} vs {b:20s} {tag:7s} dCER {100 * (ea.sum() - eb.sum()) / ln.sum():+.2f} "
              f"CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}] p={np.mean(d >= 0):.3f}")
