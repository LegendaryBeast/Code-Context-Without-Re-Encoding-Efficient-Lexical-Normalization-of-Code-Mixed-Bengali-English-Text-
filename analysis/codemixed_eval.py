"""Normalization on code-mixed vs monolingual comments, and at language-switch points (saved outputs only).

Subsets come from BanglaMix's own comment labels (`code_mix_type`, `likely_code_mixed`); references are the silver
references. No repetition guard.

  A. CER / WER / EM by subset for the controlled systems, three splits (one run each), mean over splits.
  B. The same for the three split-42 training runs of CTT, context-only CTT, and the matched word level.
  C. Two-sided paired approximate randomization tests on the code-mixed subset (10,000 permutations).
  D. Token accuracy at language-switch points (token-aligned code-mixed comments).

Run from the repo root:  python analysis/codemixed_eval.py
"""
import contextlib, io, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis")); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import review_analyses as RA
from rapidfuzz.distance import Levenshtein

G = os.path.join(ROOT, "results", "gpu")
df = RA.df
raw = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv")).set_index("id")
df["cm"] = raw["likely_code_mixed"].astype(bool)
df["cmtype"] = raw["code_mix_type"]
LANG = {"BAN", "ENG", "ROM_BAN"}
rng = np.random.RandomState(0)
R = 10000


def errs(sp, ids):
    out = []
    for c in ids:
        h, r = RA.join_spans(sp[c]), df.at[c, "ref"]
        out.append((Levenshtein.distance(h, r), len(r), Levenshtein.distance(h.split(), r.split()), len(r.split()), h == r))
    return pd.DataFrame(out, columns=["ce", "cn", "we", "wn", "em"], index=ids)


def metrics(e):
    return 100 * e.ce.sum() / e.cn.sum(), 100 * e.we.sum() / e.wn.sum(), 100 * e.em.mean()


def subsets(ids):
    x = df.loc[ids]
    out = {"all": ids, "code-mixed": list(x.index[x.cm]), "monolingual": list(x.index[~x.cm & (x.cmtype != "undetermined")])}
    for t in ["ENG+ROM_BAN", "BAN+ENG", "BAN+ENG+ROM_BAN", "BAN+ROM_BAN", "monolingual_BAN", "monolingual_ENG", "monolingual_ROM_BAN"]:
        out[t] = list(x.index[x.cmtype == t])
    return out


def randtest(pairs):
    e = np.concatenate([a.ce.values - b.ce.values for a, b in pairs]).astype(float); L = sum(a.cn.sum() for a, _ in pairs)
    d = 100 * e.sum() / L
    null = 100 * (rng.choice([-1.0, 1.0], size=(R, len(e))) @ e) / L
    return d, (np.sum(np.abs(null) >= abs(d) - 1e-12) + 1) / (R + 1)


# ---------------------------------------------------------------- A. three splits
SYS = {"word_crf": "word_crf", "ufal_plain": "ÚFAL-style", "ctx_plain": "Context-only CTT", "ctx_plain_aks_heads": "CTT"}
E = {(v, s): errs(RA.SPANS[(v, s)], RA.TEST[s]) for v in SYS for s in RA.SEEDS}
SUB = {s: subsets(RA.TEST[s]) for s in RA.SEEDS}
print("== A. CER / WER / EM by subset, mean over three splits (one run each), silver references, no guard")
print("subset sizes (comments, split 13 / 42 / 2026):", {k: [len(SUB[s][k]) for s in RA.SEEDS] for k in SUB[42]})
rows = []
for k in SUB[42]:
    for v, name in SYS.items():
        m = np.array([metrics(E[(v, s)].loc[SUB[s][k]]) for s in RA.SEEDS])
        rows.append(dict(subset=k, system=name, CER=m[:, 0].mean(), CER_sd=m[:, 0].std(ddof=1), WER=m[:, 1].mean(), EM=m[:, 2].mean()))
A = pd.DataFrame(rows)
print(A.round(2).to_string(index=False))

# ---------------------------------------------------------------- B. split 42, three runs
RUNS = {"Word level (matched)": ["nb8_oracle_tags/test_spans_word_plain_align_s42.json",
                                 "nb11_train_seeds/test_spans_word_plain_align_ts1_s42.json",
                                 "nb11_train_seeds/test_spans_word_plain_align_ts2_s42.json"],
        "Context-only CTT": ["nb5_ctt/stage1_seed42/test_spans_ctx_plain_s42.json",
                             "nb11_train_seeds/test_spans_ctx_plain_ts1_s42.json",
                             "nb11_train_seeds/test_spans_ctx_plain_ts2_s42.json"],
        "CTT": ["nb5_ctt/stage2_three_splits/test_spans_ctx_plain_aks_heads_s42.json",
                "nb11_train_seeds/test_spans_ctt_ts1_s42.json", "nb11_train_seeds/test_spans_ctt_ts2_s42.json"]}
ids42 = RA.TEST[42]
ER = {k: [errs(json.load(open(os.path.join(G, p), encoding="utf-8")), ids42) for p in v] for k, v in RUNS.items()}
print("\n== B. split 42, three training runs: mean ± std across runs")
rows = []
for k in ["all", "code-mixed", "monolingual"]:
    for name, runs in ER.items():
        m = np.array([metrics(e.loc[SUB[42][k]]) for e in runs])
        rows.append(dict(subset=k, system=name, CER=f"{m[:, 0].mean():.2f} ± {m[:, 0].std(ddof=1):.2f}",
                         WER=f"{m[:, 1].mean():.2f} ± {m[:, 1].std(ddof=1):.2f}", EM=f"{m[:, 2].mean():.2f} ± {m[:, 2].std(ddof=1):.2f}",
                         runs=[round(x, 2) for x in m[:, 0]]))
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- C. tests on the code-mixed subset
print("\n== C. two-sided paired randomization tests, dCER (negative favours the first system)")
for k in ["code-mixed", "monolingual"]:
    for a, b in [("CTT", "Word level (matched)"), ("Context-only CTT", "Word level (matched)")]:
        d, p = randtest([(x.loc[SUB[42][k]], y.loc[SUB[42][k]]) for x, y in zip(ER[a], ER[b])])
        rel = 100 * d / np.mean([metrics(y.loc[SUB[42][k]])[0] for y in ER[b]])
        print(f"  {k:12s} {a} vs {b} (3 runs, split 42): dCER {d:+.2f} ({rel:+.1f}%)  p = {p:.4f}")
    for a, b in [("ctx_plain_aks_heads", "ufal_plain"), ("ctx_plain_aks_heads", "word_crf"), ("ctx_plain", "ufal_plain")]:
        d, p = randtest([(E[(a, s)].loc[SUB[s][k]], E[(b, s)].loc[SUB[s][k]]) for s in RA.SEEDS])
        print(f"  {k:12s} {SYS[a]} vs {SYS[b]} (3 splits): dCER {d:+.2f}  p = {p:.4f}")

# ---------------------------------------------------------------- D. switch points
print("\n== D. token accuracy at language-switch points (token-aligned code-mixed comments; three splits pooled)")


def switch_flags(labels):
    """A language token (BAN/ENG/ROM_BAN) is a switch token if the nearest language token on either side differs."""
    idx = [i for i, l in enumerate(labels) if l in LANG]
    flag = {}
    for k, i in enumerate(idx):
        nb = [labels[idx[k - 1]]] if k > 0 else []
        nb += [labels[idx[k + 1]]] if k + 1 < len(idx) else []
        flag[i] = any(n != labels[i] for n in nb)
    return flag


rows = []
for s in RA.SEEDS:
    for c in SUB[s]["code-mixed"]:
        toks, labels, ref = df.at[c, "tokens"], df.at[c, "labels"], df.at[c, "ref"].split()
        if len(toks) != len(ref):
            continue
        fl = switch_flags(labels)
        for i, f in fl.items():
            r = dict(split=s, position="switch" if f else "non-switch", tag=labels[i])
            for v in SYS:
                r[v] = RA.bn(RA.SPANS[(v, s)][c][i]) == ref[i]
            rows.append(r)
T = pd.DataFrame(rows)
acc = T.groupby("position")[list(SYS)].mean().mul(100).round(2).rename(columns=SYS)
acc["tokens"] = T.groupby("position").size()
acc["CTT - word_crf"] = (acc["CTT"] - acc["word_crf"]).round(2)
print(acc.to_string())
acc2 = T[T.tag != "BAN"].groupby("position")[list(SYS)].mean().mul(100).round(2).rename(columns=SYS)
acc2["tokens"] = T[T.tag != "BAN"].groupby("position").size()
acc2["CTT - word_crf"] = (acc2["CTT"] - acc2["word_crf"]).round(2)
print("Latin-script language tokens (ENG, ROM_BAN) only:")
print(acc2.to_string())

print("split 42, three training runs (mean accuracy over runs):")
RS = {k: [json.load(open(os.path.join(G, p), encoding="utf-8")) for p in v] for k, v in RUNS.items()}
rows = []
for c in SUB[42]["code-mixed"]:
    toks, labels, ref = df.at[c, "tokens"], df.at[c, "labels"], df.at[c, "ref"].split()
    if len(toks) != len(ref):
        continue
    for i, f in switch_flags(labels).items():
        r = dict(position="switch" if f else "non-switch")
        for name, runs in RS.items():
            r[name] = np.mean([RA.bn(sp[c][i]) == ref[i] for sp in runs])
        rows.append(r)
T2 = pd.DataFrame(rows)
acc3 = T2.groupby("position")[list(RS)].mean().mul(100).round(2)
acc3["tokens"] = T2.groupby("position").size()
print(acc3.to_string())
