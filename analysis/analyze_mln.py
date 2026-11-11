"""CTT on MultiLexNorm code-switched data (nb9), after its outputs are in results/gpu/nb9_multilexnorm/.

Prints per language (iden = Indonesian-English, trde = Turkish-German):
  1. test ERR per system and training seed (official MultiLexNorm formula, case-sensitive), raw and with the
     repetition guard, and mean +- std over seeds, next to ÚFAL's published test ERR;
  2. paired sentence-clustered bootstrap (2,000 resamples, seeds pooled) of the ERR difference between systems;
  3. cost of ÚFAL-style re-encoding relative to context-only CTT (encoder bytes, end-to-end speed, training time).

Run from the repo root:  python analysis/analyze_mln.py [--dir results/gpu/nb9_multilexnorm]
"""
import argparse, glob, json, os, re, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(ROOT, "results", "gpu", "nb9_multilexnorm"))
ap.add_argument("--n_boot", type=int, default=2000)
args = ap.parse_args()

# ÚFAL's official test outputs (MultiLexNorm repository, submissions/ÚFAL-{1,2}), scored with the official formula
PUBLISHED = {"iden": (66.15, 67.18), "trde": (68.19, 68.62)}
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
RUN = re.compile(r"(.{1,4}?)\1+", re.S)


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


def per_sentence(spans, data, guarded):
    """Per test sentence: correct tokens, tokens, tokens the annotators changed."""
    cor, tot, chg = [], [], []
    for cid, toks, gold in data:
        if cid not in spans:
            continue
        pred = spans[cid]
        assert len(pred) == len(gold), cid
        if guarded:
            pred = [guard(p, w) for p, w in zip(pred, toks)]
        g = [unicodedata.normalize("NFC", x) for x in gold]
        p = [unicodedata.normalize("NFC", x) for x in pred]
        cor.append(sum(a == b for a, b in zip(g, p))); tot.append(len(g)); chg.append(sum(r != a for r, a in zip(toks, g)))
    return np.array(cor), np.array(tot), np.array(chg)


def err(cor, tot, chg):
    return 100 * (cor.sum() - tot.sum() + chg.sum()) / chg.sum()


for lang in ("iden", "trde"):
    csv = os.path.join(args.dir, f"mln_{lang}.csv")
    if not os.path.exists(csv):
        print(f"\n== {lang}: no results yet ({csv} missing)")
        continue
    d = pd.read_csv(csv, keep_default_na=False)
    d = d[d.id.str.startswith(f"{lang}_test_")]
    data = [(c, [t.rsplit("/", 1)[0] for t in tl.split(" ")], json.loads(s)) for c, tl, s in zip(d.id, d.token_labels, d.target_spans)]
    S = {}                                   # (system, seed) -> spans
    for f in sorted(glob.glob(os.path.join(args.dir, f"test_spans_{lang}_*_s*.json"))):
        name, seed = os.path.basename(f)[len("test_spans_"):-len(".json")].rsplit("_s", 1)
        S[(name[len(lang) + 1:], int(seed))] = json.load(open(f, encoding="utf-8"))
        if len(S[(name[len(lang) + 1:], int(seed))]) != len(data):
            print(f"WARNING: {os.path.basename(f)} covers {len(S[(name[len(lang) + 1:], int(seed))])} of {len(data)} test sentences")
    print(f"\n== {lang}: test ERR (official formula, case-sensitive); published ÚFAL-1 / ÚFAL-2: "
          f"{PUBLISHED[lang][0]:.2f} / {PUBLISHED[lang][1]:.2f}")
    rows = []
    for (sys_, seed), sp in sorted(S.items()):
        rows.append(dict(system=sys_, seed=seed, ERR_raw=err(*per_sentence(sp, data, False)), ERR_guard=err(*per_sentence(sp, data, True))))
    T = pd.DataFrame(rows)
    print(T.round(2).to_string(index=False))
    g = T.groupby("system")[["ERR_raw", "ERR_guard"]]
    print("\nmean ± std over seeds")
    print((g.mean().round(2).astype(str) + " ± " + g.std().round(2).astype(str)).assign(n=g.size()).to_string())

    print(f"\npaired sentence-clustered bootstrap of guarded ERR, seeds pooled ({args.n_boot} resamples; dERR > 0: first is better)")
    rng = np.random.RandomState(0)
    systems = sorted({s for s, _ in S})
    for a, b in [("ctx", "ufal"), ("ctx_copy", "ufal"), ("ctx", "word"), ("ctx_copy", "ctx")]:
        seeds = sorted({s for x, s in S if x == a} & {s for x, s in S if x == b})
        if a not in systems or b not in systems or not seeds:
            continue
        A = [per_sentence(S[(a, s)], data, True) for s in seeds]; B = [per_sentence(S[(b, s)], data, True) for s in seeds]
        ca, cb = np.concatenate([x[0] for x in A]), np.concatenate([x[0] for x in B]); chg = np.concatenate([x[2] for x in A])
        assert len(ca) == len(cb), "systems cover different test sentences"
        n, k = len(A[0][0]), len(seeds)
        diffs = []
        for _ in range(args.n_boot):
            idx = np.concatenate([i * n + rng.randint(0, n, n) for i in range(k)])
            diffs.append(100 * (ca[idx].sum() - cb[idx].sum()) / chg[idx].sum())
        diffs = np.array(diffs)
        print(f"  {a:9s} vs {b:5s} dERR {100 * (ca.sum() - cb.sum()) / chg.sum():+.2f}  CI [{np.percentile(diffs, 2.5):+.2f}, "
              f"{np.percentile(diffs, 97.5):+.2f}]  p(first not better)={np.mean(diffs <= 0):.3f}  seeds={seeds}")

    print("\ncost of ÚFAL-style re-encoding relative to context-only CTT (mean over seeds)")
    def eff(sys_):
        out = []
        for f in glob.glob(os.path.join(args.dir, f"result_{lang}_{sys_}_s*.json")):
            r = json.load(open(f)); e = r["efficiency"]["free"]
            ep = [h["train_min"] for h in r["history"] if h["stage"] == "banglamix"]
            out.append((e["encoder_bytes"], e["total_words_per_sec"], np.mean(ep)))
        return np.array(out).mean(0) if out else None
    u, c = eff("ufal"), eff("ctx")
    if u is not None and c is not None:
        print(f"  encoder bytes {u[0] / c[0]:.1f}x | CTT end-to-end speed {c[1] / u[1]:.1f}x of ÚFAL-style | training time per epoch {u[2] / c[2]:.1f}x")
