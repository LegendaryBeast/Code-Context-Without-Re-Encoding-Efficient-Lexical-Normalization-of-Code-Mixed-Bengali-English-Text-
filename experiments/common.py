"""Shared utilities for the BanglaMix experiments: loading, parsing, splits, metrics."""
import os
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "BanglaMix.csv")
SPLIT_DIR = os.path.join(ROOT, "data", "splits")
RESULTS = os.path.join(ROOT, "results", "cpu")
LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]
SEEDS = [13, 42, 2026]

os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

BN_RE = re.compile(r"[ঀ-৿]")
LAT_RE = re.compile(r"[A-Za-z]")
# Bengali nukta letters exist both precomposed and decomposed; NFC does not unify them.
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}


def bn_norm(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    s = s.replace("‌", "").replace("‍", "")
    return s


def parse_tokens(token_labels):
    pairs = [t.rsplit("/", 1) for t in token_labels.split(" ")]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def load():
    df = pd.read_csv(DATA)
    toks, labs = zip(*(parse_tokens(t) for t in df.token_labels))
    df["tokens"] = list(toks)
    df["labels"] = list(labs)
    df["targets"] = df.bengali_transliteration.map(lambda s: bn_norm(s).split())
    df["aligned"] = [len(a) == len(b) for a, b in zip(df.tokens, df.targets)]
    return df


def dup_key(s):
    return re.sub(r"[^a-zঀ-৿]", "", unicodedata.normalize("NFC", s.lower()))


def make_split(df, seed, frac=(0.8, 0.1, 0.1)):
    """Stratified (by code_mix_type) split where near-duplicate comments always share a split.
    Cached to splits/ so the Kaggle notebook uses exactly the same partition."""
    path = os.path.join(SPLIT_DIR, f"split_seed{seed}.csv")
    if os.path.exists(path):
        s = pd.read_csv(path)
        return df.merge(s, on="id")
    rng = np.random.RandomState(seed)
    key = df.cleaned_text.map(dup_key)
    groups = pd.DataFrame({"key": key, "cmt": df.code_mix_type}).groupby("key")
    gdf = pd.DataFrame({"cmt": groups.cmt.first(), "n": groups.size()})
    gsplit = {}
    for _, g in gdf.groupby("cmt"):
        g = g.iloc[rng.permutation(len(g))]
        total, acc = g.n.sum(), 0
        for k, n in g.n.items():
            gsplit[k] = "test" if acc < total * frac[2] else "val" if acc < total * (frac[2] + frac[1]) else "train"
            acc += n
    out = pd.DataFrame({"id": df.id, "split": key.map(gsplit)})
    out.to_csv(path, index=False)
    return df.merge(out, on="id")


def script_of(w):
    b, l = bool(BN_RE.search(w)), bool(LAT_RE.search(w))
    if b and l:
        return "mix"
    if b:
        return "bn"
    if l:
        return "lat"
    return "none"


# ---------------------------------------------------------------- metrics
def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer_wer(hyps, refs):
    ce = sum(lev(h, r) for h, r in zip(hyps, refs))
    cn = sum(len(r) for r in refs)
    we = sum(lev(h.split(), r.split()) for h, r in zip(hyps, refs))
    wn = sum(len(r.split()) for r in refs)
    exact = np.mean([h == r for h, r in zip(hyps, refs)])
    return {"CER": 100 * ce / cn, "WER": 100 * we / wn, "Exact": 100 * exact}


def lid_scores(y_true, y_pred):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    f1s = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out = {f"F1_{l}": 100 * f for l, f in zip(LABELS, f1s)}
    out["MacroF1"] = 100 * f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    out["Acc"] = 100 * accuracy_score(y_true, y_pred)
    out["Kappa"] = cohen_kappa_score(y_true, y_pred)
    return out


def cluster_bootstrap(per_comment_true, per_comment_pred, metric, n=1000, seed=0, other_pred=None):
    """Comment-clustered bootstrap CI; if other_pred is given, also a paired p-value (metric(pred) <= metric(other))."""
    rng = np.random.RandomState(seed)
    m = len(per_comment_true)
    vals, diffs = [], []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        yt = [t for i in idx for t in per_comment_true[i]]
        yp = [t for i in idx for t in per_comment_pred[i]]
        v = metric(yt, yp)
        vals.append(v)
        if other_pred is not None:
            yo = [t for i in idx for t in other_pred[i]]
            diffs.append(v - metric(yt, yo))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    res = {"ci_low": lo, "ci_high": hi}
    if other_pred is not None:
        res["p_value"] = float(np.mean(np.array(diffs) <= 0))
    return res


def flat(xs):
    return [x for s in xs for x in s]


def most_common(c):
    return c.most_common(1)[0][0]


def lexicon(pairs):
    d = {}
    for k, v in pairs:
        d.setdefault(k, Counter())[v] += 1
    return {k: most_common(c) for k, c in d.items()}
