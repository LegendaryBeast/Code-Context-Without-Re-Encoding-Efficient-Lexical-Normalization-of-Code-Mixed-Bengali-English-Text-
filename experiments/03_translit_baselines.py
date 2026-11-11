"""Bengali-script normalisation (back-transliteration) baselines that run on CPU.

All systems are token-level and output the space-joined sequence, scored against the full
normalised bengali_transliteration of every test comment (aligned or not).
Token-level accuracy is reported on the 93% of comments whose target is token-aligned.
"""
import difflib
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from common import LABELS, RESULTS, SEEDS, bn_norm, cer_wer, flat, load, make_split, most_common

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def key(w):
    return bn_norm(w).lower()


class Lexicon:
    """Most-frequent-target lookup, optionally keyed on the LID tag, with fuzzy fallback for OOV words."""

    def __init__(self, train, by_label, fuzzy):
        self.by_label, self.fuzzy = by_label, fuzzy
        self.d, self.w = defaultdict(Counter), defaultdict(Counter)
        for r in train.itertuples():
            for w, l, y in zip(r.tokens, r.labels, r.targets):
                self.d[(key(w), l)][y] += 1
                self.w[key(w)][y] += 1
        self.d = {k: most_common(c) for k, c in self.d.items()}
        self.w = {k: most_common(c) for k, c in self.w.items()}
        self.vocab = list(self.w)
        self.cache = {}

    def __call__(self, w, l):
        k = key(w)
        if self.by_label and (k, l) in self.d:
            return self.d[(k, l)]
        if k in self.w:
            return self.w[k]
        if self.fuzzy and l in ("ENG", "ROM_BAN", "UNK"):
            if k not in self.cache:
                m = difflib.get_close_matches(k, [v for v in self.vocab if v[:1] == k[:1]], n=1, cutoff=0.8)
                self.cache[k] = self.w[m[0]] if m else None
            if self.cache[k]:
                return self.cache[k]
        return bn_norm(w).translate(BN_DIGITS)


def run(df, seed):
    d = make_split(df, seed)
    with open(os.path.join(RESULTS, f"crf_pred_seed{seed}.pkl"), "rb") as f:
        crf = pickle.load(f)
    tr = d[(d.split == "train") & d.aligned]
    te = d[d.split == "test"]
    refs = [" ".join(t) for t in te.targets]
    lex = Lexicon(tr, by_label=False, fuzzy=False)
    lexf = Lexicon(tr, by_label=False, fuzzy=True)
    lexl = Lexicon(tr, by_label=True, fuzzy=True)
    COPY = ("BAN", "OTHER")

    systems = {
        "Copy (identity)": lambda w, l, p: bn_norm(w),
        "Word lexicon": lambda w, l, p: lex(w, l),
        "Word lexicon + fuzzy OOV": lambda w, l, p: lexf(w, l),
        "LID-conditioned lexicon (CRF LID)": lambda w, l, p: lexl(w, p),
        "LID-cond. lexicon + hard copy-gate (CRF LID)": lambda w, l, p: bn_norm(w) if p in COPY else lexl(w, p),
        "LID-conditioned lexicon (oracle LID)": lambda w, l, p: lexl(w, l),
        "LID-cond. lexicon + hard copy-gate (oracle LID)": lambda w, l, p: bn_norm(w) if l in COPY else lexl(w, l),
    }
    train_vocab = set(key(w) for w in flat(tr.tokens))
    rows = []
    for name, f in systems.items():
        hyps, tok_ok, tok_lab, tok_oov = [], [], [], []
        for r in te.itertuples():
            out = [f(w, l, p) for w, l, p in zip(r.tokens, r.labels, crf[r.id])]
            hyps.append(" ".join(out))
            if r.aligned:
                tok_ok += [o == y for o, y in zip(out, r.targets)]
                tok_lab += list(r.labels)
                tok_oov += [key(w) not in train_vocab for w in r.tokens]
        s = cer_wer(hyps, refs)
        ok, lab, oov = np.array(tok_ok), np.array(tok_lab), np.array(tok_oov)
        s["TokAcc"] = 100 * ok.mean()
        for l in LABELS:
            s[f"TokAcc_{l}"] = 100 * ok[lab == l].mean()
        s["TokAcc_OOV"] = 100 * ok[oov].mean()
        s.update(system=name, seed=seed)
        rows.append(s)
        print(f"  {name:50s} CER {s['CER']:.2f}  WER {s['WER']:.2f}  Exact {s['Exact']:.2f}")
    return rows


if __name__ == "__main__":
    df = load()
    rows = []
    for seed in SEEDS:
        print("seed", seed)
        rows += run(df, seed)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "03_translit_all_seeds.csv"), index=False)
    cols = ["CER", "WER", "Exact", "TokAcc"] + [f"TokAcc_{l}" for l in LABELS] + ["TokAcc_OOV"]
    agg = res.groupby("system", sort=False)[cols].agg(["mean", "std"])
    summ = pd.DataFrame({c: [f"{m:.2f} ± {s:.2f}" for m, s in zip(agg[c]["mean"], agg[c]["std"])] for c in cols}, index=agg.index)
    summ.to_csv(os.path.join(RESULTS, "03_translit_summary.csv"))
    pd.set_option("display.width", 250)
    print(summ.to_string())
