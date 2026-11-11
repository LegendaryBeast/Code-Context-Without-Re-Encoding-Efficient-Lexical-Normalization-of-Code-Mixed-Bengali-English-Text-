"""ERR of CTT with the copy gate on MultiLexNorm, reconstructed from saved outputs (no new run).

With the copy gate, each token's output is either the free-decoding output for that token (same decoder, greedy) or the
source token itself. The per-token choice is recovered by matching the saved copy-gate text; ERR is then scored as in
analyze_mln.py. Run from the repo root: python analysis/mln_copy_err.py
"""
import json, os, sys, unicodedata
import numpy as np, pandas as pd
M = "results/gpu/nb9_multilexnorm"
nfc = lambda s: unicodedata.normalize("NFC", s)


def choose(src, free, text):
    """Return per-token outputs (free span or source) whose non-empty join equals text, or None."""
    target = text.split()
    def rec(i, j):
        if i == len(src):
            return [] if j == len(target) else None
        for cand in dict.fromkeys([free[i], src[i]]):
            w = cand.split()
            if target[j:j + len(w)] == w:
                r = rec(i + 1, j + len(w))
                if r is not None:
                    return [cand] + r
        return None
    sys.setrecursionlimit(10000)
    return rec(0, 0)


for L in ("iden", "trde"):
    d = pd.read_csv(os.path.join(M, f"mln_{L}.csv"), keep_default_na=False); d = d[d.id.str.startswith(f"{L}_test_")]
    errs = {"free": [], "copy": []}
    for s in (13, 42, 2026):
        sp = json.load(open(os.path.join(M, f"test_spans_{L}_ctx_copy_s{s}.json"), encoding="utf-8"))
        tx = json.load(open(os.path.join(M, f"test_text_{L}_ctx_copy_s{s}_copy.json"), encoding="utf-8"))
        res = {"free": [0, 0], "copy": [0, 0]}; changed = unchanged_ok = 0; fail = 0
        tot_cor = {"free": 0, "copy": 0}; n_tok = 0; base = 0
        for c, tl, g in zip(d.id, d.token_labels, d.target_spans):
            src = [t.rsplit("/", 1)[0] for t in tl.split(" ")]; gold = json.loads(g)
            cp = choose(src, sp[c], tx[c])
            if cp is None:
                fail += 1; cp = sp[c]
            for k, out in (("free", sp[c]), ("copy", cp)):
                tot_cor[k] += sum(nfc(a) == nfc(b) for a, b in zip(gold, out))
            n_tok += len(gold); base += sum(a == b for a, b in zip(src, gold))
        for k in ("free", "copy"):
            errs[k].append(100 * (tot_cor[k] - base) / (n_tok - base))
        print(f"{L} seed {s}: ERR free {errs['free'][-1]:.2f}  copy {errs['copy'][-1]:.2f}  (unmatched sentences: {fail})")
    for k in ("free", "copy"):
        print(f"  {L} {k}: ERR {np.mean(errs[k]):.2f} ± {np.std(errs[k], ddof=1):.2f}")
