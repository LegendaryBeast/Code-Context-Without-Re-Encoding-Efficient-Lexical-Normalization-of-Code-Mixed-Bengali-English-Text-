"""Extra real analyses used by the paper figures.

  - CRF confusion matrix (seed-42 test)
  - CRF learning curve (10/25/50/100% of the seed-42 training comments)
  - Normalisation CER by code-mixing complexity C(c) for the lexicon systems (3 splits)
"""
import importlib
import json
import os
import pickle

import numpy as np
import sklearn_crfsuite
from sklearn.metrics import confusion_matrix, f1_score

from common import LABELS, RESULTS, SEEDS, bn_norm, cer_wer, flat, load, make_split

lid = importlib.import_module("02_lid_baselines")
tl = importlib.import_module("03_translit_baselines")
df = load()
out = {}

# ---- confusion matrix of the CRF on seed-42 test
d = make_split(df, 42)
te = d[d.split == "test"]
crf_pred = pickle.load(open(os.path.join(RESULTS, "crf_pred_seed42.pkl"), "rb"))
cm = confusion_matrix(flat(te.labels), flat(crf_pred[i] for i in te.id), labels=LABELS)
out["crf_confusion_seed42"] = cm.tolist()

# ---- learning curve
tr = d[d.split == "train"]
rng = np.random.RandomState(0)
order = rng.permutation(len(tr))
curve = []
for frac in (0.1, 0.25, 0.5, 1.0):
    sub = tr.iloc[order[: int(frac * len(tr))]]
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.05, max_iterations=200, all_possible_transitions=True)
    crf.fit([lid.sent_feats(ws) for ws in sub.tokens], list(sub.labels))
    p = crf.predict([lid.sent_feats(ws) for ws in te.tokens])
    curve.append({"frac": frac, "n_comments": len(sub),
                  "MacroF1": 100 * f1_score(flat(te.labels), flat(p), labels=LABELS, average="macro"),
                  "Acc": 100 * float(np.mean(np.array(flat(te.labels)) == np.array(flat(p))))})
    print("learning curve", curve[-1])
out["crf_learning_curve_seed42"] = curve


# ---- CER by complexity C(c) = number of {BAN, ENG, ROM_BAN} present
def complexity(labels):
    return len({l for l in labels if l in ("BAN", "ENG", "ROM_BAN")})


by_c = {}
for seed in SEEDS:
    d = make_split(df, seed)
    crf_pred = pickle.load(open(os.path.join(RESULTS, f"crf_pred_seed{seed}.pkl"), "rb"))
    tr = d[(d.split == "train") & d.aligned]
    te = d[d.split == "test"]
    lexf = tl.Lexicon(tr, by_label=False, fuzzy=True)
    lexl = tl.Lexicon(tr, by_label=True, fuzzy=True)
    systems = {
        "Copy": lambda w, l, p: bn_norm(w),
        "Word lexicon + fuzzy": lambda w, l, p: lexf(w, l),
        "LID lexicon + gate (CRF)": lambda w, l, p: bn_norm(w) if p in ("BAN", "OTHER") else lexl(w, p),
    }
    cs = te.labels.map(complexity).values
    refs = [" ".join(t) for t in te.targets]
    for name, f in systems.items():
        hyps = [" ".join(f(w, l, p) for w, l, p in zip(r.tokens, r.labels, crf_pred[r.id])) for r in te.itertuples()]
        for c in range(4):
            m = cs == c
            if m.any():
                s = cer_wer([h for h, k in zip(hyps, m) if k], [r for r, k in zip(refs, m) if k])
                by_c.setdefault(name, {}).setdefault(str(c), []).append(s["CER"])
    print("seed", seed, "done")
out["cer_by_complexity"] = {n: {c: [float(np.mean(v)), float(np.std(v))] for c, v in cc.items()} for n, cc in by_c.items()}
out["complexity_counts"] = {str(c): int((df.labels.map(complexity) == c).sum()) for c in range(4)}

json.dump(out, open(os.path.join(RESULTS, "05_extra.json"), "w"), indent=1)
print(json.dumps(out["cer_by_complexity"], indent=1))
