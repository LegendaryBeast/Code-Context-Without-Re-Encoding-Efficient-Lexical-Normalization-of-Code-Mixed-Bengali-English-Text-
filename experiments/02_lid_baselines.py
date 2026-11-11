"""Token-level LID baselines (classical ML) on three duplicate-aware stratified splits.

Models: script rule, train lexicon, char n-gram LR (token only), LR + context window, linear-chain CRF.
"""
import json
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import LABELS, RESULTS, SEEDS, cluster_bootstrap, flat, lexicon, lid_scores, load, make_split, script_of

EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF☀-➿]")


def rule(w):
    s = script_of(w)
    return {"bn": "BAN", "lat": "ENG", "none": "OTHER", "mix": "UNK"}[s]


def tok_feats(w, ngrams=True):
    lw = w.lower()
    f = {
        "bias": 1.0,
        "w=" + lw: 1.0,
        "script=" + script_of(w): 1.0,
        "len=" + str(min(len(w), 10)): 1.0,
        "title": float(w.istitle()),
        "upper": float(w.isupper() and len(w) > 1),
        "digit": float(any(c.isdigit() for c in w)),
        "emoji": float(bool(EMOJI_RE.search(w))),
        "elong": float(bool(re.search(r"(.)\1\1", lw))),
    }
    if ngrams:
        p = f"<{lw}>"
        for n in (1, 2, 3, 4):
            for i in range(len(p) - n + 1):
                f[f"g{n}={p[i:i + n]}"] = 1.0
    return f


def sent_feats(ws, context=2):
    base = [tok_feats(w) for w in ws]
    out = []
    for i in range(len(ws)):
        f = dict(base[i])
        for d in range(-context, context + 1):
            if d == 0:
                continue
            j = i + d
            if 0 <= j < len(ws):
                f[f"{d}:w=" + ws[j].lower()] = 1.0
                f[f"{d}:script=" + script_of(ws[j])] = 1.0
            else:
                f[f"{d}:pad"] = 1.0
        out.append(f)
    return out


def run_seed(df, seed):
    d = make_split(df, seed)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    trw, trl = list(tr.tokens), list(tr.labels)
    tew, tel = list(te.tokens), list(te.labels)
    train_vocab = set(w.lower() for w in flat(trw))
    preds = {}

    preds["Script rule"] = [[rule(w) for w in ws] for ws in tew]

    lex = lexicon((w.lower(), l) for w, l in zip(flat(trw), flat(trl)))
    preds["Train lexicon + rule"] = [[lex.get(w.lower(), rule(w)) for w in ws] for ws in tew]

    t = time.time()
    dv = DictVectorizer()
    Xtr = dv.fit_transform([tok_feats(w) for w in flat(trw)])
    lr = LogisticRegression(C=5.0, max_iter=3000)
    lr.fit(Xtr, flat(trl))
    p = lr.predict(dv.transform([tok_feats(w) for w in flat(tew)]))
    preds["Char n-gram LR (token)"] = np.split(p, np.cumsum([len(x) for x in tew])[:-1])
    print("  LR token", round(time.time() - t, 1), "s")

    t = time.time()
    dv2 = DictVectorizer()
    Xtr = dv2.fit_transform(flat(sent_feats(ws) for ws in trw))
    lr2 = LogisticRegression(C=5.0, max_iter=3000)
    lr2.fit(Xtr, flat(trl))
    p = lr2.predict(dv2.transform(flat(sent_feats(ws) for ws in tew)))
    preds["Char n-gram LR + context(±2)"] = np.split(p, np.cumsum([len(x) for x in tew])[:-1])
    print("  LR ctx", round(time.time() - t, 1), "s")

    t = time.time()
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.05, max_iterations=200, all_possible_transitions=True)
    crf.fit([sent_feats(ws) for ws in trw], trl)
    preds["CRF (char n-gram + context)"] = crf.predict([sent_feats(ws) for ws in tew])
    print("  CRF", round(time.time() - t, 1), "s")

    # Save CRF predictions for train/val/test (used by translit pipeline and the noise audit)
    allp = {}
    for part in (tr, va, te):
        for i, ws in zip(part.id, crf.predict([sent_feats(ws) for ws in part.tokens])):
            allp[i] = list(ws)
    with open(os.path.join(RESULTS, f"crf_pred_seed{seed}.pkl"), "wb") as f:
        pickle.dump(allp, f)

    rows, extra = [], {}
    gold_flat = flat(tel)
    oov = np.array([w.lower() not in train_vocab for w in flat(tew)])
    for name, pr in preds.items():
        pr = [list(x) for x in pr]
        s = lid_scores(gold_flat, flat(pr))
        pf = np.array(flat(pr))
        gf = np.array(gold_flat)
        s["OOV_Acc"] = 100 * (pf[oov] == gf[oov]).mean()
        s["IV_Acc"] = 100 * (pf[~oov] == gf[~oov]).mean()
        latin = np.isin(gf, ["ENG", "ROM_BAN"])
        s["ENG/ROM_BAN Acc"] = 100 * (pf[latin] == gf[latin]).mean()
        s.update(model=name, seed=seed)
        rows.append(s)
        # per code-mix stratum token accuracy (macro-F1 is ill-defined when strata lack classes)
        for cmt, g in te.assign(pred=pr).groupby("code_mix_type"):
            extra[(name, cmt)] = 100 * np.mean(np.array(flat(g.labels)) == np.array(flat(g.pred)))
    return rows, preds, tel, extra


if __name__ == "__main__":
    df = load()
    all_rows, strata = [], []
    for seed in SEEDS:
        print("seed", seed)
        rows, preds, tel, extra = run_seed(df, seed)
        all_rows += rows
        strata += [dict(model=m, stratum=c, seed=seed, Acc=v) for (m, c), v in extra.items()]
        if seed == 42:
            mf = lambda a, b: 100 * f1_score(a, b, labels=LABELS, average="macro", zero_division=0)
            ci = {}
            for name, pr in preds.items():
                pr = [list(x) for x in pr]
                other = [list(x) for x in preds["Char n-gram LR + context(±2)"]]
                ci[name] = cluster_bootstrap(tel, pr, mf, n=1000, other_pred=other if "CRF" in name else None)
            json.dump(ci, open(os.path.join(RESULTS, "02_lid_bootstrap_seed42.json"), "w"), indent=1)
            print(json.dumps(ci, indent=1))

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(RESULTS, "02_lid_all_seeds.csv"), index=False)
    cols = ["Acc", "MacroF1", "Kappa"] + [f"F1_{l}" for l in LABELS] + ["ENG/ROM_BAN Acc", "IV_Acc", "OOV_Acc"]
    agg = res.groupby("model", sort=False)[cols].agg(["mean", "std"])
    summ = pd.DataFrame({c: [f"{m:.2f} ± {s:.2f}" for m, s in zip(agg[c]["mean"], agg[c]["std"])] for c in cols}, index=agg.index)
    summ.to_csv(os.path.join(RESULTS, "02_lid_summary.csv"))
    st = pd.DataFrame(strata).groupby(["model", "stratum"], sort=False).Acc.mean().unstack().round(2)
    st.to_csv(os.path.join(RESULTS, "02_lid_by_stratum.csv"))
    pd.set_option("display.width", 250)
    print(summ.to_string())
    print(st.to_string())
