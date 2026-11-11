"""Label-noise audit with confident learning (Northcutt et al., 2021) on out-of-fold CRF marginals.

Outputs
  results/04_noise_matrix.csv        estimated P(observed label | latent label), data-driven
  results/04_noise_candidates.csv    all flagged tokens, ranked, with context
  results/04_audit_sheet.csv         blind 400-token sheet (200 flagged + 200 random) for manual re-annotation
"""
import json
import os

import numpy as np
import pandas as pd
import sklearn_crfsuite
from sklearn.model_selection import KFold

from common import LABELS, RESULTS, flat, load

import importlib
lid = importlib.import_module("02_lid_baselines")

df = load()
probs = [None] * len(df)
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for k, (tri, tei) in enumerate(kf.split(df)):
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.05, max_iterations=200, all_possible_transitions=True)
    crf.fit([lid.sent_feats(df.tokens[i]) for i in tri], [df.labels[i] for i in tri])
    marg = crf.predict_marginals([lid.sent_feats(df.tokens[i]) for i in tei])
    for i, m in zip(tei, marg):
        probs[i] = np.array([[t.get(l, 0.0) for l in LABELS] for t in m])
    print("fold", k)

P = np.vstack(probs)
y = np.array([LABELS.index(l) for l in flat(df.labels)])
thr = np.array([P[y == j, j].mean() for j in range(len(LABELS))])
above = P >= thr
latent = np.where(above.any(1), np.where(above, P, -1).argmax(1), -1)

C = np.zeros((5, 5), int)
for yi, li in zip(y, latent):
    if li >= 0:
        C[li, yi] += 1
noise = pd.DataFrame(C / C.sum(1, keepdims=True), index=[f"true_{l}" for l in LABELS], columns=[f"obs_{l}" for l in LABELS])
noise.round(4).to_csv(os.path.join(RESULTS, "04_noise_matrix.csv"))

ci = np.repeat(np.arange(len(df)), df.tokens.map(len))
ti = np.concatenate([np.arange(n) for n in df.tokens.map(len)])
toks = flat(df.tokens)
flag = (latent >= 0) & (latent != y)


def ctx(c, t, w=4):
    ws = df.tokens[c]
    return " ".join(ws[max(0, t - w):t]) + " [[" + ws[t] + "]] " + " ".join(ws[t + 1:t + 1 + w])


cand = pd.DataFrame({
    "comment_id": df.id.values[ci], "token_idx": ti, "token": toks,
    "given_label": [LABELS[i] for i in y], "suggested_label": [LABELS[i] if i >= 0 else "" for i in latent],
    "self_confidence": P[np.arange(len(y)), y], "flagged": flag,
})
cand["context"] = [ctx(c, t) for c, t in zip(ci, ti)]
flagged = cand[cand.flagged].sort_values("self_confidence")
flagged.drop(columns="flagged").to_csv(os.path.join(RESULTS, "04_noise_candidates.csv"), index=False)

rng = np.random.RandomState(7)
sheet = pd.concat([
    flagged.sample(200, random_state=rng),
    cand[~cand.flagged & (cand.given_label != "BAN")].sample(200, random_state=rng),
]).sample(frac=1, random_state=rng)
key = sheet[["comment_id", "token_idx", "given_label", "suggested_label", "flagged"]]
key.to_csv(os.path.join(RESULTS, "04_audit_key_DO_NOT_SHOW_ANNOTATORS.csv"), index=False)
blind = sheet[["comment_id", "token_idx", "token", "context"]].assign(annotator1_label="", annotator2_label="", notes="")
blind.to_csv(os.path.join(RESULTS, "04_audit_sheet.csv"), index=False)

pairs = flagged.groupby(["given_label", "suggested_label"]).size().sort_values(ascending=False)
summary = {
    "n_tokens": int(len(y)),
    "flagged": int(flag.sum()),
    "flagged_share": round(100 * flag.mean(), 2),
    "per_class_threshold": dict(zip(LABELS, thr.round(3).tolist())),
    "flagged_by_pair": {f"{a}->{b}": int(n) for (a, b), n in pairs.items()},
    "top_flagged_tokens": flagged.token.str.lower().value_counts().head(30).to_dict(),
}
json.dump(summary, open(os.path.join(RESULTS, "04_noise_summary.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print(noise.round(3).to_string())
