"""Data audit: verifies (or refutes) the corpus claims made in the draft and measures real annotation issues."""
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

from common import LABELS, RESULTS, SEEDS, bn_norm, flat, load, make_split, script_of

df = load()
R = {}

labs = Counter(flat(df.labels))
R["n_comments"] = len(df)
R["n_tokens"] = sum(labs.values())
R["label_counts"] = dict(labs)
R["label_share"] = {k: round(100 * v / R["n_tokens"], 2) for k, v in labs.items()}
R["code_mix_type"] = df.code_mix_type.value_counts().to_dict()
R["script_type"] = df.script_type.value_counts().to_dict()
R["aligned_comments"] = int(df.aligned.sum())
R["aligned_share"] = round(100 * df.aligned.mean(), 2)

# Is the target the identity on BAN/OTHER tokens? (draft's "Copy-Gate Invariance Theorem")
raw_same, norm_same, tot = Counter(), Counter(), Counter()
ban_changes = Counter()
for r in df[df.aligned].itertuples():
    raw_t = r.bengali_transliteration.split()
    for w, l, y, yr in zip(r.tokens, r.labels, r.targets, raw_t):
        tot[l] += 1
        raw_same[l] += w == yr
        norm_same[l] += bn_norm(w) == y
        if l == "BAN" and bn_norm(w) != y:
            ban_changes[(w, y)] += 1
R["identity_rate_raw"] = {l: round(100 * raw_same[l] / tot[l], 2) for l in LABELS}
R["identity_rate_unicode_normalized"] = {l: round(100 * norm_same[l] / tot[l], 2) for l in LABELS}
R["top_BAN_target_changes"] = [f"{a} -> {b} ({c})" for (a, b), c in ban_changes.most_common(25)]

# Label ambiguity of word types (lower-cased)
typ = defaultdict(Counter)
for ws, ls in zip(df.tokens, df.labels):
    for w, l in zip(ws, ls):
        typ[w.lower()][l] += 1
amb = {w: c for w, c in typ.items() if c["ENG"] > 0 and c["ROM_BAN"] > 0}
R["types_total"] = len(typ)
R["types_ENG_and_ROMBAN"] = len(amb)
R["tokens_in_ENG_ROMBAN_ambiguous_types"] = sum(sum(c.values()) for c in amb.values())
# Minority-label mass for frequent types: lower bound on inconsistent labels if each type had one true tag
freq = {w: c for w, c in typ.items() if sum(c.values()) >= 5}
minority = sum(sum(c.values()) - max(c.values()) for c in freq.values())
R["minority_label_tokens_freq_types(>=5)"] = minority
R["minority_label_share_of_freq_type_tokens"] = round(100 * minority / sum(sum(c.values()) for c in freq.values()), 2)
R["most_ambiguous_types"] = [
    f"{w}: {dict(c)}" for w, c in sorted(amb.items(), key=lambda x: -min(x[1]["ENG"], x[1]["ROM_BAN"]))[:30]
]

# Script vs label sanity
sl = Counter((script_of(w), l) for w, l in zip(flat(df.tokens), flat(df.labels)))
R["script_x_label"] = {f"{s}|{l}": c for (s, l), c in sorted(sl.items())}

# Near duplicates (letters only, lower-cased)
key = df.cleaned_text.map(lambda s: re.sub(r"[^a-zঀ-৿]", "", unicodedata.normalize("NFC", s.lower())))
R["near_duplicate_comments"] = int(key.duplicated(keep=False).sum())
R["near_duplicate_groups"] = int(key[key.duplicated(keep=False)].nunique())

# Splits used everywhere (also by the Kaggle notebook)
for s in SEEDS:
    d = make_split(df, s)
    R[f"split_seed{s}"] = d.split.value_counts().to_dict()
    tr = set(w.lower() for ws in d[d.split == "train"].tokens for w in ws)
    te = flat(d[d.split == "test"].tokens)
    R[f"test_oov_token_rate_seed{s}"] = round(100 * sum(w.lower() not in tr for w in te) / len(te), 2)

# Claims in the draft paper vs the released CSV
R["draft_claims_vs_csv"] = {
    "tokens": [120866, R["n_tokens"]],
    "BAN": [53973, labs["BAN"]],
    "OTHER": [9097, labs["OTHER"]],
    "BAN_target_identity_claimed_100pct": [100.0, R["identity_rate_unicode_normalized"]["BAN"]],
    "OTHER_target_identity_claimed_100pct": [100.0, R["identity_rate_unicode_normalized"]["OTHER"]],
    "second_annotator_labels_in_release": [True, any("annot" in c.lower() or "gold" in c.lower() for c in df.columns)],
}

with open(os.path.join(RESULTS, "01_data_audit.json"), "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
