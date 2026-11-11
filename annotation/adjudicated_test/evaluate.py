"""Scores the controlled systems against the silver and the adjudicated references of the 600-comment subset.

  python annotation/adjudicated_test/evaluate.py --agreed results/full_agreed.csv \
         --adjudication results/full_adjudication_done.xlsx --out annotation/adjudicated_test/results/final

Three references per comment:
  silver       the BanglaMix reference (as in the article);
  adjudicated  the adjudicated final target of every token, joined;
  variants     the adjudicated reference in which each token's target is replaced by the accepted form that matches
               the system's output for that token, if any (so any accepted spelling counts as correct).
Systems (split 42, saved test outputs): word_crf; matched word level, context-only CTT, and CTT (three training runs
each); ÚFAL-style re-encoding (one run); CTT with the copy gate (three runs; per-token copy decisions recovered from
the saved copy-gate texts). Reports CER, WER, EM for all / code-mixed / monolingual comments, two-sided paired
randomization tests (10,000 permutations), and how often the silver target was changed.
"""
import argparse, json, os
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
from common import ROOT, bn, join_spans, norm_target, norm_variants, sample_key, show, tokens_key

ap = argparse.ArgumentParser()
ap.add_argument("--agreed", required=True, help="<prefix>_agreed.csv from agreement.py")
ap.add_argument("--adjudication", required=True, help="the filled adjudication sheet (.xlsx)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
G = os.path.join(ROOT, "results", "gpu")
R_ITER = 10000
rng = np.random.RandomState(0)

K = tokens_key(); C = sample_key()
ag = pd.read_csv(args.agreed, keep_default_na=False, dtype=str)
ad = pd.read_excel(args.adjudication, sheet_name="adjudication", dtype=str, keep_default_na=False)
blank = ad[ad.final_target.str.strip() == ""]
if len(blank):
    raise SystemExit(f"{len(blank)} adjudication rows have no final_target (write <empty> for an empty target), "
                     f"e.g. {blank.item_id.tolist()[:5]}")
fin = pd.concat([ag[["item_id", "final_target", "final_variants"]], ad[["item_id", "final_target", "final_variants"]]])
if fin.item_id.duplicated().any():
    raise SystemExit("an item appears in both the agreed file and the adjudication sheet")
missing = set(K.item_id) - set(fin.item_id)
if missing:
    raise SystemExit(f"{len(missing)} tokens have no final target, e.g. {sorted(missing)[:5]}")
fin = fin.set_index("item_id").loc[K.item_id]
K["final"] = [norm_target(x) for x in fin.final_target]
K["variants"] = [norm_variants(v, t) for v, t in zip(fin.final_variants, K.final)]
K["changed"] = K.final != K.silver_target_raw
comments = list(C.comment_id)
cm = dict(zip(C.comment_id, C.code_mixed.astype(str).str.lower().eq("true")))
byc = {c: g for c, g in K.groupby("comment_id", sort=False)}
silver_ref = dict(zip(C.comment_id, C.silver_reference))
adj_ref = {c: join_spans(list(g.final)) for c, g in byc.items()}

# ---------------------------------------------------------------- system outputs (per-token spans)
def load(p):
    return json.load(open(os.path.join(G, p), encoding="utf-8"))


def copy_spans(free, text, src):
    """Per-token outputs with the copy gate: each token is either its free-decoding output or a copy of the source
    (with or without Bengali digits); the choice is recovered by matching the saved copy-gate text character by
    character, with the same joining rule as join_spans (GLUE spans attach without a space)."""
    from common import GLUE
    digits = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
    target = " ".join(text.split())
    memo = {}

    def rec(i, pos, empty):
        if i == len(src):
            return [] if pos == len(target) else None
        key = (i, pos, empty)
        if key in memo:
            return memo[key]
        res = None
        for cand in dict.fromkeys([free[i], bn(src[i]), bn(src[i]).translate(digits)]):
            if not cand:
                piece, nxt_empty = "", empty
            elif cand.startswith(GLUE):
                piece, nxt_empty = cand[len(GLUE):], False
            else:
                piece, nxt_empty = ("" if empty else " ") + cand, False
            if target.startswith(piece, pos):
                r = rec(i + 1, pos + len(piece), nxt_empty)
                if r is not None:
                    res = [cand] + r
                    break
        memo[key] = res
        return res
    return rec(0, 0, True)


RUNS = {"word_crf": ["nb5_ctt/stage1_seed42/test_spans_word_crf_s42.json"],
        "Word level (matched)": ["nb8_oracle_tags/test_spans_word_plain_align_s42.json",
                                 "nb11_train_seeds/test_spans_word_plain_align_ts1_s42.json",
                                 "nb11_train_seeds/test_spans_word_plain_align_ts2_s42.json"],
        "ÚFAL-style re-encoding": ["nb6_ufal/run1_ufal_s42_s13/test_spans_ufal_plain_s42.json"],
        "Context-only CTT": ["nb5_ctt/stage1_seed42/test_spans_ctx_plain_s42.json",
                             "nb11_train_seeds/test_spans_ctx_plain_ts1_s42.json",
                             "nb11_train_seeds/test_spans_ctx_plain_ts2_s42.json"],
        "CTT": ["nb5_ctt/stage2_three_splits/test_spans_ctx_plain_aks_heads_s42.json",
                "nb11_train_seeds/test_spans_ctt_ts1_s42.json", "nb11_train_seeds/test_spans_ctt_ts2_s42.json"]}
COPY_TEXT = ["nb5_ctt/stage2_three_splits/test_text_ctx_plain_aks_heads_s42_copy.json",
             "nb11_train_seeds/test_text_ctt_ts1_s42_copy.json", "nb11_train_seeds/test_text_ctt_ts2_s42_copy.json"]
OUT = {k: [load(p) for p in v] for k, v in RUNS.items()}
src = {c: list(g.source_token) for c, g in byc.items()}
cp_runs, unmatched = [], 0
HYP = {}          # (system, run) -> {comment: hypothesis text} where per-token spans cannot be recovered
for r_i, (free, tp) in enumerate(zip(OUT["CTT"], COPY_TEXT)):
    tx = load(tp); run = {}
    for c in comments:
        s = copy_spans(free[c], tx[c], src[c])
        if s is None:          # a repeat-loop word decoded to a different length: score the saved text itself
            unmatched += 1; s = free[c]
            HYP.setdefault(("CTT, copy gate", r_i), {})[c] = " ".join(tx[c].split())
        run[c] = s
    cp_runs.append(run)
OUT["CTT, copy gate"] = cp_runs


def scores(spans, override=None):
    """Per-comment (char errors, ref chars, word errors, ref words, exact) for the three references."""
    rows = []
    for c in comments:
        sp = spans[c]; hyp = (override or {}).get(c) or join_spans(sp)
        g = byc[c]
        var = [y if (bn(x) == y or bn(x) not in vs) else bn(x) for x, y, vs in zip(sp, g.final, g.variants)]
        var_ref = join_spans(var)
        r = {"comment_id": c}
        for name, ref in (("silver", silver_ref[c]), ("adjudicated", adj_ref[c]), ("variants", var_ref)):
            r[name + "_ce"] = Levenshtein.distance(hyp, ref); r[name + "_cn"] = len(ref)
            r[name + "_we"] = Levenshtein.distance(hyp.split(), ref.split()); r[name + "_wn"] = len(ref.split())
            r[name + "_em"] = hyp == ref
        rows.append(r)
    return pd.DataFrame(rows).set_index("comment_id")


S = {k: [scores(sp, HYP.get((k, i))) for i, sp in enumerate(v)] for k, v in OUT.items()}
SUB = {"all": comments, "code-mixed": [c for c in comments if cm[c]], "monolingual": [c for c in comments if not cm[c]]}
lines = []
say = lambda s="": (print(s), lines.append(s))
say(f"subset: {len(comments)} comments ({len(SUB['code-mixed'])} code-mixed, {len(SUB['monolingual'])} monolingual), "
    f"{len(K)} tokens; copy-gate comments scored from the saved text (spans not recoverable): {unmatched}")
say(f"silver targets changed by adjudication: {100 * K.changed.mean():.2f}% of tokens; comments with any change: "
    f"{100 * np.mean([byc[c].changed.any() for c in comments]):.1f}%; tokens with accepted variants: "
    f"{100 * np.mean([len(v) > 0 for v in K.variants]):.2f}%")
ch = K.assign(cm=K.comment_id.map(cm)).groupby("lid_tag").changed.mean().mul(100).round(2)
say("changed tokens by LID tag (%): " + ch.to_dict().__repr__())

rows = []
for sub, ids in SUB.items():
    for k, runs in S.items():
        rec = dict(subset=sub, system=k, runs=len(runs))
        for ref in ("silver", "adjudicated", "variants"):
            m = np.array([[100 * x.loc[ids, ref + "_ce"].sum() / x.loc[ids, ref + "_cn"].sum(),
                           100 * x.loc[ids, ref + "_we"].sum() / x.loc[ids, ref + "_wn"].sum(),
                           100 * x.loc[ids, ref + "_em"].mean()] for x in runs])
            for j, met in enumerate(("CER", "WER", "EM")):
                rec[f"{ref}_{met}"] = round(m[:, j].mean(), 2)
                if len(runs) > 1:
                    rec[f"{ref}_{met}_sd"] = round(m[:, j].std(ddof=1), 2)
        rows.append(rec)
TAB = pd.DataFrame(rows)
TAB.to_csv(args.out + "_scores.csv", index=False)
say("\nCER / WER / EM (mean over runs) against the silver, adjudicated, and variant-aware references")
say(TAB[["subset", "system", "runs"] + [f"{r}_{m}" for r in ("silver", "adjudicated", "variants") for m in ("CER", "WER", "EM")]]
    .to_string(index=False))


def randtest(pa, pb, ids, ref):
    e = np.concatenate([x.loc[ids, ref + "_ce"].values - y.loc[ids, ref + "_ce"].values for x, y in zip(pa, pb)]).astype(float)
    L = sum(x.loc[ids, ref + "_cn"].sum() for x in pa)
    d = 100 * e.sum() / L
    null = 100 * (rng.choice([-1.0, 1.0], size=(R_ITER, len(e))) @ e) / L
    return d, (np.sum(np.abs(null) >= abs(d) - 1e-12) + 1) / (R_ITER + 1)


say("\ntwo-sided paired randomization tests, dCER (negative favours the first system)")
pairs = [("CTT", "Word level (matched)", "3 runs pooled"), ("Context-only CTT", "Word level (matched)", "3 runs pooled"),
         ("CTT", "ÚFAL-style re-encoding", "original CTT run"), ("Context-only CTT", "ÚFAL-style re-encoding", "original run"),
         ("CTT, copy gate", "CTT", "3 runs pooled"), ("CTT", "word_crf", "original CTT run")]
for sub in ("all", "code-mixed"):
    for a, b, how in pairs:
        pa, pb = S[a], S[b]
        if len(pb) == 1:
            pa = pa[:1]
        out = []
        for ref in ("silver", "adjudicated", "variants"):
            d, p = randtest(pa, pb, SUB[sub], ref)
            out.append(f"{ref} {d:+.2f} (p={p:.4f})")
        say(f"  {sub:10s} {a} vs {b} [{how}]: " + "; ".join(out))
open(args.out + "_report.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
K.assign(final=[show(t) for t in K.final], variants=["|".join(v) for v in K.variants])[
    ["item_id", "comment_id", "token_no", "source_token", "lid_tag", "silver_target", "final", "variants", "changed"]
].to_csv(args.out + "_references.csv", index=False, encoding="utf-8")
print(f"\nwrote {args.out}_report.txt, {args.out}_scores.csv, {args.out}_references.csv")
