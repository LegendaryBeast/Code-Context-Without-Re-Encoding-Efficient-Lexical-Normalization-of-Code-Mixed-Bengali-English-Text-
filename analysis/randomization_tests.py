"""Paired approximate randomization tests (two-sided) on saved predictions; no new training.

For every comparison of two systems on the same test comments, the outputs of the two systems are swapped
independently per comment (per comment and run when runs are pooled) under the null hypothesis that they are
exchangeable; the p-value is (#{|d*| >= |d|} + 1) / (R + 1). Also recomputes the LID-tag ablation without the
repetition guard, and CER/WER/EM of the three split-42 training runs.

Run from the repo root:  python analysis/randomization_tests.py
"""
import contextlib, glob, io, json, os, pickle, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis")); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import review_analyses as RA          # loaders, guard, join_spans, bn, references
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import f1_score

R_ITER = 10000
rng = np.random.RandomState(0)
G = os.path.join(ROOT, "results", "gpu")
df, ref, toks = RA.df, RA.df["ref"], RA.df["tokens"]


def spans_file(path):
    return json.load(open(os.path.join(G, path), encoding="utf-8"))


def comment_errors(sp, ids, guarded=False):
    ce, ln, we, wn, em = [], [], [], [], []
    for c in ids:
        s = [RA.guard(x, w) for x, w in zip(sp[c], toks[c])] if guarded else sp[c]
        h, r = RA.join_spans(s), ref[c]
        ce.append(Levenshtein.distance(h, r)); ln.append(len(r))
        we.append(Levenshtein.distance(h.split(), r.split())); wn.append(len(r.split())); em.append(h == r)
    return dict(ce=np.array(ce), ln=np.array(ln), we=np.array(we), wn=np.array(wn), em=np.array(em))


def text_errors(path, ids):
    tx = json.load(open(os.path.join(G, path), encoding="utf-8"))
    ce = np.array([Levenshtein.distance(" ".join(RA.bn(tx[c]).split()), ref[c]) for c in ids])
    return dict(ce=ce, ln=np.array([len(ref[c]) for c in ids]))


def rand_test(pairs):
    """pairs: list of (errA, errB, len) arrays, one per run/split; returns dCER, two-sided p."""
    e = np.concatenate([a - b for a, b, _ in pairs]).astype(float); L = sum(l.sum() for _, _, l in pairs)
    d = 100 * e.sum() / L
    signs = rng.choice([-1.0, 1.0], size=(R_ITER, len(e)))
    null = 100 * (signs @ e) / L
    return d, (np.sum(np.abs(null) >= abs(d) - 1e-12) + 1) / (R_ITER + 1)


def show(name, pairs):
    d, p = rand_test(pairs)
    print(f"  {name:48s} dCER {d:+.2f}  p(two-sided) = {p:.4f}")
    return d, p


S = RA.SEEDS
E = {(v, s): comment_errors(RA.SPANS[(v, s)], RA.TEST[s]) for v in RA.SYSTEMS for s in S}
print("== A. Table 8 top: three splits pooled, unguarded")
for a, b in [("ctx_plain_aks_heads", "word_crf"), ("ctx_plain", "word_crf"), ("ctx_plain_aks_heads", "ufal_plain"),
             ("ctx_plain", "ufal_plain"), ("ctx_plain_aks_heads", "ctx_plain")]:
    show(f"{a} vs {b}", [(E[(a, s)]["ce"], E[(b, s)]["ce"], E[(a, s)]["ln"]) for s in S])
CP = {s: text_errors(f"nb5_ctt/{'stage1_seed42' if os.path.exists(os.path.join(G, 'nb5_ctt/stage1_seed42', f'test_text_ctx_plain_aks_heads_s{s}_copy.json')) else 'stage2_three_splits'}/test_text_ctx_plain_aks_heads_s{s}_copy.json", RA.TEST[s]) for s in S}
FR = {s: text_errors(f"nb5_ctt/{'stage1_seed42' if os.path.exists(os.path.join(G, 'nb5_ctt/stage1_seed42', f'test_text_ctx_plain_aks_heads_s{s}_free.json')) else 'stage2_three_splits'}/test_text_ctx_plain_aks_heads_s{s}_free.json", RA.TEST[s]) for s in S}
# both sides read from the saved decoded texts (free text equals the joined spans except 2 comments of split 2026)
show("copy gate vs free decoding (CTT)", [(CP[s]["ce"], FR[s]["ce"], CP[s]["ln"]) for s in S])

E8 = {s: comment_errors(spans_file(f"nb5_ctt/stage3a_8_epochs/test_spans_ctx_plain_aks_heads_e8_s{s}.json"), RA.TEST[s]) for s in S}
show("8 epochs vs 5 epochs (CTT)", [(E8[s]["ce"], E[("ctx_plain_aks_heads", s)]["ce"], E8[s]["ln"]) for s in S])
EB = {s: comment_errors(spans_file(f"nb5_ctt/stage3b_byt5_base/test_spans_ctx_plain_aks_heads_byt5-base_s{s}.json"), RA.TEST[s]) for s in (13, 42)}
show("ByT5-base vs ByT5-small (CTT, splits 13+42)", [(EB[s]["ce"], E[("ctx_plain_aks_heads", s)]["ce"], EB[s]["ln"]) for s in (13, 42)])

print("\n== B. Table 8 bottom: split 42, three training runs pooled, unguarded")
ids42 = RA.TEST[42]
runs = {"CTT": ["nb5_ctt/stage2_three_splits/test_spans_ctx_plain_aks_heads_s42.json",
                "nb11_train_seeds/test_spans_ctt_ts1_s42.json", "nb11_train_seeds/test_spans_ctt_ts2_s42.json"],
        "ctx": ["nb5_ctt/stage1_seed42/test_spans_ctx_plain_s42.json",
                "nb11_train_seeds/test_spans_ctx_plain_ts1_s42.json", "nb11_train_seeds/test_spans_ctx_plain_ts2_s42.json"],
        "word": ["nb8_oracle_tags/test_spans_word_plain_align_s42.json",
                 "nb11_train_seeds/test_spans_word_plain_align_ts1_s42.json",
                 "nb11_train_seeds/test_spans_word_plain_align_ts2_s42.json"]}
RE = {k: [comment_errors(spans_file(p), ids42) for p in v] for k, v in runs.items()}
for a, b in [("CTT", "word"), ("ctx", "word"), ("CTT", "ctx")]:
    show(f"{a} vs {b} (3 runs)", [(x["ce"], y["ce"], x["ln"]) for x, y in zip(RE[a], RE[b])])
for a, b in [("CTT", "word"), ("ctx", "word")]:   # WER and EM differences, same permutation scheme
    e = np.concatenate([x["we"] - y["we"] for x, y in zip(RE[a], RE[b])]).astype(float); W = sum(x["wn"].sum() for x in RE[a])
    d = 100 * e.sum() / W; null = 100 * (rng.choice([-1.0, 1.0], size=(R_ITER, len(e))) @ e) / W
    print(f"  {a} vs {b} (3 runs) dWER {d:+.2f}  p(two-sided) = {(np.sum(np.abs(null) >= abs(d) - 1e-12) + 1) / (R_ITER + 1):.4f}")
    e = np.concatenate([x["em"].astype(float) - y["em"] for x, y in zip(RE[a], RE[b])])
    d = 100 * e.sum() / len(e); null = 100 * (rng.choice([-1.0, 1.0], size=(R_ITER, len(e))) @ e) / len(e)
    print(f"  {a} vs {b} (3 runs) dEM  {d:+.2f}  p(two-sided) = {(np.sum(np.abs(null) >= abs(d) - 1e-12) + 1) / (R_ITER + 1):.4f}")
print("  per-run CER / WER / EM (split 42):")
for k, v in RE.items():
    cer = [100 * x["ce"].sum() / x["ln"].sum() for x in v]; wer = [100 * x["we"].sum() / x["wn"].sum() for x in v]
    em = [100 * x["em"].mean() for x in v]
    print(f"    {k:5s} CER {np.mean(cer):.2f}±{np.std(cer, ddof=1):.2f}  WER {np.mean(wer):.2f}±{np.std(wer, ddof=1):.2f}  "
          f"EM {np.mean(em):.2f}±{np.std(em, ddof=1):.2f}   runs CER {[round(c, 2) for c in cer]}")

print("\n== C. Ablations on split 42 (single runs), unguarded and guarded")
st1 = "nb5_ctt/stage1_seed42/test_spans_{}_s42.json"
A = {v: spans_file(st1.format(v)) for v in ["word_crf", "word_crf_align", "ctx_crf", "ctx_plain", "ctx_joint"]}
for a, b, lab in [("word_crf_align", "word_crf", "recover unaligned data"), ("ctx_crf", "word_crf_align", "add context"),
                  ("ctx_plain", "ctx_crf", "remove CRF tags"), ("ctx_plain", "ctx_joint", "remove joint LID")]:
    for g in (False, True):
        ea, eb = comment_errors(A[a], ids42, g), comment_errors(A[b], ids42, g)
        show(f"{lab:24s} {'guarded' if g else 'raw    '}", [(ea["ce"], eb["ce"], ea["ln"])])

print("\n== D. Tag-source table, unguarded and guarded CER (split 42)")
T = {"word/none": ["nb8_oracle_tags/test_spans_word_plain_align_s42.json"],
     "word/CRF": [st1.format("word_crf_align"), "nb8_oracle_tags/test_spans_word_crf_align_rerun_s42.json"],
     "word/annotated": ["nb8_oracle_tags/test_spans_word_crf_align_rerun_goldtags_s42.json"],
     "context/none": [st1.format("ctx_plain"), "nb6_ufal/run2_ufal_s2026_ctxplain_s42/test_spans_ctx_plain_rerun_s42.json"],
     "context/CRF": [st1.format("ctx_crf"), "nb8_oracle_tags/test_spans_ctx_crf_rerun_s42.json"],
     "context/annotated": ["nb8_oracle_tags/test_spans_ctx_crf_rerun_goldtags_s42.json"]}
for k, v in T.items():
    for g in (False, True):
        c = [100 * (x := comment_errors(spans_file(p), ids42, g))["ce"].sum() / x["ln"].sum() for p in v]
        print(f"  {k:18s} {'guarded' if g else 'raw    '} CER {np.mean(c):.2f} (runs {[round(z, 2) for z in c]})")

print("\n== E. MultiLexNorm: ERR difference, seeds pooled, two-sided randomization")
M = os.path.join(G, "nb9_multilexnorm")
for L in ("iden", "trde"):
    d = pd.read_csv(os.path.join(M, f"mln_{L}.csv"), keep_default_na=False); d = d[d.id.str.startswith(f"{L}_test_")]
    data = [(c, [t.rsplit("/", 1)[0] for t in tl.split(" ")], json.loads(s)) for c, tl, s in zip(d.id, d.token_labels, d.target_spans)]
    def per(sysn, seed):
        sp = json.load(open(os.path.join(M, f"test_spans_{L}_{sysn}_s{seed}.json"), encoding="utf-8"))
        cor = np.array([sum(RA.unicodedata.normalize("NFC", a) == RA.unicodedata.normalize("NFC", b) for a, b in zip(g, sp[c])) for c, _, g in data])
        chg = np.array([sum(r != g_ for r, g_ in zip(t, g)) for _, t, g in data])
        return cor, chg
    for a, b in [("ctx_copy", "ufal"), ("ctx", "word"), ("ctx_copy", "ctx")]:
        pa = [per(a, s) for s in (13, 42, 2026)]; pb = [per(b, s) for s in (13, 42, 2026)]
        e = np.concatenate([x[0] - y[0] for x, y in zip(pa, pb)]).astype(float); C = sum(x[1].sum() for x in pa)
        dd = 100 * e.sum() / C
        null = 100 * (rng.choice([-1.0, 1.0], size=(R_ITER, len(e))) @ e) / C
        p = (np.sum(np.abs(null) >= abs(dd) - 1e-12) + 1) / (R_ITER + 1)
        print(f"  {L} {a:8s} vs {b:5s} dERR {dd:+.2f}  p(two-sided) = {p:.4f}")

print("\n== F. LID: XLM-R vs CRF macro-F1, three splits, two-sided randomization (2,000 iterations)")
LAB = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]
diffs, nulls = [], np.zeros(2000)
for s in S:
    sp = pd.read_csv(os.path.join(ROOT, "data", "splits", f"split_seed{s}.csv")); te = sp[sp.split == "test"].id.tolist()
    X = json.load(open(os.path.join(G, "nb1_lid_transformers", f"lid_test_pred_xlm-roberta-base_s{s}_ce.json")))
    C = pickle.load(open(os.path.join(ROOT, "results", "cpu", f"crf_pred_seed{s}.pkl"), "rb"))
    gold = [df.at[c, "labels"] for c in te]; xa = [X[c] for c in te]; cb = [C[c] for c in te]
    fl = lambda Ls: [t for l in Ls for t in l]
    f = lambda p: f1_score(fl(gold), fl(p), labels=LAB, average="macro", zero_division=0)
    diffs.append(100 * (f(xa) - f(cb)))
    for k in range(2000):
        sw = rng.rand(len(te)) < 0.5
        pa = [b if w else a for a, b, w in zip(xa, cb, sw)]; pb = [a if w else b for a, b, w in zip(xa, cb, sw)]
        nulls[k] += 100 * (f(pa) - f(pb)) / len(S)
dm = np.mean(diffs)
print(f"  XLM-R - CRF macro-F1 {dm:+.2f}  p(two-sided) = {(np.sum(np.abs(nulls) >= abs(dm)) + 1) / 2001:.4f}")
