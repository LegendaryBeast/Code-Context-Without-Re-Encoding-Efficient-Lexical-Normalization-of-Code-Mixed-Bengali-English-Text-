"""Analyses that need no new training (existing test spans only).

  A. Unguarded vs guarded CER for the systems in the CTT code base, and paired comment-clustered bootstrap tests
     on both (does any conclusion depend on the repetition guard?).
  B. Chunking: how often the 160-byte chunks cut a comment, token accuracy for words next to a chunk boundary vs
     elsewhere, and CER by comment length.
  C. Automatic taxonomy of CTT's token errors (same rules as analysis/nb7_ctt_error_analysis.ipynb, Part 1).

Run from the repo root:  python analysis/review_analyses.py
"""
import glob, json, os, re, sys, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "src"))
from banglish_align import join_spans  # noqa: E402

try:
    from rapidfuzz.distance import Levenshtein
    lev = Levenshtein.distance
except ImportError:
    def lev(a, b):
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

GPU = os.path.join(ROOT, "results", "gpu")
DIRS = sorted(glob.glob(os.path.join(GPU, "nb6_ufal", "run*"))) + [
    os.path.join(GPU, "nb5_ctt", d) for d in ("stage1_seed42", "stage2_three_splits")]
SYSTEMS = {"word_crf": "Word level", "ctx_plain": "Context only", "ctx_plain_aks_heads": "CTT", "ufal_plain": "ÚFAL-style"}
SEEDS, BUDGET, N_BOOT = [13, 42, 2026], 160, 2000
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
RUN = re.compile(r"(.{1,4}?)\1+", re.S)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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


def chunk_words(words, budget):   # identical to kaggle/src/ctt.py
    chunks, cur, size = [], [], 0
    for i, w in enumerate(words):
        b = len(w.encode("utf-8")) + (1 if cur else 0)
        if cur and size + b > budget:
            chunks.append(cur)
            cur, size, b = [], 0, len(w.encode("utf-8"))
        cur.append(i)
        size += b
    if cur:
        chunks.append(cur)
    return chunks


def find(name):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


df = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv"))
df["tokens"] = df.token_labels.map(lambda tl: [t.rsplit("/", 1)[0] for t in tl.split(" ")])
df["labels"] = df.token_labels.map(lambda tl: [t.rsplit("/", 1)[1] for t in tl.split(" ")])
df["ref"] = df.bengali_transliteration.map(lambda s: " ".join(bn(s).split()))
df = df.set_index("id")
SPANS = {(v, s): json.load(open(find(f"test_spans_{v}_s{s}.json"), encoding="utf-8")) for v in SYSTEMS for s in SEEDS}
TEST = {s: sorted(SPANS[("ctx_plain_aks_heads", s)]) for s in SEEDS}
for (v, s), sp in SPANS.items():
    assert sorted(sp) == TEST[s], f"{v} s{s} covers different test comments"


def comment_err(v, s, guarded):
    ce, ln = [], []
    for c in TEST[s]:
        toks = df.at[c, "tokens"]
        spans = [guard(x, w) for x, w in zip(SPANS[(v, s)][c], toks)] if guarded else SPANS[(v, s)][c]
        hyp, ref = join_spans(spans), df.at[c, "ref"]
        ce.append(lev(hyp, ref)); ln.append(len(ref))
    return np.array(ce), np.array(ln)


E = {(v, s, g): comment_err(v, s, g) for v in SYSTEMS for s in SEEDS for g in (False, True)}

print("== A. test CER without and with the repetition guard (mean ± std over splits 13, 42, 2026)")
rows = []
for v, name in SYSTEMS.items():
    raw = [100 * E[(v, s, False)][0].sum() / E[(v, s, False)][1].sum() for s in SEEDS]
    grd = [100 * E[(v, s, True)][0].sum() / E[(v, s, True)][1].sum() for s in SEEDS]
    rows.append(dict(system=name, raw=f"{np.mean(raw):.2f} ± {np.std(raw, ddof=1):.2f}",
                     guarded=f"{np.mean(grd):.2f} ± {np.std(grd, ddof=1):.2f}"))
print(pd.DataFrame(rows).to_string(index=False))

print(f"\npaired comment-clustered bootstrap, splits pooled ({N_BOOT} resamples; dCER < 0: first better; p = share not better)")
rng = np.random.RandomState(0)
for a, b in [("ctx_plain_aks_heads", "word_crf"), ("ctx_plain", "word_crf"), ("ctx_plain_aks_heads", "ctx_plain"),
             ("ctx_plain_aks_heads", "ufal_plain"), ("ctx_plain", "ufal_plain")]:
    out = []
    for g in (False, True):
        ea = np.concatenate([E[(a, s, g)][0] for s in SEEDS]); eb = np.concatenate([E[(b, s, g)][0] for s in SEEDS])
        ln = np.concatenate([E[(a, s, g)][1] for s in SEEDS])
        sizes = [len(TEST[s]) for s in SEEDS]; off = np.cumsum([0] + sizes)
        d = np.array([100 * (ea[k].sum() - eb[k].sum()) / ln[k].sum() for k in
                      (np.concatenate([o + rng.randint(0, n, n) for o, n in zip(off, sizes)]) for _ in range(N_BOOT))])
        out.append(f"{'guarded' if g else 'raw'} {100 * (ea.sum() - eb.sum()) / ln.sum():+.2f} "
                   f"[{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}] p={np.mean(d >= 0):.3f}")
    print(f"  {SYSTEMS[a]:12s} vs {SYSTEMS[b]:12s}  " + "   ".join(out))

print(f"\n== B. chunking ({BUDGET}-byte chunks, words never split)")
nch = df.tokens.map(lambda t: len(chunk_words(t, BUDGET)))
ntok = df.tokens.map(len)
print(f"all comments: {nch.mean():.2f} chunks per comment; {100 * (nch > 1).mean():.1f}% have >1 chunk, "
      f"holding {100 * ntok[nch > 1].sum() / ntok.sum():.1f}% of all words")
tok = []
for s in SEEDS:
    for c in TEST[s]:
        toks, labels, ref = df.at[c, "tokens"], df.at[c, "labels"], df.at[c, "ref"].split()
        if len(toks) != len(ref):
            continue                                   # token accuracy on aligned comments, as in Table 4
        ch = chunk_words(toks, BUDGET)
        where = {}
        for k, idx in enumerate(ch):
            for j, i in enumerate(idx):
                cut_left, cut_right = (j == 0 and k > 0), (j == len(idx) - 1 and k < len(ch) - 1)
                where[i] = "boundary" if (cut_left or cut_right) else ("multi-chunk, inside" if len(ch) > 1 else "single chunk")
        for i, (w, l, y) in enumerate(zip(toks, labels, ref)):
            row = dict(split=s, comment=c, position=where[i], tag=l, n_tok=len(toks))
            for v in SYSTEMS:
                row[v] = bn(guard(SPANS[(v, s)][c][i], w)) == y
            tok.append(row)
T = pd.DataFrame(tok)
acc = T.groupby("position")[list(SYSTEMS)].mean().mul(100).round(2).rename(columns=SYSTEMS)
acc["tokens"] = T.groupby("position").size()
acc["CTT - word level"] = (acc["CTT"] - acc["Word level"]).round(2)
print("token accuracy (%) on aligned test comments, splits pooled:")
print(acc.to_string())
lat = T[T.tag.isin(["ENG", "ROM_BAN"])]
acc2 = lat.groupby("position")[list(SYSTEMS)].mean().mul(100).round(2).rename(columns=SYSTEMS)
acc2["tokens"] = lat.groupby("position").size()
acc2["CTT - word level"] = (acc2["CTT"] - acc2["Word level"]).round(2)
print("same, Latin-script ENG/ROM tokens only:")
print(acc2.to_string())

bins = [0, 10, 20, 40, 10 ** 6]; names = ["1-10", "11-20", "21-40", ">40"]
rows = []
for s in SEEDS:
    for k, c in enumerate(TEST[s]):
        r = dict(len_bin=pd.cut([len(df.at[c, "tokens"])], bins, labels=names)[0])
        for v in SYSTEMS:
            r[v + "_e"], r["ln"] = E[(v, s, True)][0][k], E[(v, s, True)][1][k]
        rows.append(r)
L = pd.DataFrame(rows)
g = L.groupby("len_bin", observed=True)
cer = pd.DataFrame({SYSTEMS[v]: 100 * g[v + "_e"].sum() / g["ln"].sum() for v in SYSTEMS}).round(2)
cer["comments"] = g.size()
print("guarded CER (%) by comment length in tokens, splits pooled:")
print(cer.to_string())

print("\n== C. automatic taxonomy of CTT token errors (aligned test comments, splits pooled, guarded)")
vocab = {}
for s in SEEDS:
    sp = pd.read_csv(os.path.join(ROOT, "data", "splits", f"split_seed{s}.csv"))
    vocab[s] = {bn(w).lower() for c in sp[sp.split == "train"].id for w in df.at[c, "tokens"]}


def script(w):
    has_bn = any("ঀ" <= ch <= "৿" for ch in w)
    has_lat = any(ch.isascii() and ch.isalpha() for ch in w)
    return "mixed" if has_bn and has_lat else "bengali" if has_bn else "latin" if has_lat else "none"


errs = []
for s in SEEDS:
    for c in TEST[s]:
        toks, labels, ref = df.at[c, "tokens"], df.at[c, "labels"], df.at[c, "ref"].split()
        if len(toks) != len(ref):
            continue
        for w, l, x, y in zip(toks, labels, SPANS[("ctx_plain_aks_heads", s)][c], ref):
            o = bn(guard(x, w))
            if o == y:
                continue
            src = bn(w)
            if CTRL.search(o) and CTRL.sub("", o) == y:
                cat = "Invisible control byte only"
            elif l == "BAN":
                cat = "BAN: respelling missed" if o == src else "BAN: wrong change" if y == src else "BAN: different respelling"
            elif l == "OTHER":
                cat = "Digit / symbol (OTH)"
            elif l == "UNK":
                cat = "Glued / UNK token"
            elif script(w) != "latin":
                cat = "Other"
            else:
                cat = "Latin word, unseen in training" if src.lower() not in vocab[s] else "Latin word, seen in training"
            errs.append(dict(cat=cat, near_miss=lev(o, y) <= 1))
Err = pd.DataFrame(errs)
tax = Err.groupby("cat").agg(count=("cat", "size"), near_miss_pct=("near_miss", "mean"))
tax["share_pct"] = 100 * tax["count"] / len(Err)
tax["near_miss_pct"] *= 100
print(f"{len(Err)} errors in {len(T)} aligned test tokens ({100 * len(Err) / len(T):.2f}% token error rate)")
print(tax.sort_values("count", ascending=False).round(1).to_string())
print(f"near misses (<= 1 character edit) overall: {100 * Err.near_miss.mean():.1f}%")
