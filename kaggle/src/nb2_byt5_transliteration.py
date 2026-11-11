# %% [markdown]
# # BanglaMix — Neural Bengali-script normalisation with ByT5 (Kaggle, GPU)
# Two families, both scored exactly like the local lexicon baselines (CER / WER / exact match on the full
# normalised `bengali_transliteration`, token accuracy on token-aligned comments):
#
# **A. Word-level (routed) ByT5** — one model maps `TAG word → Bengali word`.
#   Inference ablations: no tag (language-agnostic) · predicted tag · predicted tag + hard copy-gate ·
#   oracle tag + gate · lexicon-first hybrid.
#
# **B. Sentence-level ByT5** — end-to-end comment → Bengali script, plain vs. with inline LID tags.
#
# **Setup:** GPU (P100 is faster than T4 in fp32), Internet ON, attach the Kaggle dataset built from `kaggle/kaggle_dataset.zip`.
# T5-family models overflow in fp16, so training runs in fp32 (bf16 is used automatically on GPUs that support it).
# Rough T4 runtime: word-level ~25 min per seed; sentence-level ~1.5–2 h for one seed.

# %%
!pip -q install rapidfuzz 2>/dev/null

# %%
import glob, json, os, random, re, time, unicodedata
from collections import Counter, defaultdict
import numpy as np, pandas as pd, torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found under /kaggle/input"
    return hits[0]

DATA = find("BanglaMix.csv")
SPLITS = os.path.dirname(find("split_seed42.csv"))
LIDP = os.path.dirname(find("crf_pred_seed42.json"))
OUT = "/kaggle/working"
DEV = "cuda"
BF16 = torch.cuda.is_bf16_supported()
LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]

# %% [markdown]
# ## Configuration

# %%
MODEL = "google/byt5-small"     # alternatives: "google/byt5-base" (A100/L4), "csebuetnlp/banglat5", "google/mt5-small"
WORD_SEEDS = [13, 42, 2026]
SENT_SEEDS = [42]                # sentence-level is slow; add 13 / 2026 if you have GPU quota
RUN_WORD, RUN_SENT = True, True
# Which predicted LID to route with: "crf" (local CRF, shipped in kaggle_dataset.zip) or a path pattern to
# notebook-1 outputs, e.g. "/kaggle/input/<nb1-output>/lid_test_pred_xlm-roberta-base_s{seed}_ce.json"
LID_SOURCE = "crf"

# %%
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
def bn_norm(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")

df = pd.read_csv(DATA)
def parse(tl):
    p = [t.rsplit("/", 1) for t in tl.split(" ")]
    return [x[0] for x in p], [x[1] for x in p]
df["tokens"], df["labels"] = zip(*df.token_labels.map(parse))
df["targets"] = df.bengali_transliteration.map(lambda s: bn_norm(s).split())
df["aligned"] = [len(a) == len(b) for a, b in zip(df.tokens, df.targets)]

def split(seed):
    d = df.merge(pd.read_csv(f"{SPLITS}/split_seed{seed}.csv"), on="id")
    return (d[d.split == s] for s in ("train", "val", "test"))

def lid_pred(seed):
    if LID_SOURCE == "crf":
        return json.load(open(f"{LIDP}/crf_pred_seed{seed}.json"))
    return json.load(open(LID_SOURCE.format(seed=seed)))

def evaluate(te, outputs):
    """outputs: list of token-lists (or strings) per test comment, same order as te."""
    hyps = [o if isinstance(o, str) else " ".join(o) for o in outputs]
    refs = [" ".join(t) for t in te.targets]
    ce = sum(Levenshtein.distance(h, r) for h, r in zip(hyps, refs)); cn = sum(map(len, refs))
    we = sum(Levenshtein.distance(h.split(), r.split()) for h, r in zip(hyps, refs)); wn = sum(len(r.split()) for r in refs)
    s = {"CER": 100 * ce / cn, "WER": 100 * we / wn, "Exact": 100 * np.mean([h == r for h, r in zip(hyps, refs)])}
    ok, lab = [], []
    for h, r in zip(hyps, te.itertuples()):
        hs = h.split()
        if r.aligned and len(hs) == len(r.targets):
            ok += [a == b for a, b in zip(hs, r.targets)]; lab += list(r.labels)
    ok, lab = np.array(ok), np.array(lab)
    s["TokAcc"] = 100 * ok.mean()
    for l in LABELS:
        s[f"TokAcc_{l}"] = 100 * ok[lab == l].mean() if (lab == l).any() else np.nan
    return s

# %%
def train_seq2seq(pairs, val_pairs, seed, epochs, bs, lr, max_src, max_tgt, accum=1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (len(pairs) // (bs * accum) + 1)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); t = time.time()
        order = np.random.permutation(len(pairs))
        for k, i in enumerate(range(0, len(order), bs)):
            b = [pairs[j] for j in order[i:i + bs]]
            x = tok([s for s, _ in b], return_tensors="pt", padding=True, truncation=True, max_length=max_src).to(DEV)
            y = tok([t for _, t in b], return_tensors="pt", padding=True, truncation=True, max_length=max_tgt).input_ids
            y[y == tok.pad_token_id] = -100
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
                loss = model(**x, labels=y.to(DEV)).loss / accum
            loss.backward()
            if (k + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
        hyp = generate(model, tok, [s for s, _ in val_pairs], max_src, max_tgt)
        acc = 100 * np.mean([h == t for h, (_, t) in zip(hyp, val_pairs)])
        print(f"  epoch {ep} val exact {acc:.2f}  ({(time.time() - t) / 60:.1f} min)")
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tok

@torch.no_grad()
def generate(model, tok, srcs, max_src, max_tgt, bs=64):
    model.eval(); out = []
    for i in range(0, len(srcs), bs):
        x = tok(srcs[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_src).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
            g = model.generate(**x, max_new_tokens=max_tgt, num_beams=1)
        out += [s.strip() for s in tok.batch_decode(g, skip_special_tokens=True)]
    return out

# %% [markdown]
# ## A. Word-level routed ByT5

# %%
def word_pairs(part, tagged, cap=5):
    c = Counter()
    for r in part[part.aligned].itertuples():
        for w, l, y in zip(r.tokens, r.labels, r.targets):
            c[(f"{l} {w}" if tagged else w, y)] += 1
    return [p for p, n in c.items() for _ in range(min(n, cap))]

def run_word(seed):
    tr, va, te = split(seed)
    pred = lid_pred(seed)
    lex = defaultdict(Counter)
    for r in tr[tr.aligned].itertuples():
        for w, l, y in zip(r.tokens, r.labels, r.targets):
            lex[(bn_norm(w).lower(), l)][y] += 1
    lex = {k: v.most_common(1)[0][0] for k, v in lex.items()}
    rows, models = [], {}
    for tagged in (False, True):
        tp, vp = word_pairs(tr, tagged), list(set(word_pairs(va, tagged, cap=1)))[:3000]
        print(f"seed {seed} tagged={tagged}: {len(tp)} train pairs")
        models[tagged] = train_seq2seq(tp, vp, seed, epochs=4, bs=32, lr=5e-4, max_src=48, max_tgt=96, accum=4)

    def infer(tagged, tag_src, gate, hybrid=False):
        model, tok = models[tagged]
        reqs, plan = set(), []
        for r in te.itertuples():
            tags = r.labels if tag_src == "gold" else pred[r.id]
            row = []
            for w, t in zip(r.tokens, tags):
                if gate and t in ("BAN", "OTHER"):
                    row.append(("copy", bn_norm(w).translate(BN_DIGITS) if t == "OTHER" else bn_norm(w)))
                elif hybrid and (bn_norm(w).lower(), t) in lex:
                    row.append(("copy", lex[(bn_norm(w).lower(), t)]))
                else:
                    s = f"{t} {w}" if tagged else w
                    reqs.add(s); row.append(("gen", s))
            plan.append(row)
        reqs = sorted(reqs)
        gen = dict(zip(reqs, generate(model, tok, reqs, 48, 96)))
        return [[v if k == "copy" else (gen[v] or v) for k, v in row] for row in plan]

    variants = {
        "ByT5-word, no LID (language-agnostic)": (False, "none", False, False),
        "ByT5-word + predicted LID": (True, "pred", False, False),
        "ByT5-word + predicted LID + copy-gate": (True, "pred", True, False),
        "ByT5-word + predicted LID + copy-gate + lexicon-first": (True, "pred", True, True),
        "ByT5-word + oracle LID + copy-gate (upper bound)": (True, "gold", True, False),
    }
    for name, v in variants.items():
        s = evaluate(te, infer(*v)); s.update(system=name, seed=seed); rows.append(s)
        print(f"  {name:55s} CER {s['CER']:.2f} WER {s['WER']:.2f} Exact {s['Exact']:.2f}")
    del models; torch.cuda.empty_cache()
    return rows

# %%
word_rows = []
if RUN_WORD:
    for seed in WORD_SEEDS:
        word_rows += run_word(seed)
        pd.DataFrame(word_rows).to_csv(f"{OUT}/byt5_word_results.csv", index=False)

# %% [markdown]
# ## B. Sentence-level ByT5 (end-to-end)

# %%
CH = 16   # words per chunk
def sent_pairs(part, tagged, tags=None):
    out = []
    for r in part.itertuples():
        tg = tags[r.id] if tags else r.labels
        src_w = [f"{w}|{t}" for w, t in zip(r.tokens, tg)] if tagged else list(r.tokens)
        if r.aligned:
            for i in range(0, len(src_w), CH):
                out.append((" ".join(src_w[i:i + CH]), " ".join(r.targets[i:i + CH])))
        elif len(src_w) <= CH:
            out.append((" ".join(src_w), " ".join(r.targets)))
    return out

def sent_infer(model, tok, te, tagged, tags):
    srcs, idx = [], []
    for n, r in enumerate(te.itertuples()):
        tg = tags[r.id] if tagged else None
        ws = [f"{w}|{t}" for w, t in zip(r.tokens, tg)] if tagged else list(r.tokens)
        for i in range(0, len(ws), CH):
            srcs.append(" ".join(ws[i:i + CH])); idx.append(n)
    hyp = generate(model, tok, srcs, 320, 640, bs=16)
    out = [[] for _ in range(len(te))]
    for n, h in zip(idx, hyp):
        out[n].append(h)
    return [" ".join(x) for x in out]

sent_rows = []
if RUN_SENT:
    for seed in SENT_SEEDS:
        tr, va, te = split(seed)
        pred = lid_pred(seed)
        for tagged in (False, True):
            tp = sent_pairs(tr, tagged)
            vp = sent_pairs(va, tagged)[:600]
            print(f"sentence-level seed {seed} tagged={tagged}: {len(tp)} chunks")
            model, tok = train_seq2seq(tp, vp, seed, epochs=6, bs=8, lr=3e-4, max_src=320, max_tgt=640, accum=4)
            variants = [("pred", pred)] + ([("gold", {r.id: r.labels for r in te.itertuples()})] if tagged else [])
            for src_name, tags in variants:
                s = evaluate(te, sent_infer(model, tok, te, tagged, tags))
                name = "ByT5-sentence" + (f" + inline {src_name} LID" if tagged else " (plain)")
                s.update(system=name, seed=seed); sent_rows.append(s)
                print(f"  {name:45s} CER {s['CER']:.2f} WER {s['WER']:.2f} Exact {s['Exact']:.2f}")
            model.save_pretrained(f"{OUT}/byt5_sent_s{seed}_tag{int(tagged)}"); tok.save_pretrained(f"{OUT}/byt5_sent_s{seed}_tag{int(tagged)}")
            del model; torch.cuda.empty_cache()
            pd.DataFrame(sent_rows).to_csv(f"{OUT}/byt5_sentence_results.csv", index=False)

# %%
res = pd.DataFrame(word_rows + sent_rows)
cols = ["CER", "WER", "Exact", "TokAcc"] + [f"TokAcc_{l}" for l in LABELS]
summary = res.groupby("system", sort=False)[cols].agg(["mean", "std"]).round(2)
summary.to_csv(f"{OUT}/byt5_summary.csv")
summary
