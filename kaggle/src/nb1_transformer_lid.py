# %% [markdown]
# # BanglaMix — Transformer token-level LID (Kaggle, GPU)
# Fine-tunes multilingual / Bangla encoders for 5-way token LID on the **same duplicate-aware splits**
# used by the local classical baselines (`splits/split_seed{13,42,2026}.csv`).
#
# **Setup:** Accelerator = GPU T4 x2 (or P100), Internet = ON. Attach the Kaggle dataset you created from
# the Kaggle dataset `kaggle/kaggle_dataset.zip` (it contains `BanglaMix.csv`, `splits/`, `noise_matrix.csv`).
#
# Loss variants: `ce` (cross-entropy), `ls` (label smoothing 0.1), `gce` (generalised CE, q=0.7),
# `fc` (forward loss correction with the confident-learning noise matrix).
# Runtime on a T4: about 4–7 min per (model, seed, loss) for base-size models.

# %%
!pip -q install seqeval==1.2.2 2>/dev/null
!pip -q install git+https://github.com/csebuetnlp/normalizer 2>/dev/null || echo "normalizer optional"

# %%
import glob, json, os, random, time, unicodedata
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found under /kaggle/input — attach the BanglaMix Kaggle dataset (kaggle/kaggle_dataset.zip)"
    return hits[0]

DATA = find("BanglaMix.csv")
SPLITS = os.path.dirname(find("split_seed42.csv"))
OUT = "/kaggle/working"
LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]
L2I = {l: i for i, l in enumerate(LABELS)}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(DATA, SPLITS, DEV)

# %% [markdown]
# ## Configuration — edit here
# Start with one model + seed 42 to check everything works, then widen the grid.

# %%
MODELS = [
    "xlm-roberta-base",
    "csebuetnlp/banglishbert",
    "csebuetnlp/banglabert",
    "google/muril-base-cased",
    "bert-base-multilingual-cased",
    # "xlm-roberta-large",   # ~3x slower; uses the HP override below. Architecture search: run it alone first.
]
SEEDS = [13, 42, 2026]
LOSSES = ["ce"]            # add "ls", "gce", "fc" for the noise-robust training study
EPOCHS, LR, BS, MAX_WORDS, MAX_LEN = 5, 3e-5, 16, 64, 256
HP = {"xlm-roberta-large": dict(lr=1e-5, bs=8)}   # per-model overrides of LR / BS

# %%
df = pd.read_csv(DATA)
def parse(tl):
    p = [t.rsplit("/", 1) for t in tl.split(" ")]
    return [x[0] for x in p], [x[1] for x in p]
df["tokens"], df["labels"] = zip(*df.token_labels.map(parse))

try:
    from normalizer import normalize as bn_normalize
except Exception:
    bn_normalize = None

def prep_word(w, model_name):
    if bn_normalize is not None and "csebuetnlp" in model_name:
        n = bn_normalize(w)
        return n if n.strip() else w
    return w

def chunks(ws, ls, n=MAX_WORDS):
    for i in range(0, len(ws), n):
        yield ws[i:i + n], ls[i:i + n]

def encode(tok, words, labels, model_name):
    words = [prep_word(w, model_name) for w in words]
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN)
    wid = enc.word_ids()
    lab, prev = [], None
    for w in wid:
        lab.append(-100 if w is None or w == prev else L2I[labels[w]])
        prev = w
    enc["labels"] = lab
    enc["word_ids"] = wid
    return enc

def build(tok, part, model_name):
    items = []
    for r in part.itertuples():
        for k, (ws, ls) in enumerate(chunks(r.tokens, r.labels)):
            e = encode(tok, ws, ls, model_name)
            e["cid"], e["chunk"], e["nwords"] = r.id, k, len(ws)
            items.append(e)
    return items

def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id)
    att = torch.zeros((len(batch), m), dtype=torch.long)
    lab = torch.full((len(batch), m), -100)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = 1
        lab[i, :n] = torch.tensor(b["labels"])
    return ids, att, lab

# %%
T_noise = None
try:
    T_noise = torch.tensor(pd.read_csv(find("noise_matrix.csv"), index_col=0).values, dtype=torch.float)
    T_noise = 0.95 * T_noise + 0.05 * torch.eye(5)   # keep it invertible / well-conditioned
except AssertionError:
    print("noise_matrix.csv not found: 'fc' loss unavailable")

def loss_fn(logits, labels, kind):
    mask = labels != -100
    lg, y = logits[mask], labels[mask]
    if kind == "ce":
        return F.cross_entropy(lg, y)
    if kind == "ls":
        return F.cross_entropy(lg, y, label_smoothing=0.1)
    if kind == "gce":
        p = F.softmax(lg, -1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
        return ((1 - p ** 0.7) / 0.7).mean()
    if kind == "fc":   # p(observed) = p(clean) @ T, T[i,j] = P(obs=j | true=i)
        p = F.softmax(lg, -1) @ T_noise.to(lg.device)
        return F.nll_loss(torch.log(p.clamp_min(1e-7)), y)
    raise ValueError(kind)

@torch.no_grad()
def predict(model, items, pad_id, with_probs=False):
    """Word tags per comment (first sub-word); with_probs also returns the word-level softmax for ensembling."""
    model.eval()
    out, outp = {}, {}
    fallback = [0.0, 0.0, 0.0, 1.0, 0.0]                  # OTHER, for a word that got zero sub-tokens
    for i in range(0, len(items), 64):
        b = items[i:i + 64]
        ids, att, _ = collate(b, pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            lg = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits.float().cpu()
        pr, pp = lg.argmax(-1).numpy(), F.softmax(lg, -1).numpy()
        for j, e in enumerate(b):
            words, probs = ["OTHER"] * e["nwords"], [fallback] * e["nwords"]
            seen = set()
            for t, w in enumerate(e["word_ids"]):
                if w is not None and w not in seen:
                    words[w] = LABELS[pr[j, t]]
                    probs[w] = pp[j, t].round(4).tolist()
                    seen.add(w)
            out.setdefault(e["cid"], {})[e["chunk"]] = words
            outp.setdefault(e["cid"], {})[e["chunk"]] = probs
    tags = {c: [l for k in sorted(v) for l in v[k]] for c, v in out.items()}
    if not with_probs:
        return tags
    return tags, {c: [p for k in sorted(v) for p in v[k]] for c, v in outp.items()}

def scores(gold, pred):
    yt = [l for g in gold for l in g]
    yp = [l for p in pred for l in p]
    f = f1_score(yt, yp, labels=LABELS, average=None, zero_division=0)
    s = {f"F1_{l}": 100 * x for l, x in zip(LABELS, f)}
    s.update(MacroF1=100 * f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
             Acc=100 * accuracy_score(yt, yp), Kappa=cohen_kappa_score(yt, yp))
    return s

# %%
def run(model_name, seed, loss_kind):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    sp = pd.read_csv(f"{SPLITS}/split_seed{seed}.csv")
    d = df.merge(sp, on="id")
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tr_i, va_i, te_i = build(tok, tr, model_name), build(tok, va, model_name), build(tok, te, model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=5).to(DEV)
    lr, BS = HP.get(model_name, {}).get("lr", LR), HP.get(model_name, {}).get("bs", globals()["BS"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = EPOCHS * (len(tr_i) // BS + 1)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    best, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        order = np.random.permutation(len(tr_i))
        for i in range(0, len(order), BS):
            ids, att, lab = collate([tr_i[j] for j in order[i:i + BS]], tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                lg = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits
            loss = loss_fn(lg.float(), lab.to(DEV), loss_kind)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        pv = predict(model, va_i, tok.pad_token_id)
        v = scores(list(va.labels), [pv[c] for c in va.id])["MacroF1"]
        print(f"  epoch {ep} val MacroF1 {v:.2f}")
        if v > best:
            best, best_state = v, {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    pt, pp = predict(model, te_i, tok.pad_token_id, with_probs=True)
    s = scores(list(te.labels), [pt[c] for c in te.id])
    s.update(model=model_name, seed=seed, loss=loss_kind, val_MacroF1=best)
    tag = f"{model_name.replace('/', '_')}_s{seed}_{loss_kind}"
    json.dump({c: pt[c] for c in te.id}, open(f"{OUT}/lid_test_pred_{tag}.json", "w"))
    json.dump({c: pp[c] for c in te.id}, open(f"{OUT}/lid_test_prob_{tag}.json", "w"))   # for the nb5 LID ensemble
    del model; torch.cuda.empty_cache()
    return s

# %%
rows = []
res_path = f"{OUT}/transformer_lid_results.csv"
for m in MODELS:
    for loss_kind in LOSSES:
        for seed in SEEDS:
            t = time.time()
            print(m, seed, loss_kind)
            try:
                s = run(m, seed, loss_kind)
            except Exception as ex:          # keep the grid going if one checkpoint fails to load
                print("FAILED", m, ex); continue
            s["minutes"] = (time.time() - t) / 60
            rows.append(s)
            pd.DataFrame(rows).to_csv(res_path, index=False)
            print({k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()})

# %%
res = pd.DataFrame(rows)
cols = ["Acc", "MacroF1", "Kappa"] + [f"F1_{l}" for l in LABELS]
agg = res.groupby(["model", "loss"])[cols].agg(["mean", "std"]).round(2)
agg.to_csv(f"{OUT}/transformer_lid_summary.csv")
agg
