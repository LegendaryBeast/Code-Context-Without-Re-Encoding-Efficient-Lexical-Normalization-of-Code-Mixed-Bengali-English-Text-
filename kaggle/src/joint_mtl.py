"""Joint multi-task ByT5 for BanglaMix.

One byte-level encoder is shared by
  (a) a word-level LID head: mean-pooled encoder states over each word's bytes -> 5 classes, and
  (b) the Bengali-script normalisation decoder.
With --inject 1, the LID posterior of every word is projected to d_model and added to the encoder
states of that word's bytes before the decoder cross-attends to them (soft, differentiable routing;
the projection starts at zero, so training starts from the plain model).

Loss: lam * CE_LID + (1 - lam) * CE_seq2seq
  lam = 0            -> plain sentence-level ByT5 (baseline, trained under identical settings)
  lam = 1            -> encoder-only ByT5 LID tagger (decoder untrained, no normalisation output)
  0 < lam < 1        -> multi-task

At test time four normalisation outputs are scored:
  free               decoder output as is
  token-gate         words tagged BAN/OTHER are copied (only when output word count matches input)
  chunk-skip         chunks whose words are all BAN/OTHER skip the decoder entirely (measured speed-up)
  chunk-skip+gate    both
Gating uses the model's own LID head, or the CRF predictions when lam = 0.

Run: python joint_mtl.py --variant mtl_inject --lam 0.4 --inject 1 --seed 42
"""
import argparse
import glob
import json
import os
import random
import time
import unicodedata

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True)
p.add_argument("--lam", type=float, required=True)
p.add_argument("--inject", type=int, default=0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--model", default="google/byt5-small")
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--accum", type=int, default=2)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--src_budget", type=int, default=160, help="max source bytes per chunk (words are never split)")
p.add_argument("--max_tgt", type=int, default=640)
p.add_argument("--val_chunks", type=int, default=600)
p.add_argument("--input", default="/kaggle/input")
p.add_argument("--out", default="/kaggle/working/joint")
p.add_argument("--subsample", type=int, default=0, help="smoke tests only: use N random comments")
args = p.parse_args()
assert 0.0 <= args.lam <= 1.0 and not (args.inject and args.lam == 0), "injection needs a trained LID head (lam > 0)"
os.makedirs(args.out, exist_ok=True)
TAG = f"{args.variant}_s{args.seed}"

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = DEV == "cuda" and torch.cuda.is_bf16_supported()   # T5 overflows in fp16, so fp32 on T4/P100
LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]
L2I = {l: i for i, l in enumerate(LABELS)}
COPY = ("BAN", "OTHER")
EOS, OFFSET = 1, 3                                          # ByT5: id = byte + 3, eos = 1, pad = 0
MAX_SRC = 2 * args.src_budget
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def log(*a):
    print(*a, flush=True)


def find(name):
    hits = glob.glob(f"{args.input}/**/{name}", recursive=True)
    assert hits, f"{name} not found under {args.input}"
    return hits[0]


def bn_norm(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


def copy_word(w, tag):
    return bn_norm(w).translate(BN_DIGITS) if tag == "OTHER" else bn_norm(w)


# ------------------------------------------------------------------ data
df = pd.read_csv(find("BanglaMix.csv"))
if args.subsample:
    df = df.sample(args.subsample, random_state=0).reset_index(drop=True)


def parse(tl):
    pairs = [t.rsplit("/", 1) for t in tl.split(" ")]
    return [x[0] for x in pairs], [x[1] for x in pairs]


df["tokens"], df["labels"] = zip(*df.token_labels.map(parse))
df["targets"] = df.bengali_transliteration.map(lambda s: bn_norm(s).split())
df["aligned"] = [len(a) == len(b) for a, b in zip(df.tokens, df.targets)]
df = df.merge(pd.read_csv(find(f"split_seed{args.seed}.csv")), on="id")
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
crf = json.load(open(find(f"crf_pred_seed{args.seed}.json")))
train_vocab = set(bn_norm(w).lower() for ws in tr.tokens for w in ws)


def chunk_words(words, budget):
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


def to_ids(s, limit):
    return [b + OFFSET for b in s.encode("utf-8")][:limit - 1] + [EOS]


def make_examples(part):
    ex = []
    for r in part.itertuples():
        idxs = chunk_words(r.tokens, args.src_budget)
        for k, idx in enumerate(idxs):
            words = [r.tokens[i] for i in idx]
            ids, spans = [], []
            for j, w in enumerate(words):
                if j:
                    ids.append(ord(" ") + OFFSET)
                b = [x + OFFSET for x in w.encode("utf-8")]
                spans.append((len(ids), len(ids) + len(b)))
                ids += b
            ids = ids[:MAX_SRC - 1] + [EOS]
            spans = [(s, min(e, len(ids) - 1)) for s, e in spans]   # words cut by truncation get empty spans
            if r.aligned:
                tgt = " ".join(r.targets[i] for i in idx)
            elif len(idxs) == 1:
                tgt = " ".join(r.targets)
            else:
                tgt = None                                            # LID-only example (target not alignable)
            ex.append(dict(cid=r.id, chunk=k, widx=idx, words=words, ids=ids, spans=spans,
                           lid=[L2I[r.labels[i]] for i in idx], tgt=tgt,
                           tgt_ids=to_ids(tgt, args.max_tgt) if tgt is not None else None))
    return ex


tr_ex, va_ex, te_ex = make_examples(tr), make_examples(va), make_examples(te)
va_sel = [e for e in va_ex if e["tgt"] is not None][:args.val_chunks]
log(f"[{TAG}] chunks train {len(tr_ex)} val {len(va_ex)} test {len(te_ex)} | device {DEV} bf16 {BF16}")


def collate(batch, with_targets=True):
    B = len(batch)
    T = max(len(e["ids"]) for e in batch)
    W = max(len(e["spans"]) for e in batch)
    ids = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    assign = torch.zeros(B, W, T)
    lid = torch.full((B, W), -100, dtype=torch.long)
    for i, e in enumerate(batch):
        ids[i, :len(e["ids"])] = torch.tensor(e["ids"])
        att[i, :len(e["ids"])] = 1
        for w, (s, t) in enumerate(e["spans"]):
            if t > s:
                assign[i, w, s:t] = 1.0
                lid[i, w] = e["lid"][w]
    pool = assign / assign.sum(-1, keepdim=True).clamp_min(1.0)
    labels = None
    if with_targets:
        L = max([len(e["tgt_ids"]) for e in batch if e["tgt_ids"] is not None] or [1])
        labels = torch.full((B, L), -100, dtype=torch.long)
        for i, e in enumerate(batch):
            if e["tgt_ids"] is not None:
                labels[i, :len(e["tgt_ids"])] = torch.tensor(e["tgt_ids"])
    return ids, att, pool, assign, lid, labels


def batches(ex, bs, shuffle, rng):
    idx = rng.permutation(len(ex)) if shuffle else np.arange(len(ex))
    out = []
    for m in range(0, len(idx), bs * 50):                     # length bucketing inside mega-batches
        mega = sorted(idx[m:m + bs * 50], key=lambda i: len(ex[i]["ids"]) + len(ex[i]["tgt_ids"] or []))
        out += [mega[i:i + bs] for i in range(0, len(mega), bs)]
    if shuffle:
        rng.shuffle(out)
    return out


# ------------------------------------------------------------------ model
class Joint(nn.Module):
    def __init__(self, name, inject):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(name)
        H = self.t5.config.d_model
        self.drop = nn.Dropout(0.1)
        self.lid = nn.Linear(H, len(LABELS))
        self.inject = bool(inject)
        self.tag = nn.Linear(len(LABELS), H, bias=False)
        nn.init.zeros_(self.tag.weight)

    def encode(self, ids, att, pool, assign):
        h = self.t5.get_encoder()(input_ids=ids, attention_mask=att).last_hidden_state
        lid_logits = self.lid(self.drop(torch.bmm(pool.to(h.dtype), h)))
        if self.inject:
            tv = self.tag(F.softmax(lid_logits.float(), -1).to(h.dtype))
            h = h + torch.bmm(assign.transpose(1, 2).to(h.dtype), tv)
        return h, lid_logits

    def forward(self, ids, att, pool, assign, labels):
        h, lid_logits = self.encode(ids, att, pool, assign)
        cfg = self.t5.config
        dec = torch.cat([torch.full_like(labels[:, :1], cfg.decoder_start_token_id), labels[:, :-1]], 1)
        dec = dec.masked_fill(dec == -100, cfg.pad_token_id)
        out = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=att, decoder_input_ids=dec)
        return out.logits, lid_logits


def decode_bytes(seq):
    out = []
    for x in seq.tolist():
        if x == EOS:
            break
        if OFFSET <= x < OFFSET + 256:
            out.append(x - OFFSET)
    return bytes(out).decode("utf-8", errors="ignore").strip()


@torch.no_grad()
def infer(model, ex, generate=True, bs=16):
    """Returns (lid tags per example, generated string per example or None, generation seconds)."""
    model.eval()
    tags, texts, gen_time = [None] * len(ex), [None] * len(ex), 0.0
    order = sorted(range(len(ex)), key=lambda i: len(ex[i]["ids"]))
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        ids, att, pool, assign, _, _ = collate([ex[i] for i in b], with_targets=False)
        ids, att, pool, assign = ids.to(DEV), att.to(DEV), pool.to(DEV), assign.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
            h, lg = model.encode(ids, att, pool, assign)
            pred = lg.float().argmax(-1).cpu()
            for j, i in enumerate(b):
                tags[i] = [LABELS[pred[j, w]] if t > s_ else "OTHER" for w, (s_, t) in enumerate(ex[i]["spans"])]
            if generate:
                if DEV == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                g = model.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=att,
                                      max_new_tokens=min(args.max_tgt, 4 * ids.shape[1] + 16), num_beams=1, do_sample=False)
                if DEV == "cuda":
                    torch.cuda.synchronize()
                gen_time += time.time() - t0
                for j, i in enumerate(b):
                    texts[i] = decode_bytes(g[j])
    return tags, texts, gen_time


def lid_scores(gold, pred, words):
    yt, yp = sum(gold, []), sum(pred, [])
    f = f1_score(yt, yp, labels=LABELS, average=None, zero_division=0)
    s = {f"F1_{l}": 100 * x for l, x in zip(LABELS, f)}
    s.update(MacroF1=100 * f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
             Acc=100 * accuracy_score(yt, yp), Kappa=float(cohen_kappa_score(yt, yp)))
    oov = np.array([bn_norm(w).lower() not in train_vocab for w in sum(words, [])])
    ok = np.array(yt) == np.array(yp)
    s["OOV_Acc"], s["IV_Acc"] = 100 * ok[oov].mean(), 100 * ok[~oov].mean()
    return s


def norm_scores(part, hyp):
    H = [hyp[c] for c in part.id]
    refs = [" ".join(t) for t in part.targets]
    s = {"CER": 100 * sum(Levenshtein.distance(h, r) for h, r in zip(H, refs)) / sum(map(len, refs)),
         "WER": 100 * sum(Levenshtein.distance(h.split(), r.split()) for h, r in zip(H, refs)) / sum(len(r.split()) for r in refs),
         "Exact": 100 * float(np.mean([h == r for h, r in zip(H, refs)]))}
    ok, lab, oov, n_al, n_match = [], [], [], 0, 0
    for h, r in zip(H, part.itertuples()):
        if not r.aligned:
            continue
        n_al += 1
        hs = h.split()
        if len(hs) == len(r.targets):
            n_match += 1
            ok += [a == b for a, b in zip(hs, r.targets)]
            lab += list(r.labels)
            oov += [bn_norm(w).lower() not in train_vocab for w in r.tokens]
    ok, lab, oov = np.array(ok), np.array(lab), np.array(oov)
    s["TokAcc"] = 100 * ok.mean() if len(ok) else float("nan")
    for l in LABELS:
        s[f"TokAcc_{l}"] = 100 * ok[lab == l].mean() if (lab == l).any() else float("nan")
    s["TokAcc_OOV"] = 100 * ok[oov].mean() if oov.any() else float("nan")
    s["word_count_match_share"] = 100 * n_match / max(1, n_al)
    return s


def chunk_val_cer(model):
    _, texts, _ = infer(model, va_sel)
    refs = [e["tgt"] for e in va_sel]
    return 100 * sum(Levenshtein.distance(h, r) for h, r in zip(texts, refs)) / sum(map(len, refs))


# ------------------------------------------------------------------ train
model = Joint(args.model, args.inject).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
rng = np.random.RandomState(args.seed)
steps = args.epochs * (len(batches(tr_ex, args.bs, False, rng)) // args.accum + 1)
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
best, best_state, history = None, None, []
for ep in range(args.epochs):
    model.train()
    t0, tot, n = time.time(), 0.0, 0
    all_batches = batches(tr_ex, args.bs, True, rng)
    opt.zero_grad()
    for step_idx, b in enumerate(all_batches):
        ids, att, pool, assign, lid, labels = (x.to(DEV) for x in collate([tr_ex[i] for i in b]))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
            if args.lam < 1:
                dec_logits, lid_logits = model(ids, att, pool, assign, labels)
            else:
                _, lid_logits = model.encode(ids, att, pool, assign)
        loss = torch.zeros((), device=DEV)
        if args.lam > 0 and (lid != -100).any():      # guard: CE over an all-ignored batch is NaN, and 0 * NaN = NaN
            loss = loss + args.lam * F.cross_entropy(lid_logits.float().reshape(-1, len(LABELS)), lid.reshape(-1), ignore_index=-100)
        if args.lam < 1 and (labels != -100).any():
            seq = F.cross_entropy(dec_logits.float().reshape(-1, dec_logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            loss = loss + (1 - args.lam) * seq
        if not loss.requires_grad:
            continue
        (loss / args.accum).backward()
        if (step_idx + 1) % args.accum == 0 or (step_idx + 1) == len(all_batches):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
        tot, n = tot + loss.item(), n + 1
    rec = {"epoch": ep, "train_loss": tot / n, "train_min": (time.time() - t0) / 60}
    if args.lam > 0:
        vt, _, _ = infer(model, va_ex, generate=False)
        rec["val_LID_MacroF1"] = lid_scores([[LABELS[i] for i in e["lid"]] for e in va_ex], vt, [e["words"] for e in va_ex])["MacroF1"]
    if args.lam < 1:
        rec["val_chunk_CER"] = chunk_val_cer(model)
    score = -rec["val_chunk_CER"] if args.lam < 1 else rec["val_LID_MacroF1"]   # normalisation is the primary task
    history.append(rec)
    log(f"[{TAG}] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    if best is None or score > best:
        best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if DEV == "cuda":
        torch.cuda.empty_cache()
model.load_state_dict(best_state)


# ------------------------------------------------------------------ test
def per_comment(ex, values):
    out = {}
    for e, v in zip(ex, values):
        out.setdefault(e["cid"], []).append(v)
    return out


result = {"args": vars(args), "history": history, "n_test_comments": len(te), "n_test_chunks": len(te_ex)}
generate = args.lam < 1
tags, texts, t_full = infer(model, te_ex, generate=generate)
if args.lam > 0:
    pc = per_comment(te_ex, tags)
    pred = {c: sum(v, []) for c, v in pc.items()}
    result["lid"] = lid_scores(list(te.labels), [pred[c] for c in te.id], list(te.tokens))
    gate_tags = tags
    result["gate_lid_source"] = "own LID head"
else:
    gate_tags = [[crf[e["cid"]][i] for i in e["widx"]] for e in te_ex]
    result["gate_lid_source"] = "CRF"

if generate:
    def token_gate(e, text, tg):
        hs = text.split()
        if len(hs) != len(e["words"]):
            return text
        return " ".join(copy_word(w, t) if t in COPY else h for w, t, h in zip(e["words"], tg, hs))

    skip = [all(t in COPY for t in tg) for tg in gate_tags]
    keep = [i for i, s in enumerate(skip) if not s]
    _, sub_texts, t_skip = infer(model, [te_ex[i] for i in keep])   # decoder only runs on non-skipped chunks
    skip_texts = [" ".join(copy_word(w, t) for w, t in zip(e["words"], tg)) for e, tg in zip(te_ex, gate_tags)]
    for i, txt in zip(keep, sub_texts):
        skip_texts[i] = txt
    systems = {
        "free": texts,
        "token-gate": [token_gate(e, x, tg) for e, x, tg in zip(te_ex, texts, gate_tags)],
        "chunk-skip": skip_texts,
        "chunk-skip+token-gate": [token_gate(e, x, tg) for e, x, tg in zip(te_ex, skip_texts, gate_tags)],
    }
    result["norm"] = {}
    for name, outs in systems.items():
        hyp = {c: " ".join(v) for c, v in per_comment(te_ex, outs).items()}
        result["norm"][name] = norm_scores(te, hyp)
        log(f"[{TAG}] {name:22s} " + " ".join(f"{k}={result['norm'][name][k]:.2f}" for k in ("CER", "WER", "Exact", "TokAcc")))
    n_words = sum(len(e["words"]) for e in te_ex)
    result["efficiency"] = {
        "skipped_chunk_share": 100 * float(np.mean(skip)),
        "gen_seconds_full": t_full, "gen_seconds_chunk_skip": t_skip,
        "words_per_sec_full": n_words / t_full, "words_per_sec_chunk_skip": n_words / max(t_skip, 1e-9),
        "speedup": t_full / max(t_skip, 1e-9), "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
    }
    json.dump({c: " ".join(v) for c, v in per_comment(te_ex, texts).items()},
              open(f"{args.out}/test_text_{TAG}.json", "w"), ensure_ascii=False)
if args.lam > 0:
    json.dump(pred, open(f"{args.out}/test_lid_{TAG}.json", "w"))
if "lid" in result:
    log(f"[{TAG}] LID " + " ".join(f"{k}={v:.2f}" for k, v in result["lid"].items()))
json.dump(result, open(f"{args.out}/result_{TAG}.json", "w"), indent=1)
log(f"[{TAG}] done")
