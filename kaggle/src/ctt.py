"""Context-aware Token Transducer (CTT) for BanglaMix normalisation (+ joint LID).

Why: the paper's results show two failure modes.
  * Sentence-level ByT5 (incl. the joint model) decodes a whole chunk freely: it drifts, CER varies by
    +-1.8 across seeds, and gating helps only because it stops the decoder from touching Bengali spans.
  * Word-level ByT5 is the best system (5.17 CER) but sees no context ("ar" -> আর / এর) and is trained only
    on token-aligned comments, which throws away ~19% of all ENG/ROM training tokens.

CTT keeps the strengths of both: ONE byte-level encoder pass per chunk (full context), then ONE short
decoder pass per word. The decoder of word i cross-attends to the whole chunk, with word i's bytes marked by
a learned focus vector. Output length is bounded per word, so there is no drift and the output is always
token-aligned. Unaligned comments are kept via a monotonic token->span aligner (banglish_align.py).

Options (all variants share this code, splits and metrics):
  --mode word|focus|ufal word: each word is its own chunk (no context; = word-level ByT5 under this harness)
                         ufal: UFAL-style per-word re-encoding (Samuel & Straka 2021): the whole chunk is
                         re-encoded once PER WORD with marker tokens around that word, no focus vector
                         (N encoder passes per chunk instead of 1; the baseline CTT's efficiency claim is tested on)
  --ext_lid none|gold|crf|<path with {seed}>   one-hot tags of an external tagger added to each word's bytes
                         (training always uses gold tags, as in nb2; test uses the named tagger)
  --eval_ext_lid gold    also decode the test split with these tag sources (comma list), same trained weights;
                         outputs are written as variant <variant>_<source>tags (oracle-tag check, nb8)
  --lam L --inject 1     joint LID head on pooled word states; soft posteriors injected into the word bytes
  --copy_mu M            learned copy gate: predicts "target == source" per word; at test time confident words
                         skip the decoder (speed-up) -- replaces the LID-based gate that blocks BAN respelling
  --align 0|1            drop or align the 6.6% of comments whose target is not token-aligned
  --pretrain_aksharantar N   first fine-tune on N external Romanized->Bengali pairs (attacks unseen words)
  --model                google/byt5-small (default), google/byt5-base, ...
  --save_val 1           after training, decode the full validation split and report raw and guarded CER
  --data_csv NAME        dataset file under --input (default BanglaMix.csv). A `target_spans` column (JSON list, one
                         target span per source token) is used as given instead of the aligner (MultiLexNorm, nb9)
                         (checks the repetition guard on data it was not designed on)

Run: python ctt.py --variant ctx_joint --mode focus --lam 0.4 --inject 1 --copy_mu 0.5 --seed 42
Smoke test (CPU, no download): python ctt.py --variant smoke --tiny 1 --subsample 300 --epochs 1 --input <dir>
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time
import unicodedata
import zipfile

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from transformers import T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banglish_align import Aligner, join_spans  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True)
p.add_argument("--seed", type=int, default=42, help="data split (split_seed<seed>.csv); also the training seed unless --train_seed")
p.add_argument("--train_seed", type=int, default=None,
               help="training seed (initialisation, batch order, warm-start sample) on the same split; default: --seed")
p.add_argument("--mode", choices=["focus", "word", "ufal"], default="focus")
p.add_argument("--model", default="google/byt5-small")
p.add_argument("--tiny", type=int, default=0, help="smoke tests: random tiny T5, no download")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch_words", type=int, default=48, help="decoded words per micro-batch")
p.add_argument("--max_chunks", type=int, default=8, help="chunks per micro-batch")
p.add_argument("--accum", type=int, default=2)
# Memory is set by PADDED sizes, not word counts: sorting by length puts the rare 300-byte tokens together.
p.add_argument("--max_enc_tokens", type=int, default=2048, help="padded encoder bytes per micro-batch (chunks x longest)")
p.add_argument("--max_dec_tokens", type=int, default=2048, help="padded decoder bytes per micro-batch (words x longest target)")
p.add_argument("--max_word_bytes", type=int, default=64, help="longer tokens (0.1%%: emoji runs, glued text) are copied, not decoded")
p.add_argument("--src_budget", type=int, default=160, help="max source bytes per chunk (words are never split)")
p.add_argument("--max_word_tgt", type=int, default=160, help="max target bytes per word span")
p.add_argument("--lam", type=float, default=0.0, help="LID loss weight")
p.add_argument("--inject", type=int, default=0)
p.add_argument("--ext_lid", default="none")
p.add_argument("--eval_ext_lid", default="", help="extra test-time tag sources (comma list, e.g. gold), same weights")
p.add_argument("--copy_mu", type=float, default=0.0)
p.add_argument("--copy_tau", type=float, default=0.9)
p.add_argument("--detach_heads", type=int, default=0,
               help="LID/copy heads read detached encoder states: they cannot pull the encoder away from normalisation")
p.add_argument("--align", type=int, default=1)
p.add_argument("--pretrain_aksharantar", type=int, default=0)
p.add_argument("--pretrain_epochs", type=int, default=1)
p.add_argument("--val_comments", type=int, default=400)
p.add_argument("--gen_bs", type=int, default=48, help="words per generate() call")
p.add_argument("--input", default="/kaggle/input")
p.add_argument("--out", default="/kaggle/working/ctt")
p.add_argument("--subsample", type=int, default=0, help="smoke tests only: use N random comments")
p.add_argument("--data_csv", default="BanglaMix.csv", help="dataset CSV under --input (MultiLexNorm: mln.csv, nb9)")
p.add_argument("--save_val", type=int, default=0, help="decode the full validation split after training (raw + guarded)")
args = p.parse_args()
assert not (args.inject and args.lam == 0), "injection needs a trained LID head (lam > 0)"
assert not (args.inject and args.detach_heads), "detached heads are read-only side outputs; do not inject them"
EVAL_SOURCES = [s for s in args.eval_ext_lid.split(",") if s]
assert not (EVAL_SOURCES and args.ext_lid == "none"), "--eval_ext_lid needs a model trained with --ext_lid"
os.makedirs(args.out, exist_ok=True)
TAG = f"{args.variant}_s{args.seed}"

TRAIN_SEED = args.seed if args.train_seed is None else args.train_seed
random.seed(TRAIN_SEED); np.random.seed(TRAIN_SEED); torch.manual_seed(TRAIN_SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# T5 overflows in fp16. bf16 only where the hardware has it: is_bf16_supported() is True on a T4 through slow
# emulation, so check the compute capability instead (T4 = 7.5 -> fp32).
BF16 = DEV == "cuda" and torch.cuda.get_device_capability()[0] >= 8
LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]
L2I = {l: i for i, l in enumerate(LABELS)}
NO_TAG = len(LABELS)
EOS, OFFSET = 1, 3                                           # ByT5: id = byte + 3, eos = 1, pad = 0
MARK_L, MARK_R = 259, 260                                   # ufal mode: unused ByT5 extra ids mark the target word
MAX_SRC = 2 * args.src_budget
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def log(*a):
    print(*a, flush=True)


def find(name, required=True):
    hits = glob.glob(f"{args.input}/**/{name}", recursive=True)
    assert hits or not required, f"{name} not found under {args.input}"
    return hits[0] if hits else None


def bn_norm(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


def copy_variants(w):
    c = bn_norm(w)
    return c, c.translate(BN_DIGITS)


_RUN = re.compile(r"(.{1,4}?)\1+", re.S)


def _max_run(s):
    return max([len(m.group(0)) // len(m.group(1)) for m in _RUN.finditer(s)] or [1])


def rep_guard(span, src):
    """Clamp a 1-4-char unit repeated in an output span to max(3, longest repeat run in the source token + 2).
    Fixes decoder loops on elongated tokens ("Reignsssss" -> 130 x "s") that reach the length cap."""
    allowed = max(3, _max_run(bn_norm(src).lower()) + 2)

    def clamp(m):
        unit, n = m.group(1), len(m.group(0)) // len(m.group(1))
        return unit * min(allowed, n) + m.group(0)[n * len(unit):]
    return _RUN.sub(clamp, span)


def copy_class(w, t):
    """0 = needs generation, 1 = target is the source, 2 = target is the source with Bengali digits."""
    if t is None:
        return -100
    c, d = copy_variants(w)
    return 1 if t == c else 2 if t == d else 0


# ------------------------------------------------------------------ data
df = pd.read_csv(find(args.data_csv), keep_default_na=False)
if args.subsample:
    df = df.sample(args.subsample, random_state=0).reset_index(drop=True)


def parse(tl):
    pairs = [t.rsplit("/", 1) for t in tl.split(" ")]
    return [x[0] for x in pairs], [x[1] for x in pairs]


df["tokens"], df["labels"] = zip(*df.token_labels.map(parse))
df["targets"] = df.bengali_transliteration.map(lambda s: bn_norm(s).split())
df["aligned"] = [len(a) == len(b) for a, b in zip(df.tokens, df.targets)]
# Datasets that already give one target span per source token (MultiLexNorm: 1-n spans, empty spans for merges)
GIVEN_SPANS = ({c: [bn_norm(x) for x in json.loads(s)] for c, s in zip(df.id, df.target_spans)}
               if "target_spans" in df.columns else None)
if GIVEN_SPANS is not None:
    assert all(len(GIVEN_SPANS[c]) == len(t) for c, t in zip(df.id, df.tokens)), "one target span per token"
df = df.merge(pd.read_csv(find(f"split_seed{args.seed}.csv")), on="id")
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
train_vocab = set(bn_norm(w).lower() for ws in tr.tokens for w in ws)

aligner = Aligner({(w.lower(), y) for r in tr[tr.aligned].itertuples() for w, y in zip(r.tokens, r.targets)})
align_stats = {"aligned": 0, "recovered": 0, "dropped": 0}


def word_targets(r):
    if GIVEN_SPANS is not None:
        align_stats["aligned"] += 1
        return list(GIVEN_SPANS[r.id])
    if r.aligned:
        align_stats["aligned"] += 1
        return list(r.targets)
    spans = aligner.align(list(r.tokens), list(r.targets))[0] if args.align else None
    align_stats["recovered" if spans is not None else "dropped"] += 1
    return spans


def ext_tags_for(part, train, source=None):
    """External tags per comment: gold for training, the named tagger (or `source`) at test time."""
    source = source or args.ext_lid
    if source == "none":
        return None
    if train or source == "gold":
        return {r.id: list(r.labels) for r in part.itertuples()}
    path = find(f"crf_pred_seed{args.seed}.json") if source == "crf" else source.format(seed=args.seed)
    pred = json.load(open(path))
    return {c: pred[c] for c in part.id}


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


def tgt_ids(t):
    return [b + OFFSET for b in t.encode("utf-8")][:args.max_word_tgt - 1] + [EOS]


def build_chunk(cid, k, idx, words, labels, ext, tgts):
    ids, spans, nodec = [], [], []
    for j, w in enumerate(words):
        if j:
            ids.append(ord(" ") + OFFSET)
        raw = w.encode("utf-8")
        nodec.append(len(raw) > args.max_word_bytes)                 # overlong token: encoded (cut) for LID, then copied
        b = [x + OFFSET for x in raw[:args.max_word_bytes]]
        spans.append((len(ids), len(ids) + len(b)))
        ids += b
    ids = ids[:MAX_SRC - 1] + [EOS]
    spans = [(s, min(e, len(ids) - 1)) for s, e in spans]         # words cut by truncation get empty spans
    return dict(cid=cid, chunk=k, widx=idx, words=words, ids=ids, spans=spans, nodec=nodec,
                lid=[L2I[l] if l in L2I else -100 for l in labels],
                ext=[L2I.get(t, NO_TAG) for t in ext] if ext else [NO_TAG] * len(words),
                tgt=tgts, tgt_ids=[tgt_ids(t) if t is not None and not nd else None for t, nd in zip(tgts, nodec)],
                copy=[copy_class(w, t) for w, t in zip(words, tgts)])


def build_ufal(cid, k, p_, idx, words, labels, ext, tgts):
    """UFAL-style example: the whole chunk, with word p_ wrapped in marker tokens; only word p_ is decoded."""
    ids, span, nd = [], (0, 0), False
    for j, w in enumerate(words):
        if j:
            ids.append(ord(" ") + OFFSET)
        raw = w.encode("utf-8")
        b = [x + OFFSET for x in raw[:args.max_word_bytes]]
        if j == p_:
            nd = len(raw) > args.max_word_bytes
            ids.append(MARK_L)
            span = (len(ids), len(ids) + len(b))
            ids += b
            ids.append(MARK_R)
        else:
            ids += b
    ids = ids[:MAX_SRC - 1] + [EOS]
    span = (span[0], min(span[1], len(ids) - 1))
    t = tgts[p_]
    return dict(cid=cid, chunk=(k, p_), widx=[idx[p_]], words=[words[p_]], ids=ids, spans=[span], nodec=[nd],
                lid=[L2I[labels[p_]] if labels[p_] in L2I else -100],
                ext=[L2I.get(ext[p_], NO_TAG)] if ext else [NO_TAG],
                tgt=[t], tgt_ids=[tgt_ids(t) if t is not None and not nd else None], copy=[copy_class(words[p_], t)])


def make_examples(part, train, source=None):
    ext = ext_tags_for(part, train, source)
    out = []
    for r in part.itertuples():
        tg = word_targets(r) if train else None
        idxs = [[i] for i in range(len(r.tokens))] if args.mode == "word" else chunk_words(r.tokens, args.src_budget)
        for k, idx in enumerate(idxs):
            if args.mode == "ufal":
                ws, ls = [r.tokens[i] for i in idx], [r.labels[i] for i in idx]
                es = [ext[r.id][i] for i in idx] if ext else None
                ts = [tg[i] for i in idx] if tg is not None else [None] * len(idx)
                out += [build_ufal(r.id, k, p_, idx, ws, ls, es, ts) for p_ in range(len(idx))]
                continue
            out.append(build_chunk(r.id, k, idx, [r.tokens[i] for i in idx], [r.labels[i] for i in idx],
                                   [ext[r.id][i] for i in idx] if ext else None,
                                   [tg[i] for i in idx] if tg is not None else [None] * len(idx)))
    return out


def load_aksharantar(n):
    """N random (romanised, Bengali) word pairs from AI4Bharat Aksharantar (ben.zip, train file)."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("ai4bharat/Aksharantar", "ben.zip", repo_type="dataset")
    pairs = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if "train" not in name.lower() or not name.endswith(".json"):
                continue
            text = z.read(name).decode("utf-8")
            recs = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
            pairs += [(r["english word"], bn_norm(r["native word"])) for r in recs if r.get("english word") and r.get("native word")]
    rng = random.Random(TRAIN_SEED)
    rng.shuffle(pairs)
    log(f"[{TAG}] aksharantar: {len(pairs)} pairs available, using {min(n, len(pairs))}")
    return [build_chunk(f"aks{i}", 0, [0], [w], ["?"], None, [t]) for i, (w, t) in enumerate(pairs[:n])]


t0 = time.time()
tr_ex = make_examples(tr, train=True)
va_sel = va.sample(min(args.val_comments, len(va)), random_state=0)
va_ex, te_ex = make_examples(va_sel, train=False), make_examples(te, train=False)
log(f"[{TAG}] chunks train {len(tr_ex)} val {len(va_ex)} test {len(te_ex)} | align {align_stats} "
    f"({time.time() - t0:.0f}s) | device {DEV} bf16 {BF16}")


def collate(batch, dec_words=None):
    """dec_words: list of (example index in batch, word index) to decode; default: every word with a target."""
    B = len(batch)
    T = max(len(e["ids"]) for e in batch)
    W = max(len(e["spans"]) for e in batch)
    ids = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    assign = torch.zeros(B, W, T)
    lid = torch.full((B, W), -100, dtype=torch.long)
    ext = torch.full((B, W), NO_TAG, dtype=torch.long)
    cpy = torch.full((B, W), -100, dtype=torch.long)
    for i, e in enumerate(batch):
        ids[i, :len(e["ids"])] = torch.tensor(e["ids"])
        att[i, :len(e["ids"])] = 1
        ext[i, :len(e["ext"])] = torch.tensor(e["ext"])
        for w, (s, t) in enumerate(e["spans"]):
            if t > s:
                assign[i, w, s:t] = 1.0
                lid[i, w] = e["lid"][w]
                cpy[i, w] = e["copy"][w]
    if dec_words is None:
        dec_words = [(i, w) for i, e in enumerate(batch) for w, (s, t) in enumerate(e["spans"])
                     if t > s and e["tgt_ids"][w] is not None]
    bi = torch.tensor([i for i, _ in dec_words], dtype=torch.long)
    wi = torch.tensor([w for _, w in dec_words], dtype=torch.long)
    labels = None
    if dec_words and batch[dec_words[0][0]]["tgt_ids"][dec_words[0][1]] is not None:
        L = max(len(batch[i]["tgt_ids"][w]) for i, w in dec_words)
        labels = torch.full((len(dec_words), L), -100, dtype=torch.long)
        for n, (i, w) in enumerate(dec_words):
            labels[n, :len(batch[i]["tgt_ids"][w])] = torch.tensor(batch[i]["tgt_ids"][w])
    return dict(ids=ids, att=att, assign=assign, lid=lid, ext=ext, copy=cpy, bi=bi, wi=wi, labels=labels)


def n_dec(e):
    return sum(1 for (s, t), y in zip(e["spans"], e["tgt_ids"]) if t > s and y is not None)


def max_chunks_for(ex):
    """Single-word chunks (word mode, Aksharantar) fill a batch by words, not by the context-chunk cap."""
    return args.batch_words if all(len(e["words"]) == 1 for e in ex[:200]) else args.max_chunks


def max_tgt_len(e):
    return max([len(y) for (s, t), y in zip(e["spans"], e["tgt_ids"]) if t > s and y is not None] or [1])


def batches(ex, shuffle, rng, infer=False):
    """Length-bucketed batches capped by words, chunks, and PADDED encoder/decoder sizes.
    infer=True: no targets; caps are doubled (no activations kept) and the decoder cap is left to generate()."""
    idx = rng.permutation(len(ex)) if shuffle else np.arange(len(ex))
    f = 2 if infer else 1
    cap, max_w, max_enc = f * max_chunks_for(ex), f * args.batch_words, f * args.max_enc_tokens
    out = []
    for m in range(0, len(idx), 2000):                              # length bucketing inside mega-batches
        mega = sorted(idx[m:m + 2000], key=lambda i: len(ex[i]["ids"]))
        cur, words, T, L = [], 0, 0, 0
        for i in mega:
            e = ex[i]
            k = max(1, len(e["words"]) if infer else n_dec(e))
            T2, L2 = max(T, len(e["ids"])), max(L, 1 if infer else max_tgt_len(e))
            if cur and (words + k > max_w or len(cur) >= cap or (len(cur) + 1) * T2 > max_enc
                        or (not infer and (words + k) * L2 > args.max_dec_tokens)):
                out.append(cur)
                cur, words, T2, L2 = [], 0, len(e["ids"]), (1 if infer else max_tgt_len(e))
            cur.append(i)
            words, T, L = words + k, T2, L2
        if cur:
            out.append(cur)
    if shuffle:
        rng.shuffle(out)
    return out


FAKE_OOM = int(os.environ.get("CTT_FAKE_OOM_WORDS", "0"))   # tests only: raise OOM above this many decoded words


def maybe_fake_oom(n_words):
    if FAKE_OOM and n_words > FAKE_OOM:
        raise torch.OutOfMemoryError(f"fake OOM ({n_words} words > {FAKE_OOM})")


def free_cuda():
    if DEV == "cuda":
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ model
class CTT(nn.Module):
    def __init__(self):
        super().__init__()
        if args.tiny:
            cfg = T5Config(vocab_size=384, d_model=64, d_kv=16, num_heads=4, d_ff=128, num_layers=2,
                           num_decoder_layers=1, feed_forward_proj="gated-gelu", tie_word_embeddings=False,
                           decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
            self.t5 = T5ForConditionalGeneration(cfg)
        else:
            self.t5 = T5ForConditionalGeneration.from_pretrained(args.model)
        H = self.t5.config.d_model
        self.drop = nn.Dropout(0.1)
        self.lid = nn.Linear(H, len(LABELS))
        self.copy = nn.Linear(H, 3)
        self.tag = nn.Linear(len(LABELS), H, bias=False)      # LID posterior injection, starts at zero
        self.ext = nn.Embedding(NO_TAG + 1, H)                # external one-hot tag injection, starts at zero
        self.focus = nn.Embedding(2, H)                       # 0: context byte, 1: byte of the decoded word
        for w in (self.tag.weight, self.ext.weight, self.focus.weight):
            nn.init.zeros_(w)
        self.register_buffer("focus_ready", torch.zeros(()))

    def encode(self, b):
        h = self.t5.get_encoder()(input_ids=b["ids"], attention_mask=b["att"]).last_hidden_state
        A = b["assign"].to(h.dtype)
        u = torch.bmm(A / A.sum(-1, keepdim=True).clamp_min(1.0), h)
        if args.detach_heads:
            u = u.detach()
        lid_logits, copy_logits = self.lid(self.drop(u)), self.copy(self.drop(u))
        add = self.ext(b["ext"]).to(h.dtype)
        if args.inject:
            add = add + self.tag(F.softmax(lid_logits.float(), -1).to(h.dtype))
        return h + torch.bmm(A.transpose(1, 2), add), lid_logits, copy_logits

    def focus_states(self, h, b, sel=None):
        """One copy of the chunk states per decoded word, with that word's bytes marked."""
        if not bool(self.focus_ready) and args.mode != "ufal":
            with torch.no_grad():                             # scale the marker to the encoder's output scale
                self.focus.weight[1].normal_(0, float(h.detach().float().std()))
                self.focus_ready.fill_(1)
        bi, wi = (b["bi"], b["wi"]) if sel is None else (b["bi"][sel], b["wi"][sel])
        if args.mode == "ufal":                               # markers in the input already say which word to decode
            return h[bi], b["att"][bi]
        m = b["assign"][bi, wi].unsqueeze(-1).to(h.dtype)
        f0, f1 = self.focus.weight[0].to(h.dtype), self.focus.weight[1].to(h.dtype)
        return h[bi] + m * f1 + (1 - m) * f0, b["att"][bi]

    def forward(self, b):
        h, lid_logits, copy_logits = self.encode(b)
        seq_logits = None
        if b["labels"] is not None and len(b["bi"]):
            hf, att = self.focus_states(h, b)
            lab = b["labels"]
            dec = torch.cat([torch.full_like(lab[:, :1], self.t5.config.decoder_start_token_id), lab[:, :-1]], 1)
            dec = dec.masked_fill(dec == -100, self.t5.config.pad_token_id)
            seq_logits = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=hf), attention_mask=att,
                                 decoder_input_ids=dec).logits
        return seq_logits, lid_logits, copy_logits


def to_dev(b):
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}


def decode_bytes(seq):
    out = []
    for x in seq.tolist():
        if x == EOS:
            break
        if OFFSET <= x < OFFSET + 256:
            out.append(x - OFFSET)
    return bytes(out).decode("utf-8", errors="ignore").strip(" ")


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


INFER_STATS = {}


@torch.no_grad()
def infer(model, ex, gate=None):
    """gate: None (decode every word) | 'copy' (learned copy head) | 'lid' (copy BAN/OTHER by LID tag).
    Returns per-example dicts with lid tags/probs and word spans, plus generation seconds and words decoded."""
    model.eval()
    res, gen_time, n_gen = [None] * len(ex), 0.0, 0
    INFER_STATS["enc_seconds"], INFER_STATS["enc_bytes"] = 0.0, 0
    for idx in batches(ex, False, None, infer=True):
        batch = [ex[i] for i in idx]
        allw = [(i, w) for i, e in enumerate(batch) for w in range(len(e["spans"]))]
        b = to_dev(collate(batch, dec_words=allw))
        sync(); t_enc = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
            h, lg, cg = model.encode(b)
        sync(); INFER_STATS["enc_seconds"] += time.time() - t_enc
        INFER_STATS["enc_bytes"] += sum(len(e["ids"]) for e in batch)
        lp, cp = F.softmax(lg.float(), -1).cpu(), F.softmax(cg.float(), -1).cpu()
        ext_tags = b["ext"].cpu()
        out = []
        for j, e in enumerate(batch):
            tags = [LABELS[int(lp[j, w].argmax())] if t > s else "OTHER" for w, (s, t) in enumerate(e["spans"])]
            out.append(dict(tags=tags, probs=lp[j, :len(e["spans"])].numpy().round(4).tolist(), spans=[None] * len(e["spans"])))
        # decide per word: copy without decoding, or generate
        gen_sel = []
        for n, (j, w) in enumerate(allw):
            e = batch[j]
            s, t = e["spans"][w]
            c, d = copy_variants(e["words"][w])
            if t <= s or e["nodec"][w]:                                   # truncated away or overlong: copy
                out[j]["spans"][w] = c
                continue
            if gate == "copy" and float(cp[j, w, 1:].sum()) >= args.copy_tau:
                out[j]["spans"][w] = c if cp[j, w, 1] >= cp[j, w, 2] else d
                continue
            if gate == "lid":
                tag = out[j]["tags"][w] if args.lam > 0 else (LABELS[int(ext_tags[j, w])] if int(ext_tags[j, w]) < NO_TAG else None)
                if tag in ("BAN", "OTHER"):
                    out[j]["spans"][w] = d if tag == "OTHER" else c
                    continue
            gen_sel.append(n)
        g0, gbs = 0, args.gen_bs
        while g0 < len(gen_sel):
            part = gen_sel[g0:g0 + gbs]
            maxb = max(len(batch[allw[n][0]]["words"][allw[n][1]].encode("utf-8")) for n in part)
            try:
                maybe_fake_oom(len(part))
                sync(); t1 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
                    hf, att = model.focus_states(h, b, torch.tensor(part, dtype=torch.long, device=DEV))
                    g = model.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=hf), attention_mask=att,
                                          max_new_tokens=min(args.max_word_tgt, 4 * min(maxb, args.max_word_bytes) + 24),
                                          num_beams=1, do_sample=False)
                sync(); gen_time += time.time() - t1
            except torch.OutOfMemoryError:
                hf = att = g = None
                free_cuda()
                if gbs == 1:
                    raise
                gbs = max(1, gbs // 2)
                log(f"[{TAG}] OOM in generate: retrying with {gbs} words per call")
                continue
            n_gen += len(part)
            for k, n in enumerate(part):
                j, w = allw[n]
                out[j]["spans"][w] = decode_bytes(g[k])
            g0 += len(part)
        for j, i in enumerate(idx):
            res[i] = out[j]
    return res, gen_time, n_gen


def per_comment(ex, res, key):
    acc = {}
    for e, r in zip(ex, res):
        acc.setdefault(e["cid"], {})[e["chunk"]] = r[key]
    return {c: [x for k in sorted(v) for x in v[k]] for c, v in acc.items()}


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


def norm_scores(part, spans):
    """Same definitions as joint_mtl.py (CER/WER/EM on full references, TokAcc on aligned comments with a
    matching word count), plus TokAcc_direct: per-token span accuracy on aligned comments."""
    H = [join_spans(spans[c]) for c in part.id]
    refs = [" ".join(t) for t in part.targets]
    s = {"CER": 100 * sum(Levenshtein.distance(h, r) for h, r in zip(H, refs)) / sum(map(len, refs)),
         "WER": 100 * sum(Levenshtein.distance(h.split(), r.split()) for h, r in zip(H, refs)) / sum(len(r.split()) for r in refs),
         "Exact": 100 * float(np.mean([h == r for h, r in zip(H, refs)]))}
    ok, lab, oov, n_al, n_match, direct = [], [], [], 0, 0, []
    for h, r in zip(H, part.itertuples()):
        if not r.aligned:
            continue
        n_al += 1
        direct += [a == b for a, b in zip(spans[r.id], r.targets)]
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
    s["TokAcc_direct"] = 100 * float(np.mean(direct)) if direct else float("nan")
    s["word_count_match_share"] = 100 * n_match / max(1, n_al)
    return s


def validate_epoch(model):
    res, _, _ = infer(model, va_ex)
    rec = {"val_CER": norm_scores(va_sel, per_comment(va_ex, res, "spans"))["CER"]}
    if args.lam > 0:
        vt = per_comment(va_ex, res, "tags")
        rec["val_LID_MacroF1"] = lid_scores(list(va_sel.labels), [vt[c] for c in va_sel.id], list(va_sel.tokens))["MacroF1"]
    return rec


# ------------------------------------------------------------------ train
def batch_loss(model, b):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
        seq_logits, lid_logits, copy_logits = model(b)
    loss = torch.zeros((), device=DEV)
    if seq_logits is not None:
        loss = loss + F.cross_entropy(seq_logits.float().reshape(-1, seq_logits.shape[-1]),
                                      b["labels"].reshape(-1), ignore_index=-100)
    if args.lam > 0 and (b["lid"] != -100).any():   # guard: CE over an all-ignored batch is NaN
        loss = loss + args.lam * F.cross_entropy(lid_logits.float().reshape(-1, len(LABELS)), b["lid"].reshape(-1), ignore_index=-100)
    if args.copy_mu > 0 and (b["copy"] != -100).any():
        loss = loss + args.copy_mu * F.cross_entropy(copy_logits.float().reshape(-1, 3), b["copy"].reshape(-1), ignore_index=-100)
    return loss


def train(model, ex, epochs, lr, stage, validate=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(TRAIN_SEED)
    plan = batches(ex, False, rng)
    steps = epochs * (len(plan) // args.accum + 1)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    enc = [len(b) * max(len(ex[i]["ids"]) for i in b) for b in plan]
    dec = [sum(n_dec(ex[i]) for i in b) * max(max_tgt_len(ex[i]) for i in b) for b in plan]
    log(f"[{TAG}] {stage}: {len(plan)} micro-batches | padded encoder bytes max {max(enc)} mean {np.mean(enc):.0f} "
        f"| padded decoder bytes max {max(dec)} mean {np.mean(dec):.0f}")
    best, best_state, history = None, None, []
    for ep in range(epochs):
        model.train()
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0, tot, n, n_oom, n_skip = time.time(), 0.0, 0, 0, 0
        all_b = batches(ex, True, rng)
        opt.zero_grad()
        for k, bidx in enumerate(all_b):
            pending = [list(bidx)]           # a micro-batch that runs out of memory is split in half and retried
            while pending:
                part = pending.pop(0)
                try:
                    b = to_dev(collate([ex[i] for i in part]))
                    maybe_fake_oom(len(b["bi"]))
                    loss = batch_loss(model, b)
                    if loss.requires_grad:
                        (loss * len(part) / len(bidx) / args.accum).backward()
                        tot, n = tot + loss.item(), n + 1
                except torch.OutOfMemoryError:
                    b = loss = None
                    opt.zero_grad(set_to_none=True)     # an OOM inside backward may leave partial gradients
                    free_cuda()
                    n_oom += 1
                    if len(part) == 1:
                        n_skip += 1
                        log(f"[{TAG}] OOM on a single chunk ({len(ex[part[0]]['ids'])} bytes): skipped")
                        continue
                    pending = [part[:len(part) // 2], part[len(part) // 2:]] + pending
            if (k + 1) % args.accum == 0 or k + 1 == len(all_b):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
        rec = {"stage": stage, "epoch": ep, "train_loss": tot / max(1, n), "train_min": (time.time() - t0) / 60,
               "oom_splits": n_oom, "oom_skipped": n_skip}
        if DEV == "cuda":
            rec["peak_GB"] = torch.cuda.max_memory_allocated() / 2 ** 30
        if validate:
            rec.update(validate_epoch(model))
        history.append(rec)
        log(f"[{TAG}] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        if validate and (best is None or rec["val_CER"] < best):
            best, best_state = rec["val_CER"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if DEV == "cuda":
            torch.cuda.empty_cache()
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


model = CTT().to(DEV)
history = []
if args.pretrain_aksharantar:
    history += train(model, load_aksharantar(args.pretrain_aksharantar), args.pretrain_epochs, args.lr, "aksharantar", validate=False)
history += train(model, tr_ex, args.epochs, args.lr, "banglamix")


# ------------------------------------------------------------------ test
result = {"args": vars(args), "history": history, "align": align_stats,
          "n_test_comments": len(te), "n_test_chunks": len(te_ex), "n_test_words": int(sum(len(e["words"]) for e in te_ex))}
gates = [None] + (["copy"] if args.copy_mu > 0 else []) + (["lid"] if args.lam > 0 or args.ext_lid != "none" else [])
result["norm"], result["efficiency"] = {}, {}
for gate in gates:
    res, secs, n_gen = infer(model, te_ex, gate=gate)
    name = gate or "free"
    spans = per_comment(te_ex, res, "spans")
    result["norm"][name] = norm_scores(te, spans)
    result["efficiency"][name] = {"gen_seconds": secs, "decoded_word_share": 100 * n_gen / result["n_test_words"],
                                  "words_per_sec": result["n_test_words"] / max(secs, 1e-9),
                                  # end-to-end: encoder passes + generation (the cost UFAL-style re-encoding adds)
                                  "enc_seconds": INFER_STATS["enc_seconds"], "encoder_bytes": INFER_STATS["enc_bytes"],
                                  "total_words_per_sec": result["n_test_words"] / max(secs + INFER_STATS["enc_seconds"], 1e-9)}
    log(f"[{TAG}] {name:5s} " + " ".join(f"{k}={result['norm'][name][k]:.2f}" for k in ("CER", "WER", "Exact", "TokAcc", "TokAcc_OOV"))
        + f" decoded={result['efficiency'][name]['decoded_word_share']:.1f}% words/s={result['efficiency'][name]['words_per_sec']:.0f}"
        + f" end-to-end words/s={result['efficiency'][name]['total_words_per_sec']:.0f} encoder_bytes={INFER_STATS['enc_bytes']}")
    json.dump({c: join_spans(v) for c, v in spans.items()}, open(f"{args.out}/test_text_{TAG}_{name}.json", "w"), ensure_ascii=False)
    # repetition guard (post-processing, no retraining): clamp runaway repeats such as "sssss…" / "?????…"
    words = dict(zip(te.id, te.tokens))
    gspans = {c: [rep_guard(x, w) for x, w in zip(v, words[c])] for c, v in spans.items()}
    result["norm"][f"{name}+guard"] = norm_scores(te, gspans)
    result["efficiency"][f"{name}+guard"] = result["efficiency"][name]
    log(f"[{TAG}] {name + '+guard':11s} " + " ".join(f"{k}={result['norm'][name + '+guard'][k]:.2f}" for k in ("CER", "WER", "Exact")))
    json.dump({c: join_spans(v) for c, v in gspans.items()}, open(f"{args.out}/test_text_{TAG}_{name}+guard.json", "w"), ensure_ascii=False)
    if gate is None:
        if args.lam > 0:
            tags = per_comment(te_ex, res, "tags")
            result["lid"] = lid_scores(list(te.labels), [tags[c] for c in te.id], list(te.tokens))
            json.dump(tags, open(f"{args.out}/test_lid_{TAG}.json", "w"))
            json.dump(per_comment(te_ex, res, "probs"), open(f"{args.out}/test_lidprob_{TAG}.json", "w"))
            log(f"[{TAG}] LID " + " ".join(f"{k}={v:.2f}" for k, v in result["lid"].items()))
        json.dump({c: v for c, v in spans.items()}, open(f"{args.out}/test_spans_{TAG}.json", "w"), ensure_ascii=False)
for src in EVAL_SOURCES:
    # Same weights, other test-time tags: separates the effect of the tag source from training noise.
    ex = make_examples(te, train=False, source=src)
    res, secs, _ = infer(model, ex)
    spans = per_comment(ex, res, "spans")
    words = dict(zip(te.id, te.tokens))
    gspans = {c: [rep_guard(x, w) for x, w in zip(v, words[c])] for c, v in spans.items()}
    result.setdefault("norm_ext", {})[src] = {"free": norm_scores(te, spans), "free+guard": norm_scores(te, gspans)}
    vtag = f"{args.variant}_{src}tags_s{args.seed}"
    json.dump(spans, open(f"{args.out}/test_spans_{vtag}.json", "w"), ensure_ascii=False)
    json.dump({c: join_spans(v) for c, v in gspans.items()}, open(f"{args.out}/test_text_{vtag}_free+guard.json", "w"), ensure_ascii=False)
    log(f"[{TAG}] test with {src} tags: CER={result['norm_ext'][src]['free']['CER']:.2f} "
        f"guarded CER={result['norm_ext'][src]['free+guard']['CER']:.2f}")
if args.save_val:
    # The repetition guard was designed after seeing test outputs; this checks it on the full validation split.
    va_all = make_examples(va, train=False)
    res, _, _ = infer(model, va_all)
    vspans = per_comment(va_all, res, "spans")
    vwords = dict(zip(va.id, va.tokens))
    vg = {c: [rep_guard(x, w) for x, w in zip(v, vwords[c])] for c, v in vspans.items()}
    result["val"] = {"free": norm_scores(va, vspans), "free+guard": norm_scores(va, vg),
                     "guard_changed_words": int(sum(a != b for c in vspans for a, b in zip(vspans[c], vg[c])))}
    log(f"[{TAG}] VAL free CER={result['val']['free']['CER']:.3f} free+guard CER={result['val']['free+guard']['CER']:.3f} "
        f"words changed by guard={result['val']['guard_changed_words']}")
    json.dump(vspans, open(f"{args.out}/val_spans_{TAG}.json", "w"), ensure_ascii=False)
json.dump(result, open(f"{args.out}/result_{TAG}.json", "w"), indent=1)
log(f"[{TAG}] done")
