# %% [markdown]
# # BanglaMix — Architecture search: Context-aware Token Transducer (CTT)
# **Goal:** one proposed model that beats every Table III baseline on CER, WER *and* EM at once
# (best so far: ByT5-word + CRF tag, 5.17 CER; ByT5-sentence λ=0, 59.4 EM), and an LID head ≥ XLM-R (88.09).
#
# **Why a new architecture.** The sentence-level joint ByT5 decodes whole chunks freely, so it drifts (CER ±1.8
# across seeds) and gating "helps" only by keeping the decoder away from Bengali spans. Word-level ByT5 is more
# accurate but has no context and is trained only on token-aligned comments. Those discarded comments
# hold 13–14% of reference characters and ~19% of all ENG/ROM tokens, which are the hard classes.
#
# **CTT** runs one byte-encoder pass per chunk (full context), then one short decoder pass per word, with that word's
# bytes marked by a learned focus vector. The output is token-aligned by construction (so it cannot drift), the
# decoder sees context, and an aligner recovers the unaligned comments.
#
# ### Stage 1 (seed 42, screening): each row changes one thing from the row above
# | variant | what it tests |
# |---|---|
# | `word_crf` | reproduces the current best (word-level + CRF tag) under this harness |
# | `word_crf_align` | + the 834 recovered unaligned comments |
# | `ctx_crf` | + sentence context (focus decoding) |
# | `ctx_plain` | context only, no LID at all (does LID still matter once context is there?) |
# | `ctx_joint` | **proposed:** own LID head + soft posterior injection + learned copy gate (no external tagger) |
# | `ctx_joint_aks` | + warm start on 200k Aksharantar pairs (targets the unseen Latin words, 39.5% of errors) |
#
# **Stage 1 result (seed 42):** `ctx_plain` won (CER 3.64 / WER 8.45 / EM 61.78; −0.83 CER vs `word_crf`, p<0.001).
# Every form of LID conditioning *hurt* once context was available (ctx_crf +0.65, ctx_joint +0.79). Aksharantar
# helped (ctx_joint_aks −0.42 vs ctx_joint, p=0.02). The learned copy gate skipped 51% of words at no CER cost; the LID
# gate cost about 1.4 CER.
#
# ### Stage 2 (this configuration): confirm on all three splits
# | variant | seeds | why |
# |---|---|---|
# | `ctx_plain_aks_heads` | 13, 42, 2026 | **candidate final model:** ctx_plain + Aksharantar warm start, with LID and copy heads on *detached* encoder states (LID output and copy-gate speed-up without affecting normalization) |
# | `ctx_plain` | 13, 2026 | Stage-1 winner, completes 3 seeds (seed 42 comes from Stage 1) |
# | `word_crf` | 13, 2026 | baseline, completes 3 seeds |
#
# **Setup:** GPU T4 x2, Internet ON (Aksharantar and ByT5 downloads), attach the Kaggle dataset built from `kaggle/kaggle_dataset.zip`.
# Run with *Save Version → Save & Run All*. Jobs run one per GPU, and finished jobs are skipped on rerun.
# The first log lines show minutes per epoch, so multiply by `EPOCHS` to plan. Expect about 1.5–2.5 h per
# focus-mode job on a T4 (fp32) and much less for word mode.

# %%
!pip -q install rapidfuzz 2>/dev/null

# %% writefile banglish_align.py

# %% writefile ctt.py

# %% [markdown]
# ## Configuration

# %%
import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd, torch

OUT = "/kaggle/working/ctt"
SCRIPT = "/kaggle/working/ctt.py"
EPOCHS = 5
# byt5-small is the final backbone. Stage 3b tried "google/byt5-base": +0.15 CER (worse), 2x slower.
# Any other MODEL gets the suffix "_<name>" in variant names and the half-size batch settings below.
MODEL = "google/byt5-small"
if MODEL == "google/byt5-small":
    EXTRA_ARGS = ["--batch_words", "48", "--accum", "2", "--max_chunks", "8", "--gen_bs", "48"]
else:   # same 96 words per optimizer step, half-size micro-batches: fp32 weights + AdamW alone take ~9 GB
    EXTRA_ARGS = ["--batch_words", "24", "--accum", "4", "--max_chunks", "4", "--gen_bs", "32",
                  "--max_enc_tokens", "1024", "--max_dec_tokens", "1024"]
POLL_SECONDS = 120

JOINT = dict(lam=0.4, inject=1, copy_mu=0.5)
VARIANTS = {
    "word_crf":       dict(mode="word", ext_lid="crf", align=0),
    "word_crf_align": dict(mode="word", ext_lid="crf", align=1),
    "ctx_crf":        dict(mode="focus", ext_lid="crf", align=1),
    "ctx_plain":      dict(mode="focus", align=1),
    "ctx_joint":      dict(mode="focus", align=1, **JOINT),
    "ctx_joint_aks":  dict(mode="focus", align=1, pretrain_aksharantar=200000, **JOINT),
    "ctx_plain_aks_heads": dict(mode="focus", align=1, pretrain_aksharantar=200000, lam=0.4, copy_mu=0.5, detach_heads=1),
    # Stage 3a: the proposed model trained longer (validation CER was still falling at epoch 5)
    "ctx_plain_aks_heads_e8": dict(mode="focus", align=1, pretrain_aksharantar=200000, lam=0.4, copy_mu=0.5, detach_heads=1, epochs=8),
}
# which variant runs on which seeds.
#   Stage 1: {v: [42] for v in the first six}
#   Stage 2: {"ctx_plain_aks_heads": [42, 13, 2026], "ctx_plain": [13, 2026], "word_crf": [13, 2026]}
#   Stage 3a: {"ctx_plain_aks_heads_e8": [42, 13, 2026]}  (MODEL = byt5-small) -> no gain over 5 epochs
#   Stage 3b: {"ctx_plain_aks_heads": [42, 13]}           (MODEL = byt5-base)  -> worse, slower
# The search is finished; the proposed model is "ctx_plain_aks_heads" on byt5-small.
# To reproduce it: RUNS = {"ctx_plain_aks_heads": [42, 13, 2026]}
RUNS = {}
SEEDS = sorted({s for ss in RUNS.values() for s in ss})
suffix = "" if MODEL == "google/byt5-small" else "_" + MODEL.split("/")[-1]
JOBS = [dict(variant=v + suffix, seed=s, cfg=VARIANTS[v]) for v, ss in RUNS.items() for s in ss]
# longest first (Aksharantar warm start, then focus mode), so the two GPUs finish at about the same time
JOBS.sort(key=lambda j: (-int("pretrain_aksharantar" in j["cfg"]), j["cfg"]["mode"] != "focus"))
os.makedirs(OUT, exist_ok=True)
JOBS = [j for j in JOBS if not os.path.exists(f"{OUT}/result_{j['variant']}_s{j['seed']}.json")]   # resume-safe
print(len(JOBS), "jobs:", [f"{j['variant']}/s{j['seed']}" for j in JOBS])

# %% [markdown]
# ## Run jobs (one per GPU at a time)

# %%
n_gpu = max(1, torch.cuda.device_count())
queue, running = list(JOBS), {}

def launch(gpu, j):
    logf = open(f"{OUT}/log_{j['variant']}_s{j['seed']}.txt", "w")
    cmd = [sys.executable, SCRIPT, "--variant", j["variant"], "--seed", str(j["seed"]), "--epochs", str(EPOCHS),
           "--model", MODEL, "--out", OUT]
    for k, v in j["cfg"].items():
        cmd += [f"--{k}", str(v)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return subprocess.Popen(cmd + EXTRA_ARGS, stdout=logf, stderr=subprocess.STDOUT, env=env), j, logf

def tail(j):
    try:
        lines = [l for l in open(f"{OUT}/log_{j['variant']}_s{j['seed']}.txt") if l.startswith("[")]
        return lines[-1].strip() if lines else ""
    except FileNotFoundError:
        return ""

t0 = time.time()
while queue or running:
    for g in range(n_gpu):
        if g not in running and queue:
            running[g] = launch(g, queue.pop(0))
    time.sleep(POLL_SECONDS)
    for g, (proc, j, logf) in list(running.items()):
        if proc.poll() is not None:
            logf.close()
            print(f"[{(time.time() - t0) / 60:6.1f} min] finished {j['variant']}/s{j['seed']} exit={proc.returncode}")
            if proc.returncode != 0:
                text = open(logf.name).read()
                print("\n".join(l for l in text.splitlines() if l.startswith("[")))   # progress lines: where it failed
                print(text[-1500:])
            del running[g]
        else:
            print(f"[{(time.time() - t0) / 60:6.1f} min] gpu{g} {tail(j)}")

# %% [markdown]
# ## Results vs. the paper's Table III
# `free` = decode every word; `copy` = learned copy gate (words the copy head is ≥ τ sure about skip the decoder);
# `lid` = the paper's gate (copy words tagged BAN/OTHER). Compare CER, WER and Exact together, because the goal is to
# win all three.

# %%
PAPER = pd.DataFrame([
    ("paper: ByT5-word + LID tag (CRF)", 5.17, 10.76, 55.2),
    ("paper: Joint ByT5 λ=0", 6.59, 9.85, 59.4),
    ("paper: Joint ByT5 + chunk-skip", 6.25, 11.21, 57.0),
], columns=["system", "CER", "WER", "Exact"])

rows_norm, rows_lid, rows_eff = [], [], []
for f in sorted(glob.glob(f"{OUT}/result_*.json")):
    r = json.load(open(f))
    v, s = r["args"]["variant"], r["args"]["seed"]
    for mode, m in r.get("norm", {}).items():
        rows_norm.append(dict(variant=v, mode=mode, seed=s, **m))
    for mode, m in r.get("efficiency", {}).items():
        rows_eff.append(dict(variant=v, mode=mode, seed=s, **m))
    if "lid" in r:
        rows_lid.append(dict(variant=v, seed=s, **r["lid"]))
norm, lid, eff = pd.DataFrame(rows_norm), pd.DataFrame(rows_lid), pd.DataFrame(rows_eff)
norm.to_csv(f"{OUT}/ctt_norm_all.csv", index=False); lid.to_csv(f"{OUT}/ctt_lid_all.csv", index=False)
eff.to_csv(f"{OUT}/ctt_efficiency_all.csv", index=False)

def mean_std(df, keys, cols):
    g = df.groupby(keys, sort=False)
    out = g[cols].mean().round(2).astype(str) + " ± " + g[cols].std().fillna(0).round(2).astype(str)
    out["n_seeds"] = g.size()
    return out

display(PAPER)
if len(norm):
    display(mean_std(norm, ["variant", "mode"], ["CER", "WER", "Exact", "TokAcc", "TokAcc_ENG", "TokAcc_ROM_BAN", "TokAcc_OOV", "TokAcc_direct"]))
if len(eff):
    e = eff.groupby(["variant", "mode"], sort=False)[["decoded_word_share", "words_per_sec"]].mean().round(1)
    free = e.xs("free", level="mode")["words_per_sec"]
    e["speedup_vs_free"] = [round(w / free[v], 2) for (v, _), w in zip(e.index, e.words_per_sec)]
    display(e)
if len(lid):
    display(mean_std(lid, ["variant"], ["Acc", "MacroF1", "F1_ENG", "F1_ROM_BAN", "F1_UNK", "OOV_Acc"]))

# %% [markdown]
# ## Significance: paired, comment-clustered bootstrap on CER (same seed, free decoding)

# %%
import unicodedata
from rapidfuzz.distance import Levenshtein
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
def bn_norm(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")
data = pd.read_csv(glob.glob("/kaggle/input/**/BanglaMix.csv", recursive=True)[0])
ref = {i: " ".join(bn_norm(t).split()) for i, t in zip(data.id, data.bengali_transliteration)}

def paired_bootstrap(a, b, seed, mode="free", n=2000):
    A = json.load(open(f"{OUT}/test_text_{a}_s{seed}_{mode}.json")); B = json.load(open(f"{OUT}/test_text_{b}_s{seed}_free.json"))
    ids = sorted(A)
    ea = np.array([Levenshtein.distance(A[i], ref[i]) for i in ids]); eb = np.array([Levenshtein.distance(B[i], ref[i]) for i in ids])
    ln = np.array([len(ref[i]) for i in ids]); rng = np.random.RandomState(0)
    d = np.array([100 * (ea[k].sum() - eb[k].sum()) / ln[k].sum() for k in (rng.randint(0, len(ids), len(ids)) for _ in range(n))])
    return {"pair": f"{a}[{mode}] vs {b}", "seed": seed, "CER_diff": round(float(100 * (ea.sum() - eb.sum()) / ln.sum()), 3),
            "CI95": np.percentile(d, [2.5, 97.5]).round(3).tolist(), "p(a not better)": float(np.mean(d >= 0))}

for s in SEEDS:
    for a in [v for v in VARIANTS if v != "word_crf"]:
        for mode in ("free", "copy"):
            if os.path.exists(f"{OUT}/test_text_{a + suffix}_s{s}_{mode}.json") and os.path.exists(f"{OUT}/test_text_word_crf_s{s}_free.json"):
                print(paired_bootstrap(a + suffix, "word_crf", s, mode))

# %% [markdown]
# ## Optional: LID ensemble (CTT LID head + XLM-R)
# Attach the nb1 output (it now saves `lid_test_prob_*.json`) as a second input. This averages word-level
# posteriors. The two models make different errors (bytes vs. subwords), so the average can beat XLM-R alone.

# %%
LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]
from sklearn.metrics import f1_score
def parse(tl):
    return [t.rsplit("/", 1)[1] for t in tl.split(" ")]
gold_all = dict(zip(data.id, data.token_labels.map(parse)))
for s in SEEDS:
    xl = glob.glob(f"/kaggle/input/**/lid_test_prob_xlm-roberta-*_s{s}_ce.json", recursive=True)
    ct = [f for f in glob.glob(f"{OUT}/test_lidprob_*_s{s}.json")]
    if not (xl and ct):
        print(f"seed {s}: need both XLM-R probs ({len(xl)}) and CTT probs ({len(ct)}); skipping"); continue
    X = json.load(open(xl[0]))
    for f in ct:
        C = json.load(open(f))
        ids = [c for c in X if c in C]
        for w_ in (0.0, 0.3, 0.5, 0.7, 1.0):
            yt, yp = [], []
            for c in ids:
                P = w_ * np.array(X[c]) + (1 - w_) * np.array(C[c])
                yt += gold_all[c]; yp += [LABELS[i] for i in P.argmax(-1)]
            print(f"seed {s} {os.path.basename(f)} w_xlmr={w_}: macro-F1 {100 * f1_score(yt, yp, labels=LABELS, average='macro'):.2f}")
# Pick the weight on the validation split before reporting it; this cell is only a quick look at test.
