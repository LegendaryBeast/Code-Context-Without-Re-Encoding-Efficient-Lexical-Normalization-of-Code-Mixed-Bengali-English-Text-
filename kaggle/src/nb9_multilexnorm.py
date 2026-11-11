# %% [markdown]
# # CTT on a second dataset: MultiLexNorm code-switched normalization (Indonesian–English, Turkish–German)
# **Why this notebook.** Reviewers asked whether CTT works beyond BanglaMix, and whether the ÚFAL comparison holds
# against ÚFAL's *published* system. MultiLexNorm (W-NUT 2021) has two code-switched sets with one normalization per
# token, so CTT runs on it unchanged, and ÚFAL's official test outputs exist (ERR on test, official formula:
# **iden 66.15 / 67.18**, **trde 68.19 / 68.62** for ÚFAL-1 / ÚFAL-2).
#
# **What runs** (same `ctt.py` code base and settings as on BanglaMix; training seeds 13, 42, 2026 on the official
# split, so the spread over seeds is training variance only):
#
# | variant | what it is |
# |---|---|
# | `<lang>_word` | word level, no context |
# | `<lang>_ctx` | context-only CTT (one encoder pass, focus vectors) |
# | `<lang>_ctx_copy` | CTT with the learned copy gate (no Aksharantar warm start or LID head: they are Bengali-specific) |
# | `<lang>_ufal` | ÚFAL-style per-word re-encoding, same network and training as `<lang>_ctx` |
#
# Data: the MultiLexNorm repository is cloned at a fixed commit and converted by `mln_convert.py`
# (Indonesian–English uses its official dev split for epoch selection; Turkish–German has none, so 10% of its
# training sentences are held out). MultiLexNorm's own licences apply; nothing is redistributed.
#
# ### How to run (Kaggle)
# 1. No dataset needs to be attached. Settings: **GPU T4 x2**, **Internet ON** (clones MultiLexNorm, downloads ByT5-small).
# 2. Two versions keep each session well under 12 h (the job loop skips finished jobs, so a re-run resumes):
#    - **Version 1:** `LANGS = ["iden"]` (default below)
#    - **Version 2:** `LANGS = ["trde"]`
#    Check the first epoch's `train_min` in a log: the `_ufal` jobs cost about 5× the others.
# 3. *Save Version → Save & Run All*. Download the whole `/kaggle/working/ctt` folder of each version and put the
#    files of both into `results/gpu/nb9_multilexnorm/`. Then run locally: `python analysis/analyze_mln.py`.

# %%
!pip -q install rapidfuzz 2>/dev/null

# %% writefile banglish_align.py

# %% writefile ctt.py

# %% writefile mln_convert.py

# %% [markdown]
# ## Data

# %%
import glob, json, os, shutil, subprocess, sys, time, unicodedata
import numpy as np, pandas as pd, torch

MLN_REPO, MLN_COMMIT = "https://bitbucket.org/robvanderg/multilexnorm.git", "e58e946375299a04355fa2d7fa9e1c769234e68a"
SRC, DATA, OUT = "/kaggle/working/multilexnorm", "/kaggle/working/mln", "/kaggle/working/ctt"
LANGS = ["iden"]            # Version 1; Version 2: ["trde"]
if not os.path.exists(SRC):
    subprocess.run(["git", "clone", "-q", MLN_REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "-q", MLN_COMMIT], check=True)
subprocess.run([sys.executable, "/kaggle/working/mln_convert.py", "--repo", SRC, "--out", DATA, "--langs", *LANGS], check=True)
os.makedirs(OUT, exist_ok=True)
for lang in LANGS:   # keep the converted data with the results, so the analysis is self-contained
    shutil.copy(f"{DATA}/{lang}/mln.csv", f"{OUT}/mln_{lang}.csv")

# %% [markdown]
# ## Configuration and jobs

# %%
SCRIPT = "/kaggle/working/ctt.py"
EPOCHS = 10                 # MultiLexNorm sets are ~10x smaller than BanglaMix; the best epoch is chosen on validation
SEEDS = [42, 13, 2026]
MODEL = "google/byt5-small"
BASE_ARGS = ["--batch_words", "48", "--accum", "2", "--max_chunks", "8", "--gen_bs", "48", "--data_csv", "mln.csv"]
POLL_SECONDS = 60
VARIANTS = {
    "ufal":     (dict(mode="ufal", align=1), ["--max_enc_tokens", "4096"]),
    "ctx":      (dict(mode="focus", align=1), []),
    "ctx_copy": (dict(mode="focus", align=1, copy_mu=0.5, detach_heads=1), []),
    "word":     (dict(mode="word", align=1), []),
}
# most expensive first, so the two GPUs finish at about the same time
JOBS = [dict(lang=l, name=f"{l}_{v}", seed=s, cfg=c, extra=x) for v, (c, x) in VARIANTS.items() for l in LANGS for s in SEEDS]
JOBS = [j for j in JOBS if not os.path.exists(f"{OUT}/result_{j['name']}_s{j['seed']}.json")]   # resume-safe
print(len(JOBS), "jobs:", [f"{j['name']}/s{j['seed']}" for j in JOBS])

# %%
n_gpu = max(1, torch.cuda.device_count())
queue, running = list(JOBS), {}

def launch(gpu, j):
    logf = open(f"{OUT}/log_{j['name']}_s{j['seed']}.txt", "w")
    cmd = [sys.executable, SCRIPT, "--variant", j["name"], "--seed", str(j["seed"]), "--epochs", str(EPOCHS),
           "--model", MODEL, "--input", f"{DATA}/{j['lang']}", "--out", OUT]
    for k, v in j["cfg"].items():
        cmd += [f"--{k}", str(v)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return subprocess.Popen(cmd + BASE_ARGS + j["extra"], stdout=logf, stderr=subprocess.STDOUT, env=env), j, logf

def tail(j):
    try:
        lines = [l for l in open(f"{OUT}/log_{j['name']}_s{j['seed']}.txt") if l.startswith("[")]
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
            print(f"[{(time.time() - t0) / 60:6.1f} min] finished {j['name']}/s{j['seed']} exit={proc.returncode}")
            if proc.returncode != 0:
                text = open(logf.name).read()
                print("\n".join(l for l in text.splitlines() if l.startswith("[")))
                print(text[-1500:])
            del running[g]
        else:
            print(f"[{(time.time() - t0) / 60:6.1f} min] gpu{g} {tail(j)}")

# %% [markdown]
# ## Results in this session: test ERR (official MultiLexNorm formula, case-sensitive)
# ERR = (accuracy − leave-as-is accuracy) / (1 − leave-as-is accuracy), over all test tokens. Published ÚFAL test ERR:
# iden 66.15 / 67.18, trde 68.19 / 68.62 (ÚFAL-1 / ÚFAL-2).

# %%
def err_of(spans, data):
    cor = changed = total = 0
    for cid, toks, gold in data:
        if cid not in spans:     # only in smoke tests with --subsample
            continue
        pred = spans[cid]
        assert len(pred) == len(gold)
        for raw, g, p in zip(toks, gold, pred):
            g, p = unicodedata.normalize("NFC", g), unicodedata.normalize("NFC", p)
            changed += raw != g; cor += g == p; total += 1
    acc, lai = cor / total, (total - changed) / total
    return 100 * (acc - lai) / (1 - lai)

rows = []
for lang in LANGS:
    d = pd.read_csv(f"{OUT}/mln_{lang}.csv", keep_default_na=False)
    d = d[d.id.str.startswith(f"{lang}_test_")]
    data = [(c, [t.rsplit("/", 1)[0] for t in tl.split(" ")], json.loads(s)) for c, tl, s in zip(d.id, d.token_labels, d.target_spans)]
    for f in sorted(glob.glob(f"{OUT}/test_spans_{lang}_*_s*.json")):
        name, seed = os.path.basename(f)[len("test_spans_"):-len(".json")].rsplit("_s", 1)
        r = json.load(open(f"{OUT}/result_{name}_s{seed}.json"))
        e = r["efficiency"]["free"]
        rows.append(dict(lang=lang, system=name[len(lang) + 1:], seed=int(seed), ERR=err_of(sp := json.load(open(f, encoding="utf-8")), data), test_sentences=len(sp),
                         CER_guard=r["norm"]["free+guard"]["CER"], end_to_end_words_per_s=e.get("total_words_per_sec"),
                         encoder_bytes=e.get("encoder_bytes")))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/nb9_err.csv", index=False)
if len(R):
    display(R.round(2))
    display(R.groupby(["lang", "system"])[["ERR", "CER_guard", "end_to_end_words_per_s"]].agg(["mean", "std"]).round(2))
