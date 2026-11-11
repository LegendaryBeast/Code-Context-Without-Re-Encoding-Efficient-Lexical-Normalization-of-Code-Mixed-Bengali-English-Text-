# %% [markdown]
# # Training-seed replicates on split 42 (two more runs per system)
# **Why this notebook.** Reviewers note that every BanglaMix comparison rests on one training run per split, while
# retraining with the same settings moved CER by up to 0.30. This notebook adds **two new training seeds (1 and 2)
# on the same split 42**, so each system has three runs (the existing seed-42 run plus these two) and the
# comparisons can pool training seeds, as the MultiLexNorm results already do.
#
# **What runs** (same `ctt.py`, settings and metrics as the paper; `--seed 42` fixes the split, `--train_seed` changes
# initialisation, batch order and the warm-start sample):
#
# | variant | system | existing seed-42 run |
# |---|---|---|
# | `ctt_ts{1,2}` | CTT (`ctx_plain_aks_heads`: warm start, detached LID and copy heads) | nb5 stage 2 |
# | `ctx_plain_ts{1,2}` | context only (CTT without the warm start) | nb5 stage 1 |
# | `word_plain_align_ts{1,2}` | **fair word-level baseline**: aligned training data, no LID tags | nb8 |
#
# The word-level baseline answers the review point that CTT's gain over `word_crf` mixes context with alignment
# recovery and noisy tags; `word_plain_align` has the same aligned data and no tags, so it isolates context.
#
# ### How to run (Kaggle)
# 1. Attach the same Kaggle dataset as nb5 and nb8 (`kaggle/kaggle_dataset.zip`).
# 2. Settings: **GPU T4 x2**, **Internet ON** (ByT5-small and Aksharantar from Hugging Face).
# 3. *Save Version → Save & Run All*. Expect about 6 h: each GPU runs one training seed (CTT ~2.3 h, context only
#    ~2 h, word level ~1.4 h). The job loop skips finished jobs, so a re-run resumes.
# 4. Download the whole `/kaggle/working/ctt` folder and put its files into `results/gpu/nb11_train_seeds/`.
#    Then run locally: `python analysis/analyze_seeds.py`.

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
EPOCHS, SPLIT = 5, 42
MODEL = "google/byt5-small"
BASE_ARGS = ["--batch_words", "48", "--accum", "2", "--max_chunks", "8", "--gen_bs", "48"]
POLL_SECONDS = 120
SYSTEMS = {   # identical to the settings of the existing seed-42 runs
    "ctt":              dict(mode="focus", align=1, pretrain_aksharantar=200000, lam=0.4, copy_mu=0.5, detach_heads=1),
    "ctx_plain":        dict(mode="focus", align=1),
    "word_plain_align": dict(mode="word", align=1),
}
TRAIN_SEEDS = [1, 2]
# one training seed per GPU; longest job first
GPU_QUEUES = [[(f"{s}_ts{t}", t, cfg) for s, cfg in SYSTEMS.items()] for t in TRAIN_SEEDS]

os.makedirs(OUT, exist_ok=True)
done = lambda v: os.path.exists(f"{OUT}/result_{v}_s{SPLIT}.json")      # resume-safe
n_gpu = max(1, torch.cuda.device_count())
if n_gpu < len(GPU_QUEUES):
    GPU_QUEUES = [[j for q in GPU_QUEUES for j in q]]
GPU_QUEUES = [[j for j in q if not done(j[0])] for q in GPU_QUEUES]
print(n_gpu, "GPU(s); queues:", [[j[0] for j in q] for q in GPU_QUEUES])

# %% [markdown]
# ## Run jobs

# %%
def launch(gpu, job):
    v, ts, cfg = job
    logf = open(f"{OUT}/log_{v}_s{SPLIT}.txt", "w")
    cmd = [sys.executable, SCRIPT, "--variant", v, "--seed", str(SPLIT), "--train_seed", str(ts), "--epochs", str(EPOCHS),
           "--model", MODEL, "--out", OUT]
    for k, x in cfg.items():
        cmd += [f"--{k}", str(x)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return subprocess.Popen(cmd + BASE_ARGS, stdout=logf, stderr=subprocess.STDOUT, env=env), v, logf

def tail(v):
    try:
        lines = [l for l in open(f"{OUT}/log_{v}_s{SPLIT}.txt") if l.startswith("[")]
        return lines[-1].strip() if lines else ""
    except FileNotFoundError:
        return ""

t0, running = time.time(), {}
while any(GPU_QUEUES) or running:
    for g, q in enumerate(GPU_QUEUES):
        if g not in running and q:
            running[g] = launch(g, q.pop(0))
    time.sleep(POLL_SECONDS)
    for g, (proc, v, logf) in list(running.items()):
        if proc.poll() is not None:
            logf.close()
            print(f"[{(time.time() - t0) / 60:6.1f} min] finished {v} exit={proc.returncode}")
            if proc.returncode != 0:
                text = open(logf.name).read()
                print("\n".join(l for l in text.splitlines() if l.startswith("[")))
                print(text[-1500:])
            del running[g]
        else:
            print(f"[{(time.time() - t0) / 60:6.1f} min] gpu{g} {tail(v)}")

# %% [markdown]
# ## Results in this session
# Guarded test CER on split 42. Existing seed-42 runs: CTT 3.64, context only 3.58, word level (aligned, no tags) 4.08.

# %%
rows = []
for f in sorted(glob.glob(f"{OUT}/result_*_s{SPLIT}.json")):
    r = json.load(open(f))
    rows.append(dict(variant=r["args"]["variant"], train_seed=r["args"]["train_seed"],
                     CER_raw=r["norm"]["free"]["CER"], CER_guard=r["norm"]["free+guard"]["CER"],
                     EM=r["norm"]["free"]["Exact"]))
display(pd.DataFrame(rows).round(2))
