# %% [markdown]
# # BanglaMix — Reviewer check: do LID tags help normalization? (oracle tags, word × context design)
# **Why this notebook.** The paper finds that LID tags do not help normalization once context is available
# (`ctx_crf → ctx_plain`: −0.18 CER on split 42). The review raised two objections:
# 1. **Train–test mismatch.** Tagged models train with silver (gold) tags but are tested with CRF-predicted tags,
#    so the −0.18 may come from the mismatch rather than from the tags themselves.
# 2. **No factorial design.** There is no word-level model without tags, so the claimed interaction with context
#    is untested.
#
# **What runs** (split 42, same `ctt.py` code base, settings and metrics as Stage 1 of nb5):
#
# | variant | mode | tags in training | tags at test | gives cell |
# |---|---|---|---|---|
# | `ctx_crf_rerun` | context (focus) | silver | CRF **and** gold, same weights | context × predicted, context × oracle |
# | `word_crf_align_rerun` | word level | silver | CRF **and** gold, same weights | word × predicted, word × oracle |
# | `word_plain_align` | word level | none | none | word × none |
#
# The other cells already exist: context × none (`ctx_plain`, nb5 Stage 1, and `ctx_plain_rerun`, nb6) and the
# original single runs `ctx_crf` and `word_crf_align` (nb5 Stage 1). The two reruns also give one more
# run-to-run estimate each.
#
# `--eval_ext_lid gold` decodes the test split a second time with gold tags using the **same trained weights**,
# so "CRF tags vs gold tags" is free of training noise. The gold-tag outputs are saved as variant
# `<variant>_goldtags` (for example `test_spans_ctx_crf_rerun_goldtags_s42.json`).
#
# ### How to run (Kaggle)
# 1. Attach the same Kaggle dataset as nb5 and nb6 (`kaggle/kaggle_dataset.zip`: BanglaMix.csv, splits, CRF predictions).
# 2. Settings: **GPU T4 x2**, **Internet ON** (downloads ByT5-small).
# 3. *Save Version → Save & Run All*. Expect about 3–3.5 h: GPU 0 runs `ctx_crf_rerun` (~2 h), GPU 1 runs the two
#    word-level jobs (~1.4 h each). The job loop skips finished jobs, so a re-run resumes.
# 4. Download the whole `/kaggle/working/ctt` folder and put its files into `results/gpu/nb8_oracle_tags/`
#    (files directly, no `ctt/` subfolder). Then run locally: `python analysis/analyze_oracle.py`.

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
MODEL = "google/byt5-small"
BASE_ARGS = ["--batch_words", "48", "--accum", "2", "--max_chunks", "8", "--gen_bs", "48"]
POLL_SECONDS = 120

VARIANTS = {   # identical to the nb5 Stage-1 settings of ctx_crf / word_crf_align, plus the gold-tag test pass
    "ctx_crf_rerun":        dict(mode="focus", ext_lid="crf", align=1, eval_ext_lid="gold"),
    "word_crf_align_rerun": dict(mode="word", ext_lid="crf", align=1, eval_ext_lid="gold"),
    "word_plain_align":     dict(mode="word", align=1),
}
# One list per GPU, so the ~2 h context job runs next to the two ~1.4 h word-level jobs.
GPU_QUEUES = [["ctx_crf_rerun"], ["word_crf_align_rerun", "word_plain_align"]]
SEED = 42

os.makedirs(OUT, exist_ok=True)
done = lambda v: os.path.exists(f"{OUT}/result_{v}_s{SEED}.json")      # resume-safe
n_gpu = max(1, torch.cuda.device_count())
if n_gpu < len(GPU_QUEUES):                                             # one GPU: run everything in sequence
    GPU_QUEUES = [[v for q in GPU_QUEUES for v in q]]
GPU_QUEUES = [[v for v in q if not done(v)] for q in GPU_QUEUES]
print(n_gpu, "GPU(s); queues:", GPU_QUEUES)

# %% [markdown]
# ## Run jobs

# %%
def launch(gpu, v):
    logf = open(f"{OUT}/log_{v}_s{SEED}.txt", "w")
    cmd = [sys.executable, SCRIPT, "--variant", v, "--seed", str(SEED), "--epochs", str(EPOCHS), "--model", MODEL,
           "--out", OUT]
    for k, x in VARIANTS[v].items():
        cmd += [f"--{k}", str(x)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return subprocess.Popen(cmd + BASE_ARGS, stdout=logf, stderr=subprocess.STDOUT, env=env), v, logf

def tail(v):
    try:
        lines = [l for l in open(f"{OUT}/log_{v}_s{SEED}.txt") if l.startswith("[")]
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
# Guarded CER (`free+guard`) is what the paper reports. Compare with the existing split-42 cells:
# `ctx_plain` 3.58 (rerun 3.88), `ctx_crf` 3.76, `word_crf_align` 3.97.

# %%
rows = []
for f in sorted(glob.glob(f"{OUT}/result_*.json")):
    r = json.load(open(f))
    v = r["args"]["variant"]
    rows.append(dict(run=v, test_tags=r["args"]["ext_lid"], CER=r["norm"]["free"]["CER"],
                     CER_guard=r["norm"]["free+guard"]["CER"], EM=r["norm"]["free+guard"]["Exact"]))
    for src, m in r.get("norm_ext", {}).items():
        rows.append(dict(run=v, test_tags=src, CER=m["free"]["CER"], CER_guard=m["free+guard"]["CER"],
                         EM=m["free+guard"]["Exact"]))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/nb8_cells.csv", index=False)
if len(R):
    display(R.round(2))
