# %% [markdown]
# # BanglaMix — Open LLM few-shot baseline (Kaggle, GPU)
# Reviewers in 2026 will ask "why not just prompt an LLM?". This notebook answers it on a fixed random
# subset of the seed-42 test split, for both tasks (token LID and Bengali-script normalisation),
# and scores the local CRF on the **same subset** for a fair comparison.
#
# **Setup:** GPU T4 x2, Internet ON, attach the Kaggle dataset built from `kaggle/kaggle_dataset.zip`. 4-bit loading via bitsandbytes.
# ~60–90 min for 300 comments with a 7–8B model.

# %%
!pip -q install -U bitsandbytes accelerate rapidfuzz 2>/dev/null

# %%
import glob, json, os, re, unicodedata
import numpy as np, pandas as pd, torch
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found"
    return hits[0]

MODEL = "Qwen/Qwen2.5-7B-Instruct"   # or e.g. "meta-llama/Llama-3.1-8B-Instruct" (gated), "google/gemma-2-9b-it"
N_TEST, K_SHOT, SEED, BS = 300, 8, 42, 8
LABELS = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]
OUT = "/kaggle/working"

NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
def bn_norm(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")

df = pd.read_csv(find("BanglaMix.csv"))
def parse(tl):
    p = [t.rsplit("/", 1) for t in tl.split(" ")]
    return [x[0] for x in p], [x[1] for x in p]
df["tokens"], df["labels"] = zip(*df.token_labels.map(parse))
df = df.merge(pd.read_csv(find(f"split_seed{SEED}.csv")), on="id")
crf = json.load(open(find(f"crf_pred_seed{SEED}.json")))

tr = df[df.split == "train"]
te = df[(df.split == "test") & (df.tokens.map(len) <= 40)].sample(N_TEST, random_state=0)
# k-shot demos: one or two per code-mix type, short comments only
shots = (tr[tr.tokens.map(len).between(4, 14)].groupby("code_mix_type", group_keys=False)
         .apply(lambda g: g.sample(1, random_state=0)).head(K_SHOT))

# %%
GUIDE = ("Tag every whitespace-separated token of a YouTube comment with one label:\n"
         "BAN = Bengali in Bengali script; ENG = English word in Latin script; "
         "ROM_BAN = Bengali word written in Latin letters (Romanized Bangla); "
         "OTHER = punctuation, numbers, emoji, symbols; UNK = cannot be decided.\n"
         "Return ONLY a JSON list of labels, exactly one label per token, in order.")
TRANS = ("Rewrite the whole YouTube comment in Bengali script. Romanized Bangla words become standard Bengali "
         "spelling, English words are written phonetically in Bengali script (e.g. please -> প্লিজ), "
         "Bengali-script words stay as they are, numbers become Bengali digits. "
         "Keep one output word per input word. Return ONLY the rewritten comment.")

def lid_msgs(tokens):
    m = [{"role": "system", "content": GUIDE}]
    for r in shots.itertuples():
        m += [{"role": "user", "content": json.dumps(r.tokens, ensure_ascii=False)},
              {"role": "assistant", "content": json.dumps(r.labels)}]
    return m + [{"role": "user", "content": json.dumps(tokens, ensure_ascii=False)}]

def tr_msgs(text):
    m = [{"role": "system", "content": TRANS}]
    for r in shots.itertuples():
        m += [{"role": "user", "content": r.cleaned_text}, {"role": "assistant", "content": r.bengali_transliteration}]
    return m + [{"role": "user", "content": text}]

# %%
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"))

@torch.no_grad()
def chat(batch_msgs, max_new):
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
    x = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    g = model.generate(**x, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(g[:, x.input_ids.shape[1]:], skip_special_tokens=True)

# %%
rows = list(te.itertuples())
lid_out, tr_out = [], []
for i in range(0, len(rows), BS):
    b = rows[i:i + BS]
    lid_out += chat([lid_msgs(r.tokens) for r in b], max_new=12 * 40)
    tr_out += chat([tr_msgs(r.cleaned_text) for r in b], max_new=400)
    print(i + len(b), "/", len(rows))
pd.DataFrame({"id": te.id, "lid_raw": lid_out, "translit_raw": tr_out}).to_csv(f"{OUT}/llm_raw_outputs.csv", index=False)

# %%
def parse_tags(s, n):
    try:
        tags = json.loads(re.search(r"\[.*\]", s, re.S).group(0))
        tags = [t if t in LABELS else "UNK" for t in tags]
    except Exception:
        tags = []
    ok = len(tags) == n
    return (tags + ["OTHER"] * n)[:n], ok

gold, llm, crfp, valid = [], [], [], []
for r, s in zip(rows, lid_out):
    t, ok = parse_tags(s, len(r.tokens))
    gold += r.labels; llm += t; crfp += crf[r.id]; valid.append(ok)
def sc(p):
    return {"MacroF1": 100 * f1_score(gold, p, labels=LABELS, average="macro", zero_division=0),
            "Acc": 100 * accuracy_score(gold, p)}
lid_res = pd.DataFrame([dict(system=f"{MODEL} {K_SHOT}-shot", **sc(llm), well_formed=100 * np.mean(valid)),
                        dict(system="CRF (same subset)", **sc(crfp), well_formed=100.0)])

refs = [bn_norm(r.bengali_transliteration) for r in rows]
hyps = [bn_norm(h.strip().split("\n")[0]) for h in tr_out]
cer = 100 * sum(Levenshtein.distance(h, r) for h, r in zip(hyps, refs)) / sum(map(len, refs))
wer = 100 * sum(Levenshtein.distance(h.split(), r.split()) for h, r in zip(hyps, refs)) / sum(len(r.split()) for r in refs)
tr_res = pd.DataFrame([dict(system=f"{MODEL} {K_SHOT}-shot", CER=cer, WER=wer,
                            Exact=100 * np.mean([h == r for h, r in zip(hyps, refs)]))])
lid_res.to_csv(f"{OUT}/llm_lid_results.csv", index=False); tr_res.to_csv(f"{OUT}/llm_translit_results.csv", index=False)
te[["id"]].to_csv(f"{OUT}/llm_subset_ids.csv", index=False)   # so local lexicon/ByT5 systems can be scored on the same subset
display(lid_res.round(2)); display(tr_res.round(2))
