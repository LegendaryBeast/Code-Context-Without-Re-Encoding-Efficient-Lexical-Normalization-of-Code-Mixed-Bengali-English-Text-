# %% [markdown]
# # CTT error analysis (reviewer point: "what errors does the proposed model still make?")
# The paper's error taxonomy (Table 8) describes the CPU lexicon baseline. This notebook does the same for the
# **proposed CTT** (`ctx_plain_aks_heads`, guarded outputs, 3 splits), in two parts:
#
# 1. **Automatic taxonomy over all CTT token errors** (no annotation needed): rule-based categories, like
#    `experiments/06_error_taxonomy.py`.
# 2. **Human annotation of a 200-error sample** by two annotators, with categories that also separate *model* errors
#    from *reference* problems (wrong silver target, or two equally valid spellings). The silver references were
#    written by one annotator, so this tells reviewers how much of CTT's remaining error is real.
#
# **Workflow**
# 1. Run all cells once (from the `analysis/` folder). This writes, in `analysis/error_annotation/`:
#    `ctt_errors_annotator1.xlsx`, `ctt_errors_annotator2.xlsx` (identical rows, give one to each annotator; they
#    must not see each other's file) and `ctt_errors_key.csv` (automatic labels, *not* for annotators).
# 2. Annotators fill the `label` column (drop-down) and, if useful, `notes`. Each takes ~1.5 h for 200 items.
# 3. Put the two filled files back (same names) and rerun the notebook. Part 2 then prints agreement (κ), the
#    items to adjudicate, the final category shares with 95% CIs, and a LaTeX table for the paper.
# 4. Optional: after adjudication, fill `final_label` in `ctt_errors_adjudicated.xlsx` and rerun for the final table.
#
# Existing annotation files are never overwritten.

# %%
import json, os, re, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "analysis" else Path.cwd()          # the repository root
assert (ROOT / "data" / "BanglaMix.csv").exists(), f"run from the repo root or analysis/ (now {Path.cwd()})"
OUTDIR = ROOT / "analysis" / "error_annotation"
OUTDIR.mkdir(parents=True, exist_ok=True)
SYSTEM, SEEDS, N_SAMPLE, SAMPLE_SEED = "ctx_plain_aks_heads", [13, 42, 2026], 200, 0
DIRS = [ROOT / "results" / "gpu" / "nb5_ctt" / d for d in ("stage1_seed42", "stage2_three_splits", "stage3a_8_epochs", "stage3b_byt5_base")]
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
RUN = re.compile(r"(.{1,4}?)\1+", re.S)
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def bn(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


def max_run(s):
    return max([len(m.group(0)) // len(m.group(1)) for m in RUN.finditer(s)] or [1])


def guard(span, src):   # identical to rep_guard in ctt.py
    allowed = max(3, max_run(bn(src).lower()) + 2)
    return RUN.sub(lambda m: m.group(1) * min(allowed, len(m.group(0)) // len(m.group(1)))
                   + m.group(0)[len(m.group(0)) // len(m.group(1)) * len(m.group(1)):], span)


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


def find(name):
    for d in DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def script(w):
    has_bn = any("ঀ" <= c <= "৿" for c in w)
    has_lat = any(c.isascii() and c.isalpha() for c in w)
    return "mixed" if has_bn and has_lat else "bengali" if has_bn else "latin" if has_lat else "none"


df = pd.read_csv(ROOT / "data" / "BanglaMix.csv")
df["tokens"] = df.token_labels.map(lambda tl: [t.rsplit("/", 1)[0] for t in tl.split(" ")])
df["labels"] = df.token_labels.map(lambda tl: [t.rsplit("/", 1)[1] for t in tl.split(" ")])
df["targets"] = df.bengali_transliteration.map(lambda s: bn(s).split())
df["aligned"] = [len(a) == len(b) for a, b in zip(df.tokens, df.targets)]
print(len(df), "comments")

# %% [markdown]
# ## Part 1: all CTT token errors and an automatic taxonomy
# Token errors are counted on token-aligned test comments (as in the paper's token accuracy). Each error goes to the
# first matching category:
# 1. **BAN: respelling missed**: Bengali-script token copied unchanged, but the reference respells it
# 2. **BAN: wrong change**: Bengali-script token the reference keeps, but CTT changed it
# 3. **BAN: different respelling**: both changed it, differently
# 4. **Digit / symbol (OTH)**
# 5. **Glued / UNK token**: emoji or punctuation glued to a word
# 6. **Latin word unseen in training**
# 7. **Latin word seen in training**: includes context homographs (`ar` → আর / এর) and reference spelling variants
#
# Errors where the output is correct except for an invisible control byte (the decoder sometimes emits `\x02` before
# a word) are counted separately as **Invisible control byte only** and left out of the human sample.

# %%
rows = []
for s in SEEDS:
    split = pd.read_csv(ROOT / "data" / "splits" / f"split_seed{s}.csv")
    d = df.merge(split, on="id")
    vocab = {bn(w).lower() for ws in d[d.split == "train"].tokens for w in ws}
    spans = json.load(open(find(f"test_spans_{SYSTEM}_s{s}.json"), encoding="utf-8"))
    for r in d[(d.split == "test") & d.aligned].itertuples():
        out = [guard(x, w) for x, w in zip(spans[r.id], r.tokens)]
        for k, (w, l, o, y) in enumerate(zip(r.tokens, r.labels, out, r.targets)):
            rows.append(dict(split=s, comment_id=r.id, token_idx=k, token=w, tag=l, output=o, reference=y,
                             correct=o == y, oov=bn(w).lower() not in vocab, script=script(w),
                             context=" ".join(f"⟦{t}⟧" if j == k else t for j, t in enumerate(r.tokens))))
T = pd.DataFrame(rows)
print(f"{len(T)} aligned test tokens, {(~T.correct).sum()} errors, token error rate {100 * (~T.correct).mean():.2f}%")


CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def auto_category(r):
    if CTRL.search(r.output) and CTRL.sub("", r.output) == r.reference:
        return "Invisible control byte only"           # decoder emitted e.g. \x02 before a correct word
    src = bn(r.token)
    if r.tag == "BAN":
        if r.output == src:
            return "BAN: respelling missed"
        if r.reference == src:
            return "BAN: wrong change"
        return "BAN: different respelling"
    if r.tag == "OTHER":
        return "Digit / symbol (OTH)"
    if r.tag == "UNK":
        return "Glued / UNK token"
    if r.script != "latin":
        return "Other"
    return "Latin word, unseen in training" if r.oov else "Latin word, seen in training"


Err = T[~T.correct].copy()
Err["auto_category"] = Err.apply(auto_category, axis=1)
Err["char_edits"] = [lev(o, y) for o, y in zip(Err.output, Err.reference)]
Err["near_miss"] = Err.char_edits <= 1
tax = Err.groupby("auto_category").agg(count=("token", "size"), near_miss_share=("near_miss", "mean"))
tax["share_%"] = 100 * tax["count"] / len(Err)
tax["near_miss_share"] *= 100
tax = tax.sort_values("count", ascending=False)
display(tax.round(1))
print("error rate by tag (%):")
display((100 * (1 - T.groupby("tag").correct.mean())).round(2).to_frame("token_error_rate"))
for c in tax.index:
    ex = Err[Err.auto_category == c][["token", "tag", "output", "reference"]].value_counts().head(4)
    print(f"\n{c}:"); print(ex.to_string())

# %% [markdown]
# ## Part 1b: sample 200 errors for human annotation
# Stratified by split (proportional), fixed random seed. Annotators see the comment with the token marked ⟦like this⟧,
# the source token, its silver tag, CTT's output and the reference. They do **not** see the automatic category.

# %%
LABELS = {
    "M1 transliteration": "Model error: a Latin-script word rendered with a wrong Bengali spelling (e.g. kortaci -> করটাছি)",
    "M2 wrong word in context": "Model error: CTT outputs a real word, but the wrong one for this context (homographs such as ar -> আর vs এর)",
    "M3 Bengali spelling": "Model error: a Bengali-script word that should be respelled was not, or a correct one was changed",
    "M4 digit / punctuation / emoji": "Model error in numbers, punctuation or emoji (e.g. 5 -> ৫ missing, '.' -> '।')",
    "M5 glued token / segmentation": "Model error caused by a word glued to emoji/punctuation or to another word",
    "R1 reference wrong": "The silver reference is wrong and CTT's output is correct",
    "R2 both acceptable": "CTT's output and the reference are both acceptable spellings (spelling variation)",
    "O other": "None of the above (explain in notes)",
}
KEY_COLS = ["item_id", "split", "comment_id", "token_idx", "tag", "auto_category", "oov", "char_edits", "near_miss"]
SHEET_COLS = ["item_id", "context", "token", "tag", "ctt_output", "reference", "label", "notes"]
sheet_paths = [OUTDIR / f"ctt_errors_annotator{i}.xlsx" for i in (1, 2)]
key_path = OUTDIR / "ctt_errors_key.csv"

if key_path.exists():
    print("sample already exists; not overwriting", key_path)
    Key = pd.read_csv(key_path)
else:
    rng = np.random.RandomState(SAMPLE_SEED)
    pool = Err[Err.auto_category != "Invisible control byte only"]
    per_split = (pool.split.value_counts(normalize=True) * N_SAMPLE).round().astype(int)
    per_split.iloc[0] += N_SAMPLE - per_split.sum()
    sample = pd.concat([pool[pool.split == s].sample(n, random_state=rng) for s, n in per_split.items()])
    sample = sample.sample(frac=1, random_state=rng).reset_index(drop=True)      # shuffle so splits are mixed
    sample.insert(0, "item_id", [f"E{i + 1:03d}" for i in range(len(sample))])
    Key = sample.rename(columns={"output": "ctt_output"})

    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    clean = lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    for path in sheet_paths:
        wb = Workbook()
        ws = wb.active; ws.title = "annotate"
        ws.append(SHEET_COLS)
        for r in Key.itertuples():
            ws.append([clean(v) for v in (r.item_id, r.context, r.token, r.tag, r.ctt_output, r.reference, "", "")])
        dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
        ws.add_data_validation(dv); dv.add(f"G2:G{len(Key) + 1}")
        for col, w in zip("ABCDEFGH", (8, 70, 18, 9, 18, 18, 30, 30)):
            ws.column_dimensions[col].width = w
        for c in ws[1]:
            c.font = Font(bold=True); c.fill = PatternFill("solid", fgColor="DDEBF7")
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        info = wb.create_sheet("instructions")
        info.append(["How to annotate"])
        info.append(["Each row is one word where the CTT output differs from the silver reference. The word is marked ⟦like this⟧ in the comment."])
        info.append(["Pick exactly one label in column G. Use R1/R2 only when you are sure the CTT output is acceptable Bengali for this comment."])
        info.append(["Work alone; do not discuss items with the other annotator until both files are finished."])
        info.append([])
        info.append(["label", "meaning"])
        for k, v in LABELS.items():
            info.append([k, v])
        info.column_dimensions["A"].width = 32; info.column_dimensions["B"].width = 110
        info["A1"].font = Font(bold=True, size=13); info["A6"].font = Font(bold=True); info["B6"].font = Font(bold=True)
        wb.save(path)
    Key[KEY_COLS + ["token", "ctt_output", "reference", "context"]].to_csv(key_path, index=False)   # last: marks the sample as done
    print("wrote", *sheet_paths, key_path, sep="\n  ")
print(Key.auto_category.value_counts().to_string())

# %% [markdown]
# ## Part 2: after annotation: agreement, adjudication, final shares
# Runs only when both annotator files have labels.

# %%
from sklearn.metrics import cohen_kappa_score


def read_labels(path):
    a = pd.read_excel(path, sheet_name="annotate")
    return a.set_index("item_id")["label"].astype("string").str.strip()


ann = [read_labels(p) for p in sheet_paths]
done = [int(a.notna().sum()) for a in ann]
print("labelled items:", done)
if min(done) < len(Key):
    print("Part 2 waits until both annotators have labelled all", len(Key), "items.")
else:
    A = pd.DataFrame({"a1": ann[0], "a2": ann[1]}).join(Key.set_index("item_id"))
    A["a1_major"], A["a2_major"] = A.a1.str[0], A.a2.str[0]                      # M (model) / R (reference) / O
    print(f"Cohen's kappa, fine labels: {cohen_kappa_score(A.a1, A.a2):.3f} | model vs. reference: "
          f"{cohen_kappa_score(A.a1_major, A.a2_major):.3f} | raw agreement {100 * (A.a1 == A.a2).mean():.1f}%")
    display(pd.crosstab(A.a1, A.a2))

    adj_path = OUTDIR / "ctt_errors_adjudicated.xlsx"
    if not adj_path.exists():
        out = A.reset_index()[["item_id", "context", "token", "tag", "ctt_output", "reference", "a1", "a2"]]
        out["final_label"] = np.where(out.a1 == out.a2, out.a1, "")
        out.to_excel(adj_path, index=False)
        print(f"{int((A.a1 != A.a2).sum())} disagreements -> fill final_label in", adj_path, "then rerun")
    final = pd.read_excel(adj_path).set_index("item_id")["final_label"]
    if final.isna().any() or (final.astype(str).str.strip() == "").any():
        print("adjudication not finished; shares below use items where both annotators agree")
        final = A.a1.where(A.a1 == A.a2).dropna()
    rng = np.random.RandomState(0)
    labels = final.values
    share = pd.Series(labels).value_counts(normalize=True) * 100
    boot = np.array([pd.Series(labels[rng.randint(0, len(labels), len(labels))]).value_counts(normalize=True)
                     .reindex(share.index, fill_value=0).values * 100 for _ in range(2000)])
    res = pd.DataFrame({"share_%": share, "ci_low": np.percentile(boot, 2.5, axis=0), "ci_high": np.percentile(boot, 97.5, axis=0)})
    display(res.round(1))
    ref_share = share[[k for k in share.index if str(k).startswith("R")]].sum()
    print(f"{ref_share:.1f}% of sampled CTT token errors are reference problems (R1 + R2), not model errors.")
    print("\nLaTeX rows:")
    for k, r in res.iterrows():
        print(f"{k} & {r['share_%']:.1f} & [{r.ci_low:.1f}, {r.ci_high:.1f}] \\\\")
