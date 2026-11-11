"""Draws the adjudicated normalization test subset and writes the blind annotation sheets.

Sample (BanglaMix split-42 test set, 1,261 comments):
  * every code-mixed comment (likely_code_mixed = True; 464 comments), and
  * 136 monolingual comments drawn at random, allocated to monolingual_BAN / _ENG / _ROM_BAN in proportion to
    their test-set sizes (411 / 207 / 129 -> 75 / 38 / 23). "undetermined" comments (50) are not sampled.
The 600 comments are shuffled once (seed below); the first 50 in that order form the pilot.

Each sheet has one row per source token, prefilled with the silver target. For comments whose reference is not
token-aligned in the corpus, the silver targets come from the aligner, exactly as in training (lexicon from the
aligned split-42 training comments). No system output is shown anywhere.

Writes (in this folder):
  sample_key.csv          one row per comment (order, id, mix type, alignment, pilot flag)
  tokens_key.csv          one row per token with the silver target (the reference for scoring; never edited)
  sheets/pilot_annotator_{A,B}.xlsx   the first 50 comments
  sheets/full_annotator_{A,B}.xlsx    all 600 comments (the pilot comments come first)

Run from the repo root:  python annotation/adjudicated_test/make_sample.py
"""
import os, sys
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "src"))
from banglish_align import GLUE, Aligner  # noqa: E402

SEED, SPLIT, N_PILOT = 2026, 42, 50
MONO = {"monolingual_BAN": 75, "monolingual_ENG": 38, "monolingual_ROM_BAN": 23}
GLUE_MARK = "~"        # shown to annotators: "attach to the previous word without a space"
FLAGS = ["source unclear", "token misaligned", "offensive or personal data", "other (see notes)"]

import unicodedata
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}


def bn(s):                       # identical to analysis/review_analyses.py
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


df = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv"), keep_default_na=False).set_index("id")
df["tokens"] = df.token_labels.map(lambda tl: [t.rsplit("/", 1)[0] for t in tl.split(" ")])
df["labels"] = df.token_labels.map(lambda tl: [t.rsplit("/", 1)[1] for t in tl.split(" ")])
df["targets"] = df.bengali_transliteration.map(lambda s: bn(s).split())
df["aligned"] = [len(a) == len(b) for a, b in zip(df.tokens, df.targets)]
df["cm"] = df.likely_code_mixed.astype(str).str.lower().eq("true")
sp = pd.read_csv(os.path.join(ROOT, "data", "splits", f"split_seed{SPLIT}.csv"))
train, test = sp[sp.split == "train"].id.tolist(), sp[sp.split == "test"].id.tolist()

rng = np.random.RandomState(SEED)
T = df.loc[test]
chosen = sorted(T.index[T.cm])
for t, n in MONO.items():
    pool = sorted(T.index[(~T.cm) & (T.code_mix_type == t)])
    chosen += list(rng.choice(pool, n, replace=False))
order = list(rng.permutation(chosen))
assert len(order) == len(set(order)) == 464 + sum(MONO.values())

tr = df.loc[train]
aligner = Aligner({(w.lower(), y) for r in tr[tr.aligned].itertuples() for w, y in zip(r.tokens, r.targets)})

comments, tokens = [], []
for k, c in enumerate(order, 1):
    r = df.loc[c]
    if r.aligned:
        spans = list(r.targets)
    else:
        spans = aligner.align(list(r.tokens), list(r.targets))[0]
        assert spans is not None, c
    comments.append(dict(order=k, comment_no=f"C{k:03d}", comment_id=c, code_mix_type=r.code_mix_type,
                         code_mixed=bool(r.cm), aligned_in_corpus=bool(r.aligned), n_tokens=len(r.tokens),
                         pilot=k <= N_PILOT, silver_reference=" ".join(r.targets)))
    for i, (w, l, y) in enumerate(zip(r.tokens, r.labels, spans), 1):
        ctx = " ".join(f"⟦{x}⟧" if j == i else x for j, x in enumerate(r.tokens, 1))
        tokens.append(dict(item_id=f"C{k:03d}-T{i:02d}", comment_no=f"C{k:03d}", comment_id=c, token_no=i,
                           source_token=w, lid_tag=l, context=ctx, silver_target=y.replace(GLUE, GLUE_MARK),
                           silver_target_raw=y))
C, K = pd.DataFrame(comments), pd.DataFrame(tokens)
C.to_csv(os.path.join(HERE, "sample_key.csv"), index=False, encoding="utf-8")
K.to_csv(os.path.join(HERE, "tokens_key.csv"), index=False, encoding="utf-8")

COLS = [("item_id", 11), ("comment_no", 10), ("token_no", 7), ("source_token", 18), ("lid_tag", 9),
        ("context", 60), ("silver_comment_reference", 45), ("silver_target", 18), ("corrected_target", 20), ("acceptable_variants", 24),
        ("flag", 20), ("notes", 30)]
EDIT = {"corrected_target", "acceptable_variants", "flag", "notes"}
GREY, BLUE, YEL = PatternFill("solid", fgColor="EDEDED"), PatternFill("solid", fgColor="E3F0FA"), PatternFill("solid", fgColor="FFF7D6")
INSTR = [
    "Adjudicated BanglaMix test subset: annotation sheet",
    "",
    "Read annotation/adjudicated_test/GUIDELINES.md before starting. Work alone; do not discuss items with the other annotator.",
    "One row = one source token. 'context' shows the whole comment with the token marked ⟦like this⟧.",
    "'silver_comment_reference' (first row of each comment) is the whole corpus reference; 'silver_target' is its part for this token. Do not edit either.",
    "corrected_target: prefilled with the silver target. Keep it if it is correct; otherwise overwrite it with the correct Bengali-script form.",
    "    Leave it empty only if the token should produce no output (its text is already contained in a neighbouring target).",
    "    A leading ~ means 'attach to the previous word without a space' (for example ~। for a full stop).",
    "acceptable_variants: other spellings that are equally correct, separated by |  (for example  প্লিজ|প্লীজ). Leave empty if none.",
    "flag: choose from the list only when needed (source unclear, token misaligned, offensive or personal data, other).",
    "notes: optional short explanation, required when flag = other.",
    "Do not change the order of rows, the item_id column, or the sheet name. Save as .xlsx.",
]


def write_sheet(path, ks, annotator):
    wb = Workbook()
    ws0 = wb.active; ws0.title = "instructions"
    for i, line in enumerate(INSTR, 1):
        ws0.cell(i, 1, line).font = Font(bold=(i == 1), size=12 if i == 1 else 11)
    ws0.cell(len(INSTR) + 2, 1, f"Annotator: {annotator}").font = Font(bold=True)
    ws0.column_dimensions["A"].width = 140
    ws = wb.create_sheet("tokens")
    for j, (name, w) in enumerate(COLS, 1):
        cell = ws.cell(1, j, name); cell.font = Font(bold=True)
        cell.fill = YEL if name in EDIT else GREY
        ws.column_dimensions[cell.column_letter].width = w
    dv = DataValidation(type="list", formula1='"' + ",".join(FLAGS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    prev, band = None, False
    for i, r in enumerate(ks.itertuples(), 2):
        if r.comment_no != prev:
            band, prev = not band, r.comment_no
        vals = dict(item_id=r.item_id, comment_no=r.comment_no, token_no=r.token_no, source_token=r.source_token,
                    lid_tag=r.lid_tag, context=r.context, silver_comment_reference=refs[r.comment_no] if r.token_no == 1 else "",
                    silver_target=r.silver_target,
                    corrected_target=r.silver_target, acceptable_variants="", flag="", notes="")
        for j, (name, _) in enumerate(COLS, 1):
            cell = ws.cell(i, j, vals[name])
            cell.number_format = "@"                       # keep digits and leading ~ as text
            cell.alignment = Alignment(wrap_text=name in ("context", "silver_comment_reference"), vertical="top")
            if band and name not in EDIT:
                cell.fill = BLUE
        dv.add(f"K{i}")
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:L{len(ks) + 1}"
    wb.save(path)


refs = dict(zip(C.comment_no, C.silver_reference))
os.makedirs(os.path.join(HERE, "sheets"), exist_ok=True)
pilot_ids = set(C.comment_no[C.pilot])
for a in ("A", "B"):
    write_sheet(os.path.join(HERE, "sheets", f"pilot_annotator_{a}.xlsx"), K[K.comment_no.isin(pilot_ids)], a)
    write_sheet(os.path.join(HERE, "sheets", f"full_annotator_{a}.xlsx"), K, a)

P = C[C.pilot]
print(f"sample: {len(C)} comments, {len(K)} tokens; code-mixed {C.code_mixed.sum()}, monolingual {(~C.code_mixed).sum()}; "
      f"not token-aligned in the corpus {(~C.aligned_in_corpus).sum()}")
print(C.code_mix_type.value_counts().to_string())
print(f"pilot: {len(P)} comments, {K.comment_no.isin(pilot_ids).sum()} tokens; code-mixed {P.code_mixed.sum()}, "
      f"not token-aligned {(~P.aligned_in_corpus).sum()}")
