"""Inter-annotator agreement and the adjudication sheet.

  python annotation/adjudicated_test/agreement.py --a sheets/full_annotator_A_done.xlsx --b sheets/full_annotator_B_done.xlsx \
         --out annotation/adjudicated_test/results/full

Reports (on the items present in both sheets):
  * Cohen's kappa on whether each token needs a change to its silver target (changed / unchanged);
  * exact agreement on corrected_target; compatible agreement (one annotator's target is among the other's
    accepted forms); agreement on the variant sets;
  * comment-level agreement: identical references, and CER of B's reference against A's;
  * change rates per annotator by LID tag and by code-mixed vs monolingual, and flag counts.
Writes <out>_agreement.txt, <out>_agreed.csv (tokens final without adjudication) and <out>_adjudication.xlsx
(every token on which the annotators differ in target, variants, or flag).
"""
import argparse, os
import numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from rapidfuzz.distance import Levenshtein
from common import EMPTY, join_spans, read_sheet, sample_key, show, tokens_key

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--out", required=True, help="output prefix, e.g. annotation/adjudicated_test/results/pilot")
args = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

K = tokens_key().set_index("item_id"); C = sample_key().set_index("comment_no")
A, B = read_sheet(args.a), read_sheet(args.b)
items = [i for i in K.index if i in A.index and i in B.index]
if not items:
    raise SystemExit("the two sheets share no item_id")
for name, S in (("A", A), ("B", B)):
    extra = set(S.index) - set(K.index)
    if extra:
        raise SystemExit(f"sheet {name} has unknown item_id values, e.g. {sorted(extra)[:3]}")
K = K.loc[items]; A = A.loc[items]; B = B.loc[items]
silver = K.silver_target_raw
lines = []
say = lambda s="": (print(s), lines.append(s))


def kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    po = np.mean(x == y)
    pe = np.mean(x) * np.mean(y) + (1 - np.mean(x)) * (1 - np.mean(y))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


chA, chB = (A.target != silver).values, (B.target != silver).values
setA = [set([t] + v) for t, v in zip(A.target, A.variants)]
setB = [set([t] + v) for t, v in zip(B.target, B.variants)]
exact = (A.target == B.target).values
compat = np.array([a in sb or b in sa for a, b, sa, sb in zip(A.target, B.target, setA, setB)])
same_set = np.array([sa == sb for sa, sb in zip(setA, setB)])
ncom = K.comment_no.nunique()
say(f"items: {len(items)} tokens in {ncom} comments")
say(f"tokens changed from silver: A {100 * chA.mean():.1f}%, B {100 * chB.mean():.1f}%; both {100 * (chA & chB).mean():.1f}%")
say(f"Cohen's kappa (changed vs unchanged): {kappa(chA, chB):.3f}")
say(f"exact agreement on corrected_target: {100 * exact.mean():.2f}%   compatible agreement: {100 * compat.mean():.2f}%   "
    f"identical accepted-form sets: {100 * same_set.mean():.2f}%")
if (chA | chB).any():
    say(f"   among tokens changed by at least one annotator ({(chA | chB).sum()}): exact {100 * exact[chA | chB].mean():.1f}%, "
        f"compatible {100 * compat[chA | chB].mean():.1f}%")

refA = {c: join_spans(list(g)) for c, g in A.target.groupby(K.comment_no, sort=False)}
refB = {c: join_spans(list(g)) for c, g in B.target.groupby(K.comment_no, sort=False)}
same_c = np.mean([refA[c] == refB[c] for c in refA])
cer_ab = 100 * sum(Levenshtein.distance(refB[c], refA[c]) for c in refA) / max(1, sum(len(refA[c]) for c in refA))
say(f"comments with identical references: {100 * same_c:.1f}%   CER of B against A: {cer_ab:.2f}%")

say("\nchange rate (%) by LID tag and subset")
T = pd.DataFrame({"tag": K.lid_tag.values, "cm": C.loc[K.comment_no, "code_mixed"].astype(str).str.lower().eq("true").values,
                  "A": chA, "B": chB, "exact": exact})
tab = T.groupby("tag")[["A", "B", "exact"]].mean().mul(100).round(1); tab["tokens"] = T.groupby("tag").size()
say(tab.to_string())
tab = T.groupby("cm")[["A", "B", "exact"]].mean().mul(100).round(1); tab["tokens"] = T.groupby("cm").size()
tab.index = ["monolingual" if not x else "code-mixed" for x in tab.index]
say(tab.to_string())
say("\nflags: A " + str(A.flag[A.flag != ""].value_counts().to_dict()) + "; B " + str(B.flag[B.flag != ""].value_counts().to_dict()))
open(args.out + "_agreement.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- agreed tokens and adjudication sheet
need = (~exact) | (~same_set) | (A.flag != "").values | (B.flag != "").values
agreed = pd.DataFrame({"item_id": np.array(items)[~need], "final_target": [show(t) for t in A.target[~need]],
                       "final_variants": ["|".join(v) for v in A.variants[~need]]})
agreed.to_csv(args.out + "_agreed.csv", index=False, encoding="utf-8")

cols = [("item_id", 11), ("comment_no", 10), ("source_token", 16), ("lid_tag", 9), ("context", 55), ("silver_target", 16),
        ("A_target", 16), ("A_variants", 18), ("A_flag", 14), ("A_notes", 20), ("B_target", 16), ("B_variants", 18),
        ("B_flag", 14), ("B_notes", 20), ("final_target", 18), ("final_variants", 20), ("adjudicator_notes", 24)]
wb = Workbook(); ws0 = wb.active; ws0.title = "instructions"
for i, line in enumerate([
        "Adjudication sheet: tokens on which annotators A and B differ (target, variants, or flag).",
        "For every row, enter final_target and final_variants (variants separated by |). Read GUIDELINES.md first.",
        f"final_target is prefilled only where A and B gave the same target. Write {EMPTY} for a deliberately empty target.",
        "Leave no final_target blank. Do not reorder rows or edit the other columns."], 1):
    ws0.cell(i, 1, line); ws0.column_dimensions["A"].width = 120
ws = wb.create_sheet("adjudication")
YEL, GREY = PatternFill("solid", fgColor="FFF7D6"), PatternFill("solid", fgColor="EDEDED")
for j, (n, w) in enumerate(cols, 1):
    c = ws.cell(1, j, n); c.font = Font(bold=True); c.fill = YEL if n.startswith("final") or n.startswith("adjud") else GREY
    ws.column_dimensions[c.column_letter].width = w
for r, i in enumerate(np.array(items)[need], 2):
    a, b, k = A.loc[i], B.loc[i], K.loc[i]
    vals = dict(item_id=i, comment_no=k.comment_no, source_token=k.source_token, lid_tag=k.lid_tag, context=k.context,
                silver_target=show(k.silver_target_raw), A_target=show(a.target), A_variants="|".join(a.variants),
                A_flag=a.flag, A_notes=a.notes, B_target=show(b.target), B_variants="|".join(b.variants), B_flag=b.flag,
                B_notes=b.notes, final_target=show(a.target) if a.target == b.target else "",
                final_variants="|".join(sorted(set(a.variants) | set(b.variants))) if a.target == b.target else "",
                adjudicator_notes="")
    for j, (n, _) in enumerate(cols, 1):
        cell = ws.cell(r, j, vals[n]); cell.number_format = "@"
        cell.alignment = Alignment(wrap_text=(n == "context"), vertical="top")
ws.freeze_panes = "C2"
wb.save(args.out + "_adjudication.xlsx")
print(f"\nwrote {args.out}_agreement.txt, {args.out}_agreed.csv ({len(agreed)} tokens), "
      f"{args.out}_adjudication.xlsx ({int(need.sum())} tokens)")
