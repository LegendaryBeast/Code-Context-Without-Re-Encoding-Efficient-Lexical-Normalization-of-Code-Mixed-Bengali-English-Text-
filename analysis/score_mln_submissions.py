"""Rescore the official MultiLexNorm 2021 submissions on the code-switched test sets (published rows of Table 7).

Needs a MultiLexNorm checkout (https://bitbucket.org/robvanderg/multilexnorm, commit e58e946375299a04355fa2d7fa9e1c769234e68a)
and the converted gold in results/gpu/nb9_multilexnorm/mln_{iden,trde}.csv. ERR uses the official formula (case-sensitive),
exactly as analysis/analyze_mln.py does for the systems trained here.

Run from the repo root:  python analysis/score_mln_submissions.py --repo <multilexnorm checkout>
"""
import argparse, glob, json, os, unicodedata
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--dir", default=os.path.join(ROOT, "results", "gpu", "nb9_multilexnorm"))
args = ap.parse_args()


def load(path):
    out, cur = [], []
    for line in open(path, encoding="utf-8"):
        t = line.rstrip("\n").split("\t")
        if t == [""]:
            if cur:
                out.append(cur)
            cur = []
            continue
        cur.append((t[0], t[1] if len(t) > 1 else ""))
    if cur:
        out.append(cur)
    return out


sub = os.path.join(args.repo, "submissions")
for lang in ("iden", "trde"):
    d = pd.read_csv(os.path.join(args.dir, f"mln_{lang}.csv"), keep_default_na=False)
    d = d[d.id.str.startswith(f"{lang}_test_")]
    gold = [json.loads(s) for s in d.target_spans]
    raw = [[t.rsplit("/", 1)[0] for t in tl.split(" ")] for tl in d.token_labels]
    rows = []
    for team in sorted(os.listdir(sub)):
        f = glob.glob(os.path.join(sub, team, "intrinsic_evaluation", lang, "*"))
        if not f or team in ("gold", "LAI"):
            continue
        pred = load(f[0])
        assert len(pred) == len(gold) and all(len(p) == len(g) for p, g in zip(pred, gold)), team
        cor = chg = tot = 0
        for ps, g, r in zip(pred, gold, raw):
            for (_, pn), gn, rw in zip(ps, g, r):
                cor += unicodedata.normalize("NFC", pn) == unicodedata.normalize("NFC", gn)
                chg += rw != gn
                tot += 1
        rows.append((team, 100 * (cor - tot + chg) / chg))
    print(f"\n== {lang}: test ERR of the official submissions")
    for team, e in sorted(rows, key=lambda x: -x[1]):
        print(f"  {team:12s} {e:6.2f}")
