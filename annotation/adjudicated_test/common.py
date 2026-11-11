"""Shared helpers for the adjudicated-test scripts: reading sheets and normalizing targets."""
import os, sys, unicodedata
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "kaggle", "src"))
from banglish_align import GLUE, join_spans  # noqa: E402,F401

GLUE_MARK = "~"
EMPTY = "<empty>"          # explicit empty target in the adjudication sheet
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}


def bn(s):                 # identical to analysis/review_analyses.py
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


def norm_target(x):
    """Sheet cell -> internal target: NFC, single spaces, leading ~ -> GLUE, '<empty>' -> ''."""
    x = "" if x is None or (isinstance(x, float) and x != x) else str(x)
    x = " ".join(bn(x).split())
    if x == EMPTY:
        return ""
    if x.startswith(GLUE_MARK):
        x = GLUE + x[len(GLUE_MARK):].lstrip()
    return x


def norm_variants(x, main):
    x = "" if x is None or (isinstance(x, float) and x != x) else str(x)
    out = []
    for v in x.split("|"):
        v = norm_target(v)
        if v and v != main and v not in out:
            out.append(v)
    return out


def show(t):
    return EMPTY if t == "" else t.replace(GLUE, GLUE_MARK)


def tokens_key():
    return pd.read_csv(os.path.join(HERE, "tokens_key.csv"), keep_default_na=False, dtype={"token_no": int})


def sample_key():
    return pd.read_csv(os.path.join(HERE, "sample_key.csv"), keep_default_na=False)


def read_sheet(path):
    """Annotator sheet -> DataFrame indexed by item_id with target (str) and variants (list)."""
    s = pd.read_excel(path, sheet_name="tokens", dtype=str, keep_default_na=False)
    need = {"item_id", "corrected_target", "acceptable_variants", "flag", "notes"}
    missing = need - set(s.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    if s.item_id.duplicated().any():
        raise SystemExit(f"{path}: duplicated item_id values")
    s = s.set_index("item_id")
    s["target"] = s.corrected_target.map(norm_target)
    s["variants"] = [norm_variants(v, t) for v, t in zip(s.acceptable_variants, s.target)]
    s["flag"] = s.flag.fillna("").str.strip()
    return s[["target", "variants", "flag", "notes"]]
