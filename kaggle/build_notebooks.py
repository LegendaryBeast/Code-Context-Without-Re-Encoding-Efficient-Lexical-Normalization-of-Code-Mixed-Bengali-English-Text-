"""Builds kaggle/notebooks/*.ipynb from kaggle/src/nb*.py (cells split on '# %%'), analysis/nb7 from analysis/src,
and kaggle/kaggle_dataset.zip (the Kaggle dataset: BanglaMix.csv, splits, CRF predictions, noise matrix).
Run from anywhere:  python kaggle/build_notebooks.py"""
import glob
import json
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def to_nb(src_path, out_path):
    cells, cur, kind = [], [], "code"

    def flush():
        text = "\n".join(cur).strip("\n")
        if not text:
            return
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text})

    for line in open(src_path, encoding="utf-8").read().splitlines():
        if line.startswith("# %% writefile "):
            # embed a sibling source file as a %%writefile cell, so the script stays testable on its own
            flush()
            name = line.split()[-1]
            body = open(os.path.join(os.path.dirname(src_path), name), encoding="utf-8").read().rstrip("\n")
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": f"%%writefile /kaggle/working/{name}\n{body}"})
            cur, kind = [], "code"
        elif line.startswith("# %%"):
            flush()
            cur, kind = [], "markdown" if "[markdown]" in line else "code"
        else:
            cur.append(line)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


for src in sorted(glob.glob(os.path.join(HERE, "src", "nb*.py"))):
    out = os.path.join(HERE, "notebooks", os.path.basename(src).replace(".py", ".ipynb"))
    to_nb(src, out)
    print("built", out)
for src in sorted(glob.glob(os.path.join(ROOT, "analysis", "src", "nb*.py"))):      # local notebooks: no GPU flag
    out = os.path.join(ROOT, "analysis", os.path.basename(src).replace(".py", ".ipynb"))
    to_nb(src, out)
    nb = json.load(open(out, encoding="utf-8")); nb["metadata"].pop("accelerator", None)
    json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("built", out)

# Kaggle dataset zip, built from the files kept in data/ and results/cpu/
import zipfile
zpath = os.path.join(HERE, "kaggle_dataset.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(ROOT, "data", "BanglaMix.csv"), "BanglaMix.csv")
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "splits", "*.csv"))):
        z.write(f, "splits/" + os.path.basename(f))
    z.write(os.path.join(ROOT, "results", "cpu", "04_noise_matrix.csv"), "noise_matrix.csv")
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "cpu", "crf_pred_seed*.pkl"))):
        d = pickle.load(open(f, "rb"))
        z.writestr(os.path.basename(f).replace(".pkl", ".json"), json.dumps({k: list(v) for k, v in d.items()}))
print("built", zpath, sorted(zipfile.ZipFile(zpath).namelist()))
