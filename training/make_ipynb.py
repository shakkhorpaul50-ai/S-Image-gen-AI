"""Convert kaggle_train_150m.py (# %% cells) to .ipynb (stdlib only)."""
import json

SRC = "kaggle_train_150m.py"
DST = "kaggle_train_150m.ipynb"

with open(SRC, encoding="utf-8") as f:
    lines = f.read().splitlines()

cells = []
cur_kind, cur = None, []


def flush():
    global cur_kind, cur
    if cur_kind is None:
        return
    if cur_kind == "markdown":
        text = [ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln for ln in cur]
        while text and text[0].strip() == "":
            text.pop(0)
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": "\n".join(text)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": "\n".join(cur)})
    cur_kind, cur = None, []


for ln in lines:
    s = ln.strip()
    if s.startswith("# %% [markdown]"):
        flush()
        cur_kind = "markdown"
    elif s.startswith("# %%"):
        flush()
        cur_kind = "code"
    else:
        if cur_kind is None:
            cur_kind = "code"
        cur.append(ln)
flush()

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {
          "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python", "version": "3.10"}},
      "cells": cells}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"wrote {DST}: {len(cells)} cells "
      f"({sum(c['cell_type'] == 'code' for c in cells)} code, "
      f"{sum(c['cell_type'] == 'markdown' for c in cells)} markdown)")
