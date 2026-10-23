# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_batches.py — 合并 v3/LOO 批次 CSV 与曲线 npz（schema 相同，去重）"""
import sys, csv
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "data" / "formal"


def merge_csv(out, *dirs):
    seen, rows = set(), []
    for d in dirs:
        p = Path(d) / "cec2022_raw_results.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            k = (r["algo"], r["func_id"], r["dim"], r["run_id"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    if rows:
        with open(out, "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("merged %d rows -> %s" % (len(rows), out))


def merge_npz(out, *dirs):
    store = {}
    for d in dirs:
        p = Path(d) / "conv_curves.npz"
        if p.exists():
            with np.load(p) as z:
                store.update({k: z[k] for k in z.files})
    np.savez_compressed(out, **store)
    print("merged %d curves -> %s" % (len(store), out))


if __name__ == "__main__":
    merge_csv(BASE / "cec2022_raw_results.csv", BASE, HERE.parent / "data" / "v3")
    merge_npz(BASE / "conv_curves.npz", BASE, HERE.parent / "data" / "v3")
