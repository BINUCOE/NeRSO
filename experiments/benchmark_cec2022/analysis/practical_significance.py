#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# practical_significance.py — ERT + 实用等价三层口径分析（预注册版）
# 用法: python practical_significance.py <results_dir1> [results_dir2 ...]
# 输入: 各目录 cec2022_raw_results.csv + conv_curves.npz（schema 相同）
# 预注册常数（2026-10-13 冻结，事后不得调整）:
#   EPS_BAND = 1e-8   实用等价带：双方中位均低于此 => 强制平局
#   DELTA    = 0.5    dex 带宽：|log10 中位比| < 0.5 => 平局（带外）
#   TARGETS  = (1e-2, 1e-5, 1e-8)  ERT 三级目标精度
# 输出: 1) 严格 vs 实用 W/T/L  2) 解决格数阶梯  3) ERTx成功率分层表
import sys
import numpy as np
import pandas as pd
from pathlib import Path

EPS_BAND = 1e-8
DELTA = 0.5
TARGETS = (1e-2, 1e-5, 1e-8)
PROT = "EIO"
FOES = ("NPDOA", "AD-SCA", "AD-PSO", "SCA", "PSO", "wDecay",
        "DE", "SHADE", "LSHADE", "NLSHADERSPMID", "NeRSO")


# 曲线存原始目标值(bf)，需减 f_global 得误差。
# opfunu 1.0.1 的 f_global（实测，注意：不是 fid*100 的猜值！）：
FG_FALLBACK = {1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0, 6: 1800.0,
               7: 2000.0, 8: 2200.0, 9: 2300.0, 10: 2400.0, 11: 2600.0, 12: 2700.0}
FG = {}


def load(dirs):
    frames, store = [], {}
    for d in dirs:
        d = Path(d)
        frames.append(pd.read_csv(d / "cec2022_raw_results.csv"))
        with np.load(d / "conv_curves.npz") as z:
            store.update({k: z[k] for k in z.files})
    global FG
    try:
        from core.common import build_problem
        for pid in range(1, 13):
            _, _, FG[pid] = build_problem(pid, 10, smoke=False)
    except Exception:
        FG = dict(FG_FALLBACK)  # opfunu 不可用时用实测硬表
    return pd.concat(frames, ignore_index=True), store


def fe_grid(budget):
    g = np.unique(np.round(
        np.logspace(np.log10(50), np.log10(budget), 121)).astype(int))
    g[-1] = budget
    return g


def fht(curve, fe, target, fgl):
    # 曲线=原始目标值 bf；误差=|bf-fgl|；对数-对数插值首达 FE；未命中 None
    logt = np.log10(target)
    e = np.where(np.isinf(curve), np.nan, np.abs(curve - fgl))
    le = np.log10(np.maximum(e, 1e-300))
    hit = np.where(le <= logt)[0]
    if len(hit) == 0:
        return None
    i = max(int(hit[0]), 1)
    x0, x1 = np.log10(fe[i - 1]), np.log10(fe[i])
    y0, y1 = le[i - 1], le[i]
    if y1 == y0:
        return fe[i]
    return min(10 ** (x0 + (logt - y0) * (x1 - x0) / (y1 - y0)), fe[-1])


def ert_stats(store, algo, fid, dim, budget):
    fe = fe_grid(budget)
    out = {}
    for t in TARGETS:
        fhts, ns = [], 0
        for r in range(60):
            k = "%s__%d__%d__%d" % (algo, fid, dim, r)
            if k not in store:
                continue
            f = fht(store[k], fe, t, FG[fid])
            if f is None:
                fhts.append(fe[-1])
            else:
                fhts.append(f)
                ns += 1
        n = len(fhts)
        sr = ns / max(n, 1)
        ert = (sum(fhts) / ns) if ns > 0 else None
        out[t] = (sr, ert)
    return out


def main():
    df, store = load(sys.argv[1:])
    med = df.groupby(["algo", "func_id", "dim"]).error.median().reset_index()
    piv = med.pivot_table(index=["func_id", "dim"],
                          columns="algo", values="error")
    print("== 1) 严格 vs 实用 W/T/L（24 格，中位口径，主角 %s）==" % PROT)
    print("%-16s %-14s %-14s" % ("对手", "严格 胜/平/负", "实用 胜/平/负"))
    for b in FOES:
        if b not in piv.columns:
            continue
        sw = st = sl = pw = pt = pl = 0
        for idx in piv.index:
            ma, mb = piv.loc[idx, PROT], piv.loc[idx, b]
            if not np.isfinite(ma) or not np.isfinite(mb):
                continue
            if ma < mb: sw += 1
            elif ma > mb: sl += 1
            else: st += 1
            if (ma < EPS_BAND and mb < EPS_BAND) or \
                    abs(np.log10(ma / mb)) < DELTA:
                pt += 1
            elif ma < mb: pw += 1
            else: pl += 1
        print("%-16s %6d/%2d/%2d   %6d/%2d/%2d" % (b, sw, st, sl, pw, pt, pl))
    print()
    print("== 2) 解决格数阶梯（中位误差 < eps，共 24 格）==")
    for eps in TARGETS:
        n = (piv < eps).sum().sort_values(ascending=False)
        kept = ", ".join("%s:%d" % (a, c) for a, c in n.items() if c > 0)
        print("  eps=%.0e : %s" % (eps, kept))
    print()
    print("== 3) ERT(kFE)x成功率 分层表（D=10，代表格）==")
    algos = [PROT, "NPDOA", "AD-SCA", "PSO", "SHADE"]
    hdr = "%-6s" % "格子"
    for a in algos:
        hdr += "  %-26s" % (a[:11] + " 1e-2|1e-5|1e-8")
    print(hdr)
    for fid in (1, 6, 9, 11):
        row = "F%-5d" % fid
        for a in algos:
            st = ert_stats(store, a, fid, 10, 200000)
            cells = []
            for t in TARGETS:
                sr, ert = st[t]
                if ert is None:
                    cells.append("    --(%2.0f%%)" % (100 * sr))
                else:
                    cells.append("%4.0fk(%2.0f%%)" % (ert / 1000, 100 * sr))
            row += "  %-26s" % "|".join(cells)
        print(row)


if __name__ == "__main__":
    main()
