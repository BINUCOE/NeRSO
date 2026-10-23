#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tier_pareto.py — 精度分档的性能-效率综合表（IDE 一键直跑，2026-10-15 修订）

三档目标：ε ∈ {1e-3 工程档, 1e-8 研究档, 1e-10 极限档}。
每档输出：覆盖格数 | 解出格平均 ERT(kFE) | 折算平均墙钟(s)。
口径：ERT=BBOB 标准（未命中记预算，30 runs 均值）；墙钟=ERT×实测秒/FE（CSV runtime/fe）。

修订要点（相对初版）：
  1) f_global 一律由 core.common.build_problem 现取——与 runner 记账逐位一致；
     初版手工 FG 表把 F2/F4/F5/F6/F7 最优点写错（如 F2=400 实为 1100），已废止；
  2) 缺 conv_curves.npz 的结果目录自动跳过（打印警告），不再 FileNotFound 崩；
  3) CSV 无 fe 列时按维度预算回退；曲线长度与 FE 网格不齐时自动重采样对齐；
  4) CONFIG 区一键可改；默认吃 WS 下 data/formal / data/v3 / data/v4。

用法（PyCharm 绿色直跑，或命令行覆盖目录）：
  python experiments/benchmark_cec2022/tier_pareto.py [dir1 dir2 ...]
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG（IDE 一键区，改这里即可）──
HERE = Path(__file__).resolve().parent
WS = HERE.parents[2]
DEFAULT_DIRS = ["formal", "v3", "v4", "loo"]
TIERS = (1e-3, 1e-8, 1e-10)
BUDGET = {10: 200_000, 20: 500_000}
FOCUS = None  # None=全部算法；或填元组，如 ("NeRSO", "SHADE", "DE")

sys.path.insert(0, str(WS))
from core.common import build_problem  # f_global 唯一权威来源（含 Schwefel 补丁）

_FG = {}


def f_global(fid, dim):
    if (fid, dim) not in _FG:
        _, _, fg = build_problem(fid, dim, smoke=False)
        _FG[(fid, dim)] = fg
    return _FG[(fid, dim)]


def fe_grid(budget, n=121):
    g = np.unique(np.round(np.logspace(np.log10(50), np.log10(budget), n)).astype(int))
    g[-1] = budget
    return g


def fht(curve, fe, target, fg):
    """首次命中 FE（log-log 线性插值）；未命中返回 None。曲线可为 raw fitness 或 error。"""
    # 曲线一律为 raw best fitness（runner 存 algo.g_fit，run_formal_cec2022.py 已确认），
    # 不允许"自动判别"——raw fitness 域（如 F1≈300）与 error 域（如 F8≈1e10）会互相误判。
    logt = np.log10(target)
    c = np.asarray(curve, dtype=float)
    le = np.log10(np.maximum(np.abs(c - fg), 1e-300))
    m = len(c)
    if len(fe) != m:  # 网格不齐时按点数重建对数网格
        fe = fe_grid(fe[-1], n=m)
    hit = np.where(le <= logt)[0]
    if len(hit) == 0:
        return None
    i = max(int(hit[0]), 1)
    x0, x1 = np.log10(fe[i - 1]), np.log10(fe[i])
    y0, y1 = le[i - 1], le[i]
    if y1 == y0:
        return fe[i]
    return min(10 ** (x0 + (logt - y0) * (x1 - x0) / (y1 - y0)), fe[-1])


def discover_dirs():
    """自动发现：WS 根与 experiments/benchmark_cec2022 两处，凡含正式 CSV 的目录全部收。"""
    roots = [WS / "experiments" / "benchmark_cec2022" / "data", HERE.parent / "data"]
    found, seen = [], set()
    for r in roots:  # 后扫到的同名目录跳过（防双处拷贝导致双计）
        if not r.exists():
            continue
        for d in sorted(r.iterdir()):
            if d.is_dir() and (d / "cec2022_raw_results.csv").exists() and d.name not in seen:
                seen.add(d.name)
                found.append(d)
    return found


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or discover_dirs()
    frames, store, spf = [], {}, {}
    for d in dirs:
        csv = d / "cec2022_raw_results.csv"
        npz = d / "conv_curves.npz"
        if not csv.exists():
            print("[warn] 跳过 %s（无 CSV）" % d);
            continue
        fr = pd.read_csv(csv)
        frames.append(fr)
        if npz.exists():
            with np.load(npz) as z:
                store.update({k: z[k] for k in z.files})
        else:
            print("[warn] %s 无 conv_curves.npz，该目录只参与秒/FE 统计" % d)
        if "fe" in fr.columns:
            s = (fr.runtime / fr.fe)
        else:
            s = (fr.runtime / fr.dim.map(BUDGET))
        spf.update(s.groupby(fr.algo).median().to_dict())
    if not frames:
        raise SystemExit("没有任何可用结果目录（自动发现已扫描 WS 根与 experiments/benchmark_cec2022；"
                         "可用命令行显式传目录：python tier_pareto.py <dir> [dir2 ...]")
    df = pd.concat(frames, ignore_index=True)
    algos = sorted(df.algo.unique()) if FOCUS is None else [a for a in FOCUS if a in set(df.algo)]
    
    for eps in TIERS:
        print("\n== ε = %.0e 档 ==" % eps)
        print("%-16s %-6s %-14s %-14s" % ("算法", "覆盖", "平均ERT(kFE)", "平均墙钟(s)"))
        rows = []
        for a in algos:
            cov, erts = 0, []
            for fid in range(1, 13):
                for dim, budget in ((10, 200000), (20, 500000)):
                    fe = fe_grid(budget)
                    fg = f_global(fid, dim)
                    fhts = []
                    for r in range(30):
                        k = "%s__%d__%d__%d" % (a, fid, dim, r)
                        if k not in store:
                            continue
                        f = fht(store[k], fe, eps, fg)
                        fhts.append(fe[-1] if f is None else f)
                    if not fhts:
                        continue
                    ns = sum(1 for x in fhts if x < fe[-1])
                    if ns == 0:
                        continue
                    cov += 1
                    erts.append(sum(fhts) / ns)
            if cov:
                rows.append((a, cov, np.mean(erts) / 1000.0, np.mean(erts) * spf.get(a, np.nan)))
        for a, cov, mert, mwc in sorted(rows, key=lambda x: (-x[1], x[2])):
            print("%-16s %-6d %-14.1f %-14.1f" % (a, cov, mert, mwc))


if __name__ == "__main__":
    main()
