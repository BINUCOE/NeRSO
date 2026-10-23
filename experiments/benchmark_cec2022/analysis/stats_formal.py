#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_formal.py — EIO 工程统计与表格生成

vendor 自 AD 仓库 stats_formal.py（备忘录 v1.4 第五部分，R7 口径原样保留：
Wilcoxon 符号秩按 run_id 配对；Friedman + Nemenyi，CD = q·sqrt(k(k+1)/(6N))），
仅 3 处适配：
  1. FLAGSHIPS 硬编码 → --flagships 参数（默认 EIO，逗号分隔多个旗舰）
  2. 曲线输入适配本工程 runner 的单文件 conv_curves.npz
     （key = "algo__func_id__dim__run_id"），聚合输出 conv_curves_avg.npz
  3. 默认结果目录 = benchmark_cec2022/data/formal

PyCharm 绿色直跑；命令行 --results / --flagships 可选。
"""

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats as sst

# Nemenyi q_alpha (alpha=0.05, two-tailed)
Q_ALPHA_05 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
              6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def _resolve_csv(path, what):
    p = os.path.normpath(os.path.abspath(path))
    if not os.path.isfile(p):
        raise SystemExit("[{}] 找不到{}：\n  尝试路径：{}".format(
            os.path.basename(__file__), what, p))
    return p


def _norm_dir(path):
    return os.path.normpath(os.path.abspath(path))


def load_raw(path):
    rows = []
    with open(path, "r") as fp:
        for r in csv.DictReader(fp):
            rows.append(dict(algo=r["algo"], func_id=int(r["func_id"]),
                             dim=int(r["dim"]), run_id=int(r["run_id"]),
                             error=float(r["error"]),
                             best_fitness=float(r["best_fitness"]),
                             f_global=float(r["f_global"])))
    return rows


def summarize(rows, results_dir):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["func_id"], r["dim"], r["algo"])].append(r)
    fg = {(fid, d, a): rs[0]["f_global"] for (fid, d, a), rs in cells.items()}

    full_path = os.path.join(results_dir, "cec2022_stats_full.csv")
    sum_path = os.path.join(results_dir, "cec2022_summary_by_function.csv")
    with open(full_path, "w", newline="") as fp, open(sum_path, "w", newline="") as fs:
        w = csv.writer(fp)
        w.writerow(["func_id", "dim", "algo", "n", "mean", "std", "best", "worst", "median"])
        ws = csv.writer(fs)
        ws.writerow(["func_id", "dim", "algo", "f_global", "mean_error"])
        for (fid, d, a) in sorted(cells):
            errs = np.array([r["error"] for r in cells[(fid, d, a)]])
            w.writerow([fid, d, a, len(errs), "{:.6e}".format(errs.mean()),
                        "{:.6e}".format(errs.std(ddof=1) if len(errs) > 1 else 0.0),
                        "{:.6e}".format(errs.min()), "{:.6e}".format(errs.max()),
                        "{:.6e}".format(np.median(errs))])
            ws.writerow([fid, d, a, "{:.10e}".format(fg[(fid, d, a)]),
                         "{:.6e}".format(errs.mean())])
    print("[stats] {} / {}".format(full_path, sum_path))
    return cells


def wilcoxon_table(rows, results_dir, flagships):
    by = defaultdict(dict)
    for r in rows:
        by[(r["func_id"], r["dim"], r["algo"])][r["run_id"]] = r["error"]
    out_path = os.path.join(results_dir, "cec2022_wilcoxon.csv")
    with open(out_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["func_id", "dim", "flagship", "opponent", "n_pairs",
                    "median_flag", "median_opp", "p_value", "verdict"])
        for (fid, d, flag) in sorted(by):
            if flag not in flagships:
                continue
            algos = sorted({a for (f2, d2, a) in by if f2 == fid and d2 == d})
            for opp in algos:
                if opp == flag:
                    continue
                ra, rb = by[(fid, d, flag)], by[(fid, d, opp)]
                common = sorted(set(ra) & set(rb))
                if not common:
                    continue
                xa = np.array([ra[i] for i in common])
                xb = np.array([rb[i] for i in common])
                diff = xa - xb
                if np.all(diff == 0):
                    p, verdict = float("nan"), "="
                else:
                    p = float(sst.wilcoxon(xa, xb)[1])
                    ma, mb = np.median(xa), np.median(xb)
                    verdict = ("+" if ma < mb else ("-" if ma > mb else "="))
                    if p >= 0.05:
                        verdict = "="
                w.writerow([fid, d, flag, opp, len(common),
                            "{:.6e}".format(np.median(xa)), "{:.6e}".format(np.median(xb)),
                            "{:.4e}".format(p), verdict])
    print("[stats] {}".format(out_path))


def friedman_nemenyi(rows, results_dir):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["func_id"], r["dim"], r["algo"])].append(r["error"])
    dims = sorted({d for (_, d, _) in cells})
    lines = []
    for d in dims:
        funcs = sorted({f for (f, d2, _) in cells if d2 == d})
        algos = sorted({a for (_, d2, a) in cells if d2 == d})
        M = np.array([[np.mean(cells[(f, d, a)]) for a in algos] for f in funcs])
        ranks = np.array([sst.rankdata(row) for row in M])
        avg = ranks.mean(axis=0)
        k, nfn = len(algos), len(funcs)
        stat, p = sst.friedmanchisquare(*[M[:, j] for j in range(k)])
        q = Q_ALPHA_05.get(k, 3.164)
        cd = q * np.sqrt(k * (k + 1) / (6.0 * nfn))
        lines.append("D={}: Friedman chi2={:.3f} p={:.4e} | Nemenyi CD={:.3f} (k={}, N={})".format(
            d, stat, p, cd, k, nfn))
        for j in np.argsort(avg):
            lines.append("  {:<14s} avg_rank={:.3f}".format(algos[j], avg[j]))
        sig = []
        for i in range(k):
            for j in range(i + 1, k):
                if abs(avg[i] - avg[j]) > cd:
                    sig.append("{} vs {} (|d|={:.2f})".format(algos[i], algos[j], abs(avg[i] - avg[j])))
        lines.append("  CD 显著对: " + ("; ".join(sig) if sig else "无"))
    out_path = os.path.join(results_dir, "cec2022_friedman.txt")
    with open(out_path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("[stats] {}".format(out_path))
    for ln in lines:
        print("    " + ln)


def aggregate_curves(results_dir):
    """读取 runner 单文件 conv_curves.npz（key = algo__func__dim__run），
    聚合为 key "cec2022_F{02d}_D{d}__{algo}" 的平均曲线。"""
    src = os.path.join(results_dir, "conv_curves.npz")
    if not os.path.exists(src):
        print("[stats] 跳过曲线聚合：无 {}".format(src))
        return
    cells = defaultdict(list)
    with np.load(src) as z:
        for k in z.files:
            parts = k.split("__")
            if len(parts) != 4:
                continue
            algo, fid, d, _ = parts
            cells[(int(fid), int(d), algo)].append(z[k])
    store = {}
    for (fid, d, algo), curves in sorted(cells.items()):
        store["cec2022_F{:02d}_D{:d}__{}".format(fid, d, algo)] = np.mean(curves, axis=0)
    out_path = os.path.join(results_dir, "conv_curves_avg.npz")
    np.savez(out_path, **store)
    print("[stats] {}（{} 条平均曲线）".format(out_path, len(store)))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(here, "..", "data", "formal"))
    ap.add_argument("--flagships", default="EIO",
                    help="旗舰算法（逗号分隔，Wilcoxon 表的基准列）")
    a = ap.parse_args()
    a.results = _norm_dir(a.results)
    flagships = [s.strip() for s in a.flagships.split(",") if s.strip()]
    raw_path = _resolve_csv(os.path.join(a.results, "cec2022_raw_results.csv"),
                            "基准原始成绩（cec2022_raw_results.csv）")
    rows = load_raw(raw_path)
    print("[stats] 载入 {} 行 | 旗舰: {}".format(len(rows), flagships))
    summarize(rows, a.results)
    wilcoxon_table(rows, a.results, flagships)
    friedman_nemenyi(rows, a.results)
    aggregate_curves(a.results)


if __name__ == "__main__":
    main()
