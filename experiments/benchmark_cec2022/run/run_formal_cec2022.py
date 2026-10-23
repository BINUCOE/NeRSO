#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_formal_cec2022.py — EIO 工程 CEC2022 正式基准运行器

以 AD 仓库 E1 版为基准改造，协议口径（N=50、预算、seed 公式、FE 网格、
CSV schema、resume、并行 Pool）一行未改，仅：
  1. 算法构造改走 core.common.make()（注册表）
  2. 所有算法统一套 FECounter（公平性唯一口径）
  3. 每 run 记 runtime 落 CSV（计时需求）
  4. 新增 --smoke 冒烟模式（无 opfunu 环境用经典函数替代）

PyCharm 用法：直接绿色运行 = 正式全量；勾选/修改下方 CONFIG 或
命令行参数均可（命令行仅为可选兼容）。
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # 工程根（core 包所在）
sys.path.insert(0, str(HERE.parents[3]))  # 兼容从仓库根运行

from core.common import (FECounter, build_problem, fe_grid, make, seed_for, algo_names)

# ─────────────────── CONFIG（PyCharm 一键运行改这里）───────────────────
CONFIG = {"algos": None,  # None = 全部已注册；或 ["NeRSO","NPDOA",...]
          "smoke": False,  # True = 冒烟模式（N=20, FE=1000, D=10, 代表函数）
          "runs": None,  # None = 协议默认 30；smoke 时默认 3
          "workers": None,  # None = min(12, cpu-2)
          "out_dir": str(HERE.parent / "data" / "formal"), }

# 冒烟口径（按需求方要求：不要跑满 12 个 F，挑代表性的即可）
SMOKE_FUNCS = (1, 4, 6, 9, 11)
SMOKE = {"N": 20, "BUDGET": {10: 1000}, "DIMS": (10,), "RUNS": 3}

CSV_FIELDS = ["algo", "func_id", "dim", "run_id", "seed", "budget", "best_fitness", "error", "f_global", "fe", "gens",
              "runtime"]


# ─────────────────── 单个 run ───────────────────
def run_task(args):
    algo_name, func_id, dim, run_id, smoke = args
    t0 = time.perf_counter()
    
    seed = seed_for(run_id, func_id, dim)
    np.random.seed(seed)
    
    func_raw, bounds, f_global = build_problem(func_id, dim, smoke=smoke)
    cnt = FECounter(func_raw)
    N = SMOKE["N"] if smoke else 50
    budget = (SMOKE["BUDGET"][dim] if smoke else {10: 200000, 20: 500000}[dim])
    
    algo = make(algo_name, D=dim, N=N, bounds=bounds, func=cnt)
    algo.max_iter = budget // N  # 依赖 max_iter 的算法（wDecay/AD 系）所需
    algo.max_fe = budget  # 依赖 NFE 调度的算法（L-SHADE 系）所需
    
    if hasattr(algo, "start"):  # NPDOA：初始化评估显式化（= 原版 optimize 前半）
        algo.start()
    
    # 与 E1 逐位回归对齐：有自报 fe_count 的算法以自报为准做停止判定
    # （AD 开关板语义；wDecay/EIO/DE/NPD 系无 fe_count，走统一计数器）
    fe = (lambda: algo.fe_count) if hasattr(algo, "fe_count") else (lambda: cnt.n)
    
    grid = fe_grid(budget, N)
    curve = np.full(len(grid), np.inf)
    curve[0] = algo.g_fit  # grid[0] == N：初始化后首次评估
    
    fe_idx, gens = 1, 0
    while fe() < budget:
        algo.step()
        gens += 1
        now = fe()
        while fe_idx < len(grid) and now >= grid[fe_idx]:
            curve[fe_idx] = algo.g_fit
            fe_idx += 1
    # 前向填充，末尾兜底
    for i in range(1, len(curve)):
        if not np.isfinite(curve[i]):
            curve[i] = curve[i - 1]
    curve[-1] = algo.g_fit  # 最终值（FE 可能略超预算，社区惯例如实记录）
    
    # 公平性提示：自报 FE 与实测计数器应接近（E1 口径以自报为准；允许小偏差）
    if hasattr(algo, "fe_count") and abs(algo.fe_count - cnt.n) > max(10, 0.01 * cnt.n):
        print("  [warn] {}: fe_count {} 与实测 {} 偏差较大".format(algo_name, algo.fe_count, cnt.n), flush=True)
    
    runtime = round(time.perf_counter() - t0, 2)
    row = {"algo": algo_name, "func_id": func_id, "dim": dim, "run_id": run_id, "seed": seed, "budget": budget,
           "best_fitness": float(algo.g_fit), "error": abs(float(algo.g_fit) - f_global), "f_global": float(f_global),
           "fe": int(fe()), "gens": gens, "runtime": runtime}
    return row, curve


# ─────────────────── 主流程 ───────────────────
def load_done(csv_path):
    done = set()
    if csv_path.exists():
        with open(csv_path, newline="") as fp:
            for r in csv.DictReader(fp):
                done.add((r["algo"], r["func_id"], r["dim"], r["run_id"]))
    return done


def _fmt_hms(sec):
    sec = int(sec)
    if sec >= 3600:
        return "{:d}h{:02d}m".format(sec // 3600, (sec % 3600) // 60)
    if sec >= 60:
        return "{:d}m{:02d}s".format(sec // 60, sec % 60)
    return "{:d}s".format(sec)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="EIO — CEC2022 formal benchmark")
    ap.add_argument("--algos", default=None, help="逗号分隔算法名（默认全部已注册）")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：N=20, FE=1000, D=10, 代表函数 F1/4/6/9/11")
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    
    smoke = a.smoke or CONFIG["smoke"]
    algos = (a.algos.split(",") if a.algos else CONFIG["algos"] or algo_names())
    runs = a.runs or CONFIG["runs"] or (SMOKE["RUNS"] if smoke else 30)
    workers = a.workers or CONFIG["workers"] or max(1, min(12, os.cpu_count() - 2))
    out_dir = Path(a.out_dir or CONFIG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    funcs = SMOKE_FUNCS if smoke else list(range(1, 13))
    dims = SMOKE["DIMS"] if smoke else [10, 20]
    tasks = [(al, fi, d, r, smoke) for al in algos for fi in funcs for d in dims for r in range(runs)]
    
    csv_path = out_dir / ("smoke_results.csv" if smoke else "cec2022_raw_results.csv")
    done = load_done(csv_path)
    todo = [t for t in tasks if (t[0], str(t[1]), str(t[2]), str(t[3])) not in done]
    print("[E1] 算法 {} 个 | 函数 {} | 维度 {} | runs {} | 总任务 {} | "
          "已完成 {} | 待跑 {}".format(len(algos), funcs, dims, runs, len(tasks), len(tasks) - len(todo), len(todo)),
          flush=True)
    if not todo:
        print("[E1] 全部已完成，无事可做。")
        return
    
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "smoke": smoke, "algos": algos, "funcs": funcs,
                "dims": dims, "runs": runs, "seed_base": 666,
                "budget": (SMOKE["BUDGET"] if smoke else {"10": 200000, "20": 500000}), "code_sha256": {}, }
    for mod in ("core/common.py", "core/ad_switchboard.py", "core/pso_family.py", "core/sca_family.py",
                "core/eio_family.py", "core/eio_v3.py", "core/eio_v4.py", "core/npdoa.py", "core/cec_schwefel_fix.py"):
        p = HERE.parents[2] / mod
        if p.exists():
            manifest["code_sha256"][mod] = sha256_of(p)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    
    curves, write_i = {}, 0
    t_start = time.perf_counter()
    write_path = csv_path
    with open(write_path, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        if fp.tell() == 0:
            writer.writeheader()
        last_print = time.perf_counter() - 30  # 保证开跑 30 秒内出第一条进度
        count_step = max(1, min(len(todo) // 20, 50))
        with mp.Pool(workers) as pool:
            for row, curve in pool.imap_unordered(run_task, todo):
                writer.writerow(row)
                fp.flush()
                curves[(row["algo"], row["func_id"], row["dim"], row["run_id"])] = curve
                write_i += 1
                now = time.perf_counter()
                if (write_i % count_step == 0 or write_i == len(todo) or now - last_print >= 60):
                    el = now - t_start
                    eta = el / write_i * (len(todo) - write_i)
                    print("[E1] {}/{} ({:.1f}%) | 已用 {} | ETA {}".format(write_i, len(todo),
                                                                           100.0 * write_i / len(todo), _fmt_hms(el),
                                                                           _fmt_hms(eta)), flush=True)
                    last_print = now
    
    out_npz = out_dir / ("smoke_curves.npz" if smoke else "conv_curves.npz")
    # 增量合并：保留已有曲线（如从 E1 存档导入的旧基线），只覆盖本次重跑的键
    merged = {}
    if out_npz.exists():
        with np.load(out_npz) as z:
            merged = {k: z[k] for k in z.files}
    merged.update({"__".join(map(str, k)): v for k, v in curves.items()})
    np.savez_compressed(out_npz, **merged)
    print("[E1] 完成。结果: {}".format(write_path))


if __name__ == "__main__":
    main()
