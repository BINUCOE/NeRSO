# KEEP：v3 插桩已关闭（P1-P7 结论固化）；本文件是仓库唯一的插桩批模板，供 v4/NeRSO 后续插桩实验复用（三司会审终裁）。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
instrumented_runs.py — 预测体系插桩实验（脚本 B，P-E1~P-E7 一次跑全）

矩阵：12 函数 × 2 维 × 3 seeds，全预算（200k/500k FE），与用户机器全量同协议。
事件捕获纪律（A5）：清零型事件用"精确零+路径排除"签名，禁跨代阈值法。
  逃逸 fires_t = #(S_post == 0.0 精确) & (S_pre > 0.1) —— 泄漏分支 S_pre(1−β) 不会精确归零。

用法：python tests/instrumented_runs.py            # 全矩阵（~1 小时，12 workers）
      python tests/instrumented_runs.py --smoke    # 2 函数单 seed 短预算（分钟级）
输出：tests/instrumented_summary.csv（P-E 指标表，直接进论文）
      tests/instrumented_traces.npz（原始轨迹）
"""
import sys
import os
import csv
import time
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from core.common import FECounter, build_problem, seed_for
from core.eio_v3 import EIO_v3, _eig_period

# 拉伸律（自由响应零交叉实测，铁案）：u → 周期
STRETCH_LAW = [(0.1, 19.3), (1.0, 22.5), (2.0, 33.8), (4.0, 54.0)]


def band_period(x, lo=5.0, hi=200.0):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 * lo:
        return np.nan, np.nan
    spec = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    fr = np.fft.rfftfreq(n, 1.0)
    m = (fr >= 1.0 / hi) & (fr <= 1.0 / lo)
    if m.sum() < 3:
        return np.nan, np.nan
    b, fb = spec[m], fr[m]
    k = int(np.argmax(b))
    return 1.0 / fb[k], (b[k] / np.median(b) if np.median(b) > 0 else np.nan)


def predicted_period(u):
    us = np.array([s[0] for s in STRETCH_LAW])
    ps = np.array([s[1] for s in STRETCH_LAW])
    return float(np.exp(np.interp(np.log(max(u, 1e-3)), np.log(us), np.log(ps))))


def run_one(args):
    fid, dim, seed, budget, smoke = args
    np.random.seed(seed_for(seed, fid, dim))
    func_raw, bounds, _ = build_problem(fid, dim, smoke=smoke)
    cnt = FECounter(func_raw)
    algo = EIO_v3(D=dim, N=50, bounds=bounds, func=cnt)
    algo.max_iter = budget // 50
    algo.max_fe = budget

    log = {k: [] for k in ("E", "I", "theta", "amp", "spread", "ei_abs",
                           "disp", "fires", "refine_acc", "gfit")}
    while cnt.n < budget:
        pos_pre = algo.pos.copy()
        s_pre = algo.S.copy()
        pb_pre = algo.pb_fit.copy()
        algo.step()
        # 逃逸签名：精确零 + 路径排除（S_pre>0.1 排除泄漏归零假象）
        fires = int(((algo.S == 0.0) & (s_pre > 0.1) &
                     (algo.pb_fit <= pb_pre)).sum())
        d = algo.E - algo.I
        log["E"].append(algo.E.copy())
        log["I"].append(algo.I.copy())
        log["theta"].append(float(1.0 + algo.coupling * np.tanh(algo.E_slow.mean())))
        log["amp"].append(float(np.mean(np.abs(d) * (1.0 + algo.coupling * np.tanh(algo.E_slow)))))
        log["spread"].append(float(np.linalg.norm(
            algo.pos - algo.pos.mean(axis=0), axis=1).mean()))
        log["ei_abs"].append(float(np.abs(algo.rms_in * 1.0)))  # rms_in≈|ei_in| 尺度
        log["disp"].append(float(np.linalg.norm(algo.pos - pos_pre, axis=1).mean()))
        log["fires"].append(fires)
        log["refine_acc"].append(int((algo.pb_fit < pb_pre).sum()))
        log["gfit"].append(float(algo.g_fit))

    E = np.array(log["E"]); I = np.array(log["I"])
    d = E - I
    fires = np.array(log["fires"], float)
    G = len(fires)

    # P-E1 周期与拉伸律：每粒子带限周期中位 + 饱和率先验预测
    pers = [band_period(d[:, i]) for i in range(0, min(algo.N, 50), 7)]
    pers = [p for p, s in pers if not np.isnan(p)]
    per_med = float(np.median(pers)) if pers else np.nan
    sat = float((np.abs(E) > 0.9).mean())
    u_eff = float(np.median(log["ei_abs"]))
    per_pred = predicted_period(u_eff)

    # P-E3 θ 驻留
    th = np.array(log["theta"])
    dw = []
    cur = 0
    for x in (th > np.median(th)):
        cur = cur + 1 if x else 0
        dw.append(cur)
    dwell_med = float(np.median([x for x in dw if x > 0])) if any(dw) else np.nan

    # P-E4 Fano（100 代窗）
    wins = [fires[i:i + 100].sum() for i in range(0, G - 99, 50)]
    wins = np.array(wins, float)
    fano = float(wins.var() / wins.mean()) if wins.mean() > 0 else np.nan

    # P-E2 amp-位移耦合（Spearman）
    from scipy.stats import spearmanr
    amp = np.array(log["amp"]); disp = np.array(log["disp"])
    rho = float(spearmanr(amp, disp)[0])

    # P-E6 kick/spread：fires>0 的代，kick=S·L_env≈L_env(≈1.0+)，比 spread
    spread = np.array(log["spread"])
    fire_idx = np.where(fires > 0)[0]
    kick_ratio = float(np.median(1.0 / (spread[fire_idx] + 1e-12))) if len(fire_idx) else np.nan

    row = {
        "func": fid, "dim": dim, "seed": seed,
        "P1_period_meas": round(per_med, 2), "P1_period_pred": round(per_pred, 2),
        "P1_sat": round(sat, 3), "P1_u_eff": round(u_eff, 3),
        "P2_spearman": round(rho, 3),
        "P3_theta_dwell": round(dwell_med, 1),
        "P4_fano": round(fano, 2),
        "P6_kick_over_spread": round(kick_ratio, 2) if kick_ratio == kick_ratio else "",
        "fires_total": int(fires.sum()), "refine_acc_total": int(sum(log["refine_acc"])),
        "best": "%.6e" % algo.g_fit,
    }
    trace = {k: np.array(v) for k, v in log.items() if k in
             ("theta", "amp", "spread", "fires", "gfit")}
    return row, trace, (fid, dim, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.smoke:
        tasks = [(f, 10, 0, 12500, True) for f in (1, 9)]
    else:
        tasks = [(f, d, s, {10: 200000, 20: 500000}[d], False)
                 for f in range(1, 13) for d in (10, 20) for s in (3, 4, 5)]
    print("tasks:", len(tasks), flush=True)
    summary_path = HERE / ("instrumented_summary_smoke.csv" if a.smoke
                           else "instrumented_summary.csv")
    fields = None
    traces = {}
    t0 = time.perf_counter()
    with mp.Pool(a.workers) as pool, open(summary_path, "a", newline="") as fp:
        writer = None
        for i, (row, trace, key) in enumerate(pool.imap_unordered(run_one, tasks), 1):
            if fields is None:
                fields = list(row.keys())
                writer = csv.DictWriter(fp, fieldnames=fields)
                if fp.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            fp.flush()
            traces["__".join(map(str, key))] = trace
            if i % 10 == 0 or i == len(tasks):
                print("%d/%d | %.0fs" % (i, len(tasks), time.perf_counter() - t0), flush=True)
    np.savez_compressed(str(HERE / ("instrumented_traces_smoke.npz" if a.smoke
                                    else "instrumented_traces.npz")), **traces)
    print("saved", summary_path)


if __name__ == "__main__":
    main()
