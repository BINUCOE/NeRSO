#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p8_audit.py — P-E8 切换时刻正式插桩审计：16 格补缺 + 对照臂（IDE 一键，增量落盘）

已有资产：tests/data/pilot_v4.npz 已有 8 格（F1/F3/F9/F11 × D10/D20 × 30 seeds ×
{EIO,NeRSO-bi,NeRSO-clock,NeRSO-fullEsc,AD-PSO}）的 switch_log（全预算，SW_ 键）——本脚本不重跑。
本脚本补正式 24 格的其余 8 函数（F2,F4,F5,F6,F7,F8,F10,F12）× 2 维 × 3 seeds，
臂 {NeRSO-fullEsc, NeRSO-clock}（时钟臂必须同批：其切换时刻≡0.6·预算是结构性基线，同图对照）。
顺带记录 S 逐代轨迹 + 散布轨迹 → SFA 驻留≈T_SFA 在 EIO 谱系的核验（P1 复现），一批两事。

产出 data/p8_audit.npz（键全部 ASCII）：
  SW_<arm>_F<f>_D<d>_s<r>  object array [(gen, "L_to_H"), ...]  切换表（算法内 switch_log）
  SJ_<arm>_F<f>_D<d>_s<r>  (gens, N) float16  SFA 积累器 S 逐代轨迹
  SP_<arm>_F<f>_D<d>_s<r>  (gens,)   float16  群体散布逐代轨迹
  ER_<arm>_F<f>_D<d>_s<r>  scalar            终误差 |g_fit − f_global|
种子口径与正式协议一致：seed_for(run, func, dim)，run ∈ {0,1,2}。
用法：PyCharm 绿色直跑；中断重跑自动续（键级跳过已完成的 run）。
"""
import os, sys, time, multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.common import FECounter, build_problem, seed_for, make, BUDGET

ARMS = ("NeRSO-fullEsc", "NeRSO-clock")  # 前身正典臂 + 时钟对照臂（同批同图）
FUNCS = (2, 4, 5, 6, 7, 8, 10, 12)  # pilot 已覆盖 {1,3,9,11}，此处补缺
SEEDS = (0, 1, 2)  # 插桩口径同 TEVC E4：3 seeds 全基准
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "p8_audit.npz")


def run(arg):
    """单 run：正式协议（全预算、seed_for 配对、FECounter 计费），
    逐步记录 S 轨迹与散布轨迹，终了取 switch_log。"""
    name, fid, dim, run_id = arg
    np.random.seed(seed_for(run_id, fid, dim))
    func_raw, bounds, f_global = build_problem(fid, dim, smoke=False)
    cnt = FECounter(func_raw)
    algo = make(name, D=dim, N=50, bounds=bounds, func=cnt)
    budget = BUDGET[dim]
    algo.max_iter = budget // 50
    algo.max_fe = budget
    s_traj, sp_traj = [], []
    while cnt.n < budget:
        algo.step()
        s_traj.append(algo.S.astype(np.float16))
        sp_traj.append(np.linalg.norm(algo.pos - algo.pos.mean(axis=0), axis=1).mean())
    err = abs(algo.g_fit - f_global)
    sw = [(int(t), m.replace("\u2192", "_to_")) for t, m in getattr(algo, "switch_log", [])]
    return (name, fid, dim, run_id, err, sw, np.stack(s_traj), np.asarray(sp_traj, np.float16))


def _load():
    store = {}
    if os.path.exists(OUT):
        with np.load(OUT, allow_pickle=True) as z:
            store = {k: z[k] for k in z.files}
    return store


def _save(store):
    tmp = OUT + ".tmp.npz"  # 必须以 .npz 结尾：np.savez 对非 .npz 名会自动追加后缀
    np.savez(tmp, **store)
    os.replace(tmp, OUT)


def _key(prefix, name, f, d, r):
    return "%s_%s_F%d_D%d_s%d" % (prefix, name, f, d, r)


if __name__ == "__main__":
    store = _load()
    tasks = [(a, f, d, s) for a in ARMS for f in FUNCS for d in (10, 20) for s in SEEDS if
             _key("ER", a, f, d, s) not in store]
    print("tasks:", len(tasks), flush=True)
    t0 = time.perf_counter()
    with mp.Pool(12) as pool:
        for i, (name, f, d, r, err, sw, sj, sp) in enumerate(pool.imap_unordered(run, tasks), 1):
            k = lambda pfx: _key(pfx, name, f, d, r)
            store[k("ER")] = np.float64(err)
            store[k("SW")] = np.array(sw, dtype=object)
            store[k("SJ")] = sj
            store[k("SP")] = sp
            if i % 10 == 0 or i == len(tasks):
                _save(store)
                el = time.perf_counter() - t0
                eta = el / i * (len(tasks) - i)
                print("[p8] %d/%d (%.0f%%) | %dm%02ds | ETA %dm%02ds" % (
                i, len(tasks), 100.0 * i / len(tasks), el // 60, el % 60, eta // 60, eta % 60), flush=True)
    _save(store)
    print("[p8] done ->", OUT, flush=True)
