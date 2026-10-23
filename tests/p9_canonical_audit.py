#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p9_canonical_audit.py — 门控正典 NeRSO 插桩审计（一键，增量落盘，约3–5分钟）

动机（Fig.8 数据命脉）：现有 p8_audit.npz / pilot_v4.npz 的 SW/SJ/SP 键全部属于
NeRSO-fullEsc（全时逃逸前身版）与 NeRSO-clock 两臂。fullEsc 臂 59.9% 的点火事件
发生在相切换之后——Fig.8(a)(c)(e) 的 LIF 锯齿、驻留分布与 Fano 因子均受 H 相事件
主导，与门控正典（H 相只积累不复位）的机制面相悖。本脚本对正典 NeRSO 全量补跑，
Fig.8(a)(c)(d)(e) 与 Table IX 前三行应全部改由本批数据出图。

设计：与 p8_audit.py 同构（同协议、同种子口径、同键名规范），仅换臂名与函数覆盖。
产出 data/p9_canonical.npz（键全部 ASCII）：
  SW_NeRSO_F<f>_D<d>_s<r>  object array [(gen, "L_to_H"), ...]  切换表（含删失=空数组）
  SJ_NeRSO_F<f>_D<d>_s<r>  (gens, 50) float16  SFA 积累器 S 逐代轨迹（复位后存储）
  SP_NeRSO_F<f>_D<d>_s<r>  (gens,)   float16  群体散布逐代轨迹
  ER_NeRSO_F<f>_D<d>_s<r>  scalar            终误差 |g_fit − f_global|
运行完毕自动打印正典口径的四项统计（驻留中位、Fano、踢出/散布下界、地板占比）
与切换时刻分布（min/max、时钟等价性已由 p8 批承担，此处不再跑时钟臂）。
用法：PyCharm 绿色直跑；中断重跑自动续（键级跳过已完成 run）。
"""
import os, sys, time, datetime, subprocess, multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.common import FECounter, build_problem, seed_for, make, BUDGET

ARM = "NeRSO"  # 门控正典（common.py: _v4_loo_factory(esc_l_gate=True)）
FUNCS = tuple(range(1, 13))  # 全 12 函数 × 2 维 × 3 seeds = 72 runs
SEEDS = (0, 1, 2)  # 插桩口径同 TEVC E4：3 seeds 全基准
KAPPA, RNGW = 0.02, 200.0  # 散布地板系数与搜索域宽度（Table IX 口径）


def _repo_root(start):
    """从脚本所在位置向上找仓库根（含 core/ 与 experiments/ 的目录），与脚本放置位置无关。"""
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, "core")) and os.path.isdir(os.path.join(d, "experiments")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.dirname(os.path.abspath(__file__))
        d = parent


OUT = os.path.join(_repo_root(os.path.dirname(os.path.abspath(__file__))), "experiments", "benchmark_cec2022", "data",
                   "p9_canonical.npz")


def run(arg):
    """单 run：正式协议（全预算、seed_for 配对、FECounter 计费），
    逐步记录 S 轨迹与散布轨迹，终了取 switch_log。"""
    fid, dim, run_id = arg
    np.random.seed(seed_for(run_id, fid, dim))
    func_raw, bounds, f_global = build_problem(fid, dim, smoke=False)
    cnt = FECounter(func_raw)
    algo = make(ARM, D=dim, N=50, bounds=bounds, func=cnt)
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
    return (fid, dim, run_id, err, sw, np.stack(s_traj), np.asarray(sp_traj, np.float16))


def _load():
    if os.path.exists(OUT):
        with np.load(OUT, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    return {}


def _save(store):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)  # 输出目录不存在时自动创建
    tmp = OUT + ".tmp.npz"  # 必须以 .npz 结尾：np.savez 对非 .npz 名会自动追加后缀
    np.savez(tmp, **store)
    os.replace(tmp, OUT)


def _key(prefix, f, d, r):
    return "%s_%s_F%d_D%d_s%d" % (prefix, ARM, f, d, r)


def _manifest():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=_repo_root(os.path.dirname(os.path.abspath(__file__))), capture_output=True,
                                text=True).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    return np.array({"created": datetime.datetime.now().isoformat(timespec="seconds"), "commit": commit, "algo": ARM,
        "funcs": FUNCS, "dims": (10, 20), "seeds": SEEDS, "budget_fe": dict(BUDGET), "N": 50,
        "purpose": "canonical gated NeRSO instrumented audit for Fig.8(a)(c)(d)(e) + Table IX "
                   "(replaces fullEsc-arm p8/pilot data for those panels)", }, dtype=object)


def _sw_items(arr):
    """兼容三种历史存储格式：0-d 单值、1-d 元组/字符串数组、(n,2) 二维数组。"""
    arr = np.asarray(arr, dtype=object)
    if arr.size == 0:
        return []
    if arr.ndim == 2:
        return [(arr[i, 0], arr[i, 1]) for i in range(arr.shape[0])]
    out = []
    for it in arr.ravel():
        if isinstance(it, (tuple, list)) and len(it) == 2:
            out.append((it[0], it[1]))
            continue
        s = str(it).strip()
        if ":" in s:
            t, m = s.split(":", 1)
        elif "," in s:
            t, m = s.split(",", 1)
        else:
            continue
        out.append((t, m))
    return out


def summarize(store):
    """正典口径四项统计 + 切换分布（SJ 为复位后存储 → 归零行检测点火）。"""
    alpha_S = 1.0 / 95.8
    dwells, fanos_L, ratios, ratios_fl = [], [], [], []
    floor_d, floor_nd, tot = 0, 0, 0
    sw_all = []
    censored = []
    for k in store:
        if not k.startswith("SJ_"):
            continue
        S = store[k].astype(np.float64)
        SP = store["SP_" + k[3:]].astype(np.float64)
        T = len(SP)
        dim = int(k.split("_D")[1].split("_")[0])
        ml = BUDGET[dim] // 50
        fire = (S == 0) & (np.vstack([np.ones((1, S.shape[1])), S[:-1]]) > 0)
        swkey = "SW_" + k[3:]
        tsw = None
        if swkey in store:
            for t, m in _sw_items(store[swkey]):
                if "L_to_H" in str(m):
                    tsw = int(float(str(t).strip().strip("()'\" ")))
                    break
        if tsw is None:
            censored.append(k)
        else:
            sw_all.append(tsw / ml)
        for j in range(S.shape[1]):
            idx = np.where(fire[:, j])[0]
            dwells.extend(np.diff(idx).tolist())
        cnt = fire.sum(axis=1)
        W = 96  # 5P ≈ 96 代计数窗
        lim = tsw if tsw is not None else T
        for i in range(0, min(lim, T) - W + 1, 24):  # L 相窗口（终点在切换前）
            c = cnt[i:i + W]
            if c.mean() > 0:
                fanos_L.append(c.var() / c.mean())
        decay = 1.0 - np.arange(T) / ml
        ti = np.where(fire)
        for t, j in zip(*ti):
            if t == 0:
                continue
            Sfire = S[t - 1, j] + alpha_S
            Lenv = max(SP[t], KAPPA * decay[t] * RNGW)
            r = Sfire * Lenv / max(SP[t], 1e-300)
            ratios.append(r)
            if SP[t] <= KAPPA * decay[t] * RNGW:
                ratios_fl.append(r)
            if SP[t] <= KAPPA * RNGW:  # 无 decay 口径（Table IX 脚注）
                pass
        floor_d += int(np.sum(SP <= KAPPA * decay * RNGW))  # 代码口径（含 decay）
        floor_nd += int(np.sum(SP <= KAPPA * RNGW))  # 脚注口径（无 decay）
        tot += T
    dwells = np.array(dwells);
    ratios = np.array(ratios);
    sw_all = np.array(sw_all)
    print("== 正典 NeRSO 插桩统计 ==")
    print("驻留: n=%d 中位=%.1f (导出 95.8)" % (len(dwells), np.median(dwells)))
    print("Fano(L相窗, W=96, stride=24): n=%d 中位=%.2f" % (len(fanos_L), np.median(fanos_L)))
    print("踢出/散布: n=%d 下界=%.3f 中位=%.2f; 地板区间(含decay)内中位=%.2f n=%d" % (
    len(ratios), ratios.min() if len(ratios) else np.nan, np.median(ratios) if len(ratios) else np.nan,
    np.median(ratios_fl) if len(ratios_fl) else np.nan, len(ratios_fl)))
    print("地板占比: 含decay(代码)=%.1f%%  无decay(现脚注)=%.1f%%" % (floor_d / tot * 100, floor_nd / tot * 100))
    if len(sw_all):
        print("切换时刻: n=%d min=%.2f%% max=%.2f%%  删失=%d" % (
        len(sw_all), sw_all.min() * 100, sw_all.max() * 100, len(censored)))
        print("删失键:", censored)


if __name__ == "__main__":
    store = _load()
    tasks = [(f, d, s) for f in FUNCS for d in (10, 20) for s in SEEDS if _key("ER", f, d, s) not in store]
    print("tasks:", len(tasks), flush=True)
    t0 = time.perf_counter()
    with mp.Pool(12) as pool:
        for i, (f, d, r, err, sw, sj, sp) in enumerate(pool.imap_unordered(run, tasks), 1):
            store[_key("ER", f, d, r)] = np.float64(err)
            sw_arr = np.empty(len(sw), dtype=object)  # 强制一维：元素为 (gen, tag) 元组
            for _i, _x in enumerate(sw):
                sw_arr[_i] = _x
            store[_key("SW", f, d, r)] = sw_arr
            store[_key("SJ", f, d, r)] = sj
            store[_key("SP", f, d, r)] = sp
            if i % 10 == 0 or i == len(tasks):
                _save(store)
                el = time.perf_counter() - t0
                print("[p9] %d/%d (%.0f%%) | %dm%02ds" % (i, len(tasks), 100.0 * i / len(tasks), el // 60, el % 60),
                      flush=True)
    store["MANIFEST"] = _manifest()
    _save(store)
    print("[p9] done ->", OUT)
    summarize(store)
