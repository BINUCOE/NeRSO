#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common.py — EIO 工程公共层
  1) 协议常量（与 AD 论文 E1 完全一致，全工程唯一出处）
  2) 统一 FE 计数器（公平性唯一口径）
  3) 算法注册表（惰性导入，算法文件零侵入）
  4) 问题构造（正式 = opfunu 1.0.1 + Schwefel 补丁；smoke = 经典函数替代）
"""

import importlib
import math as _math

import numpy as np

# numpy>=2 移除了 np.math；AD 谱系代码（pso_family/sca_family）依赖它
if not hasattr(np, "math"):
    np.math = _math

# ─────────────────── 协议常量 ───────────────────
N_POP = 50
FUNCS_ALL = tuple(range(1, 13))
DIMS_ALL = (10, 20)
BUDGET = {10: 200_000, 20: 500_000}
RUNS_DEFAULT = 30
SEED_BASE = 666
N_GRID = 121  # FE 采样网格点数（曲线用）


def seed_for(run_id, func_id, dim):
    """E1 原公式：seed = 666 + run_id + func_id*100 + dim*1000"""
    return SEED_BASE + run_id + func_id * 100 + dim * 1000


def fe_grid(budget, n_pop=N_POP):
    g = np.unique(np.round(np.logspace(np.log10(n_pop), np.log10(budget), N_GRID)).astype(int))
    g[-1] = budget
    return g


# ─────────────────── 统一 FE 计数 ───────────────────
class FECounter:
    """包装目标函数，精确统计真实评估次数。一切算法、一切内部调用
    （精英精炼、档案评估……）都必须经由被包装的 func 才能被计入。"""
    
    def __init__(self, func):
        self.func = func
        self.n = 0
    
    def __call__(self, x):
        self.n += 1
        return float(self.func(x))


# ─────────────────── 算法注册表 ───────────────────
# name → "模块路径:类名"。算法文件保持原样零改动，在此登记即可参赛。
def _sb(preset):
    """AD 论文开关板旗舰装配工厂（vendor 自 Neurodynamics_Swarm_Optimization）。"""
    
    def factory(D, N, bounds, func):
        from core.ad_switchboard import make as ad_make
        return ad_make(preset, D=D, N=N, bounds=bounds, func=func)
    
    return factory


def _v3_loo_factory(**over):
    """LOO 消融工厂：over 为构造参数极端值/开关，其余用正典默认。"""
    
    def factory(D, N, bounds, func):
        from algorithms.eio import EIO
        return EIO(D=D, N=N, bounds=bounds, func=func, **over)
    
    return factory


def _v4_clock_factory():
    """对照臂：v4 的纯时钟版切换（新颖性裁决——若状态门控≈时钟调度则新颖性归零）。"""
    
    def factory(D, N, bounds, func):
        from algorithms.nerso import NeRSO
        return NeRSO(D=D, N=N, bounds=bounds, func=func, clock_switch=True)
    
    return factory


def _v4_s2_factory():
    """触发器 v2（审计诊断=切换过晚+抖动）：R_LO 0.05→0.20、迟滞对 0.20/0.40、
    最小驻留 2·T_SFA→T_SFA。预注册迭代（ARCHIVE §34 补），只此一次。"""
    
    def factory(D, N, bounds, func):
        from algorithms.nerso import NeRSO
        return NeRSO(D=D, N=N, bounds=bounds, func=func, r_lo=0.20, r_hi=0.40, t_min_frac=1.0, oneway=True)
    
    return factory


def _v4_clock_at_factory(frac):
    """时钟敏感性臂：固定切点 frac（0.4/0.8），对照 0.6。"""
    
    def factory(D, N, bounds, func):
        from algorithms.nerso import NeRSO
        a = NeRSO(D=D, N=N, bounds=bounds, func=func, clock_switch=True)
        a._clock_frac = frac
        return a
    
    return factory


def _v4_loo_factory(**over):
    """v4 零件级 LOO 工厂：触发器用 v4s2 正典参数，over 为消融开关。
    三臂（TECHNICAL §2 决策树，各 720 runs）：
      NeRSO-noEsc    esc_off=True      逃逸通道在切换核里还承重吗（预期复制 v3 F9 结论）
      NeRSO-Hnoise      h_noise=True      "H 相关噪声"spec 决策上秤（预期有害或中性）
      NeRSO-H-noEsc mode_lock=H+esc_off  精确分离 F9 灾难=核惯性还是逃逸缺失"""
    base = dict(r_lo=0.20, r_hi=0.40, t_min_frac=1.0, oneway=True)
    base.update(over)
    
    def factory(D, N, bounds, func):
        from algorithms.nerso import NeRSO
        return NeRSO(D=D, N=N, bounds=bounds, func=func, **base)
    
    return factory


def _v4_lock_factory(mode):
    """v4 消融臂：锁定单模式（L-only=一阶祖先型，H-only=纯二阶精修型）。"""
    
    def factory(D, N, bounds, func):
        from algorithms.nerso import NeRSO
        return NeRSO(D=D, N=N, bounds=bounds, func=func, mode_lock=mode)
    
    return factory


_ALGO_SOURCES = {  # Tier 3 经典基线（= E1 同一装配，逐位口径）
    "PSO": _sb("PSO"), "wDecay": "algorithms.pso_family:PSO_wDecay", "SCA": _sb("SCA"),  # Tier 2 自家前作（AD 论文旗舰终版）
    "AD-PSO": _sb("v1.8"), "AD-SCA": _sb("ADSCAv2.1"),  # Tier 1 前身旗舰（TEVC 正典）
    "EIO": "algorithms.eio:EIO", "NPDOA": "algorithms.npdoa:NPDOA", "NeRSO-clock": _v4_clock_factory(), "NeRSO-fullEsc": _v4_s2_factory(),
    "NeRSO-L": _v4_lock_factory("L"),
    "NeRSO-H": _v4_lock_factory("H"),  # v4 零件级 LOO（三臂，各 720 runs，见工厂 docstring）
    "NeRSO-noEsc": _v4_loo_factory(esc_off=True), "NeRSO-Hnoise": _v4_loo_factory(h_noise=True),
    "NeRSO-H-noEsc": _v4_loo_factory(mode_lock="H", esc_off=True),
    # 设计修订臂：逃逸仅 L 相开火（预注册预测见 LOO_RESULTS_2026-10-15.md §四）
    "NeRSO":    _v4_loo_factory(esc_l_gate=True), "EIO-noFastEI": _v3_loo_factory(tau=1e6),
    "EIO-noSlow": _v3_loo_factory(coupling=0.0), "EIO-noFloor": _v3_loo_factory(kappa=0.0),
    "EIO-noHomeo": _v3_loo_factory(homeo_off=True), "EIO-noEInoiseShape": _v3_loo_factory(noise_shape_off=True),
    "EIO-noEscape": _v3_loo_factory(esc_off=True), "EIO-noRefine": _v3_loo_factory(refine_off=True),
    "EIO-noEInoiseAmp": _v3_loo_factory(noise_amp_off=True),
    
# GA/BBO/WHO/RAO 已收编 archive/classic_ext.py（待命支线，三司会审终裁 #2）；请回时按路径恢复
    "DE": "algorithms.de_family:DE", "SHADE": "algorithms.de_family:SHADE", "LSHADE": "algorithms.de_family:LSHADE",
    "NLSHADERSPMID": "algorithms.de_family:NLSHADERSPMID", }

_CACHE = {}


def _resolve(name):
    if name not in _ALGO_SOURCES:
        raise KeyError("未注册算法: {}（现有: {}）".format(name, sorted(_ALGO_SOURCES)))
    if name not in _CACHE:
        target = _ALGO_SOURCES[name]
        if callable(target):
            _CACHE[name] = target
        else:
            mod_path, cls_name = target.split(":")
            _CACHE[name] = getattr(importlib.import_module(mod_path), cls_name)
    return _CACHE[name]


def make(name, D, N, bounds, func):
    """统一构造接口：所有算法必须接受 (D, N, bounds, func) 关键字参数，
    并暴露 .step()（推进一代，返回当前 g_fit）与 .g_fit 属性。"""
    return _resolve(name)(D=D, N=N, bounds=bounds, func=func)


def algo_names():
    return sorted(_ALGO_SOURCES)


# ─────────────────── 问题构造 ───────────────────
def _smoke_standins(func_id):
    """冒烟专用：无 opfunu 环境下的代表性经典函数替代。
    覆盖三类代表性状：F1 单模态、F6/F9 多模态、F4 病态山谷、F11 可分离组合。"""
    x_ = lambda x: np.asarray(x, dtype=float)
    table = {1: ("Sphere", lambda x: float(np.sum(x_(x) ** 2)), (-100.0, 100.0)), 4: (
        "Rosenbrock", lambda x: float(np.sum(100.0 * (x_(x)[1:] - x_(x)[:-1] ** 2) ** 2 + (1.0 - x_(x)[:-1]) ** 2)),
        (-30.0, 30.0)), 6: ("Ackley", lambda x: float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x_(x) ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * x_(x)))) + 20.0 + np.e),
                            (-32.0, 32.0)), 9: (
        "Rastrigin", lambda x: float(10.0 * len(x) + np.sum(x_(x) ** 2 - 10.0 * np.cos(2 * np.pi * x_(x)))),
        (-5.12, 5.12)), 11: ("Griewank", lambda x: float(
        np.sum(x_(x) ** 2) / 4000.0 - np.prod(np.cos(x_(x) / np.sqrt(np.arange(1, len(x) + 1)))) + 1.0),
                             (-600.0, 600.0)), }
    if func_id not in table:
        raise KeyError("smoke 模式只提供代表性函数 {}，收到 F{}".format(sorted(table), func_id))
    return table[func_id]


def build_problem(func_id, dim, smoke=False):
    """返回 (func, bounds, f_global)。正式口径 = opfunu 1.0.1 + Schwefel 补丁。"""
    if smoke:
        _, func, bounds = _smoke_standins(func_id)
        return func, bounds, 0.0
    import core.cec_schwefel_fix  # noqa: F401  应用 Schwefel 延拓修复（F401 等）
    from opfunu.cec_based import cec2022
    f = getattr(cec2022, "F{:d}2022".format(func_id))(ndim=dim)
    bounds = (float(f.lb[0]), float(f.ub[0]))
    return (lambda x: float(f.evaluate(x))), bounds, float(f.f_global)
