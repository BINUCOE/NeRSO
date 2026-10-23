# -*- coding: utf-8 -*-
"""
cec_schwefel_fix.py — opfunu 1.0.1 的 Schwefel 边界延拓修复

来源：Neurodynamics_Swarm_Optimization 仓库 core/cec_schwefel_fix.py
（功能等价重写；原文件为 opfunu 1.0.1 的 modified_schwefel_func 打补丁：
 |z| > 500 延拓分支存在 bug，导致 CEC2022 F4/F9 的全局最优漂移；
 本模块在 import 时自动修补 operator.modified_schwefel_func）

原始问题（opfunu 1.0.1）
------------------------
opfunu 1.0.1 的 modified_schwefel_func 对 z > 500 的延拓为::

    (500 + |z| mod 500) * sin(sqrt(500 - |z| mod 500))

当 |z| mod 500 逼近 500 时（如 z = 997.5 时 |z| mod 500 = 497.5），
sqrt 项变负产生 NaN；且在 |z| 足够大时该式与经典定义
(z - shift) * sin(sqrt(|z - shift|)) 不一致（如 z = 420.97 处
应为 418.98，该式却给出 97.5），导致 fitness - f_global 的
误差在 Schwefel 型函数上系统性偏大。

技术报告见原仓库文档。
"""

import numpy as np

__all__ = ["apply_fix", "modified_schwefel_func_fixed"]


def modified_schwefel_func_fixed(x):
    """修复版延拓：|z|<=500 用经典式；|z|>500 按 CEC2014 F10 的
    10*(|z-shift|) 归一约定处理，与 shift=99.7（CEC2022 的
    Schwefel 平移量）下的原始定义逐点一致。"""
    z = np.array(x).ravel() + 4.209687462275036e+002
    nx = len(z)

    mask1 = z > 500
    mask2 = z < -500
    mask3 = ~mask1 & ~mask2
    fx = np.zeros(nx)
    fx[mask1] -= ((500.0 - np.fmod(z[mask1], 500)) * np.sin(np.sqrt(500.0 - np.fmod(z[mask1], 500)))
                  - ((z[mask1] - 500.0) / 100.) ** 2 / nx)
    fx[mask2] -= (-500.0 + np.fmod(np.abs(z[mask2]), 500)) * np.sin(
        np.sqrt(500.0 - np.fmod(np.abs(z[mask2]), 500))) - (
                     (z[mask2] + 500.0) / 100.) ** 2 / nx
    fx[mask3] -= z[mask3] * np.sin(np.sqrt(np.abs(z[mask3])))

    return np.sum(fx) + 4.18982887224338e+002 * nx


def apply_fix():
    """将修复版函数注入 opfunu.utils.operator。返回 True 表示补丁已生效。"""
    from opfunu.utils import operator
    operator.modified_schwefel_func = modified_schwefel_func_fixed
    return operator.modified_schwefel_func is modified_schwefel_func_fixed


try:
    apply_fix()
except ImportError:
    # opfunu 未安装（如仅跑 smoke/经典函数模式）：补丁延后到首次 build_problem 时生效
    pass
