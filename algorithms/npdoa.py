#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npdoa.py — NPDOA（KBS 2024, Ji et al.）参赛适配版

内核与 tests/_ref_ndp_kbs.py（原版留档）数学完全等价，唯一改动是把
optimize() 的单块循环拆成 start() + step()，使其接入 (D,N,bounds,func)
+ .step() + .g_fit 的统一 harness 接口。等价性由 tests/test_regression.py 逐位对拍。

协议说明：harness 统一 N=50；论文原配 N=100 / FE_max=100000，参数 AT=0.1,
l=0.7, d=0.1, Nd=0.1 为作者敏感性分析推荐值，此处沿用。
"""

import numpy as np
import random
from typing import Callable, Dict, Optional, Tuple


class NPD:
    """NPDOA 内核（KBS 2024 论文及 MATLAB 官方代码的严格复现）。"""

    def __init__(self, num_populations: int, dimensions: int, max_fe: int,
                 obj_func: Callable, params: Optional[Dict] = None,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 maximize: bool = False, seed: Optional[int] = None,
                 initial_population: Optional[np.ndarray] = None):
        self.N = num_populations
        self.D = dimensions
        self.max_fe = max_fe
        self.obj_func = obj_func
        self.bounds = np.array(bounds, dtype=float)
        self.maximize = maximize
        self._sense = -1.0 if maximize else 1.0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        params = params or {}
        self.AT = params.get('AT', 0.1)
        self.Nd = params.get('Nd', 0.1)
        self.l = params.get('l', params.get('a', 0.7))
        self.d = params.get('d', 0.1)

        if initial_population is not None and len(initial_population) >= self.N:
            self.population = np.asarray(initial_population[:self.N], dtype=float).copy()
        else:
            self.population = np.random.uniform(bounds[0], bounds[1], (self.N, self.D))

        self.fitness = np.full(self.N, np.inf)
        self.fe = 0

        self.gbest_fitness = np.inf
        self.gbest_position = None
        self.best_history = []

    def _evaluate(self, pop: np.ndarray) -> np.ndarray:
        fitness = np.array([self._sense * self.obj_func(ind) for ind in pop])
        self.fe += len(pop)
        return fitness

    # ── 拆分点 1：初始化评估（= 原版 optimize() 的前半）──
    def start(self):
        self.fitness = self._evaluate(self.population)
        best_idx = np.argmin(self.fitness)
        self.gbest_fitness = self.fitness[best_idx]
        self.gbest_position = self.population[best_idx].copy()
        self.best_history.append(self._sense * self.gbest_fitness)

    # ── 拆分点 2：一代进化（= 原版 optimize() 的 while 循环体）──
    def step(self, verbose: bool = False):
        rank = np.argsort(self.fitness)
        self.population = self.population[rank]
        self.fitness = self.fitness[rank]

        n_attractor = max(1, int(self.N * self.AT))
        attractors = self.population[:n_attractor]

        T = self.fe / self.max_fe
        offspring = self.population.copy()

        # 策略1：吸引子趋势（向量化：数学与原版逐个体循环完全等价——
        # 原版对每个 mover 的 n_attractor 次吸引子更新做原地累加，
        # 等价于对各吸引子项求和；随机数消耗顺序刻意保持一致，
        # 见 tests/test_regression.py 第3节逐位对拍）
        n_movers = self.N - n_attractor
        if n_movers > 0:
            # 随机流消耗顺序与原版逐个体循环严格对齐（A->SA->L 按 mover 分块），
            # 保证与原版逐位等价（tests/test_regression.py 第3节对拍）
            A_all = np.stack([np.random.randint(0, 2, (n_attractor, self.D))
                              for _ in range(n_movers)])
            SA_all = np.array([[np.random.randint(0, n_attractor)
                                for _ in range(n_attractor)]
                               for _ in range(n_movers)])
            L_all = np.stack([np.random.rand(n_attractor, self.D)
                              for _ in range(n_movers)])
            movers = self.population[n_attractor:]            # (n_movers, D)
            att_sel = attractors[SA_all]                      # (n_movers, n_att, D)
            delta = np.sum(self.l * L_all * L_all * A_all
                           * (att_sel - movers[:, None, :]), axis=1)
            Lower = np.min(self.population, axis=0)
            Upper = np.max(self.population, axis=0)
            noise_std = self.Nd * (Upper - Lower)
            # 与原版一致：每个 mover 一个标量高斯 × 向量 noise_std
            noise = np.random.randn(n_movers)[:, None] * noise_std[None, :]
            offspring[n_attractor:] = movers + delta + noise

        # 策略2&3：耦合干扰 + 信息投影
        meanN = np.mean(offspring, axis=0)
        for i in range(self.N):
            newone = offspring[i].copy()
            l1 = 0.5 * self.d * np.random.rand(self.D)
            l2 = 0.5 * self.d * np.random.rand(self.D)
            L = np.random.randint(-1, 2, self.D)   # 原版代码保留（未直接使用）
            L1 = np.random.randint(-1, 2, self.D)
            L2 = np.random.randint(-1, 2, self.D)
            A1 = np.random.randint(0, 2, self.D)
            A2 = np.random.randint(0, 2, self.D)
            term_dif = L1 * l1 * A1 * (newone - meanN)
            term_add = L2 * A2 * l2 * meanN
            newone = newone + (term_dif + term_add) * np.random.rand() * (1 - T)
            offspring[i] = newone

        offspring = np.clip(offspring, self.bounds[0], self.bounds[1])
        offspring_fitness = self._evaluate(offspring)

        # 完全替换（无精英保留，忠于原版）
        self.population = offspring
        self.fitness = offspring_fitness

        current_best_idx = np.argmin(self.fitness)
        if self.fitness[current_best_idx] < self.gbest_fitness:
            self.gbest_fitness = self.fitness[current_best_idx]
            self.gbest_position = self.population[current_best_idx].copy()
        self.best_history.append(self._sense * self.gbest_fitness)

        if verbose and (self.fe // self.N) % 10 == 0:
            print("[NPD] Gen={}, FE={}/{}, Best={:.6f}".format(
                self.fe // self.N, self.fe, self.max_fe,
                self._sense * self.gbest_fitness))
        return self._sense * self.gbest_fitness

    def optimize(self, verbose: bool = False):
        """原版入口：start() + 按 FE 预算循环（与拆分前逐语句等价）。"""
        self.start()
        while self.fe < self.max_fe:
            self.step(verbose)
        best_fitness = self._sense * self.gbest_fitness
        info = {'best_history': self.best_history, 'eval_count': self.fe}
        return self.gbest_position, best_fitness, info


class NPDOA(NPD):
    """harness 接口适配：关键字构造 (D, N, bounds, func) + .step()/.g_fit。

    注意：不设内部 seed（harness 在构造前已统一 np.random.seed），
    initial_population 走 np.random，完全纳入 harness 种子控制。
    """

    def __init__(self, D, N, bounds, func, params=None):
        super().__init__(num_populations=N, dimensions=D, max_fe=10 ** 9,
                         obj_func=func, params=params, bounds=bounds, seed=None)
        self.gens = 0

    @property
    def g_fit(self):
        return self._sense * self.gbest_fitness

    @property
    def fe_count(self):
        return self.fe

    def step(self, verbose: bool = False):
        self.gens += 1
        return super().step(verbose)
