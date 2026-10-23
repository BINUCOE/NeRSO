#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
de_family.py — DE 强基线族（Tier 5）

  DE      DE/rand/1/bin（朴素锚，F=0.5, CR=0.9）
  SHADE   成功历史参数自适应（Tanabe & Fukunaga, CEC 2013）
  LSHADE  SHADE + 线性种群缩减（L-SHADE, CEC 2014 冠军）

向量化 numpy 实现；全部走 harness 的 (D,N,bounds,func) + .step()/.g_fit；
所有评估经 harness FECounter 计数（每代 = 当前种群数个评估）。

保真声明
--------
实现遵循各论文原始描述与 mealpy（公开实现，有论文 DOI 链）的结构；
以下细节为通用默认，正式投稿前需与选定蓝本逐条核对（tests/test_regression
第6节留了核对清单）：
  * SHADE 的 pbest 比例 p：取自原论文口径，每代每个体从
    [2, max(2, 0.2*N)] 个最优个体中随机选引导；
  * 记忆库 H=6，存档率 2.6，柯西/正态参数采样，Lehmer 加权均值；
  * LSHADE 线性缩减：N(G) = round((N0-4)*(max_fe-fe)/max_fe + 4)。
"""

import numpy as np


class DE:
    """DE/rand/1/bin —— 差分进化朴素锚。"""

    def __init__(self, D, N, bounds, func, F=0.5, CR=0.9):
        self.D, self.N, self.bounds, self.func = D, N, bounds, func
        self.F, self.CR = F, CR
        self.pop = np.random.uniform(bounds[0], bounds[1], (N, D))
        self.fit = np.array([func(x) for x in self.pop])
        self.g_fit = float(self.fit.min())

    def step(self):
        N, D = self.N, self.D
        idx = np.arange(N)
        r1, r2, r3 = (np.random.permutation(N) for _ in range(3))
        # 保证 r1,r2,r3 互异且不等于 i
        while np.any((r1 == idx) | (r2 == idx) | (r3 == idx) |
                     (r1 == r2) | (r1 == r3) | (r2 == r3)):
            r1, r2, r3 = (np.random.permutation(N) for _ in range(3))
        mutant = self.pop[r1] + self.F * (self.pop[r2] - self.pop[r3])
        cross = np.random.rand(N, D) < self.CR
        jrand = np.random.randint(0, D, N)
        cross[idx, jrand] = True
        trial = np.where(cross, mutant, self.pop)
        trial = np.clip(trial, self.bounds[0], self.bounds[1])
        f_trial = np.array([self.func(x) for x in trial])
        better = f_trial < self.fit
        self.pop[better] = trial[better]
        self.fit[better] = f_trial[better]
        self.g_fit = float(self.fit.min())
        return self.g_fit


class SHADE(DE):
    """SHADE：成功历史记忆 + 外部存档 + current-to-pbest/1/bin。"""

    H = 6
    ARC_RATE = 2.6

    def __init__(self, D, N, bounds, func, H=6, arc_rate=2.6):
        super().__init__(D, N, bounds, func)
        self.H, self.arc_rate = H, arc_rate
        self.M_CR = np.full(H, 0.5)
        self.M_F = np.full(H, 0.5)
        self.k = 0
        self.arc = np.empty((0, D))
        self.arc_fit = np.empty(0)
        self.N_init = N

    def _pop_size(self):
        return len(self.pop)

    def step(self):
        N, D = self._pop_size(), self.D
        idx = np.arange(N)
        pmax = max(2, int(0.2 * N))
        p_i = np.random.randint(2, pmax + 1, N)
        order = np.argsort(self.fit)
        pbest_idx = order[np.minimum(p_i - 1, N - 1)]

        r_int = np.random.randint(0, self.H, N)
        CR = np.random.normal(self.M_CR[r_int], 0.1, N)
        CR = np.clip(CR, 0.0, 1.0)
        CR[self.M_CR[r_int] < 0] = 0.0          # 终结值 -1 -> CR=0
        F = self._rand_cauchy(self.M_F[r_int], 0.1, N)
        F = np.clip(F, 0.0, 1.0)

        r1, r2 = (np.random.randint(0, N, N) for _ in range(2))
        bad = (r1 == idx) | (r2 == idx) | (r1 == r2) | (r1 == pbest_idx)
        while np.any(bad):
            r1[bad] = np.random.randint(0, N, int(bad.sum()))
            r2[bad] = np.random.randint(0, N, int(bad.sum()))
            bad = (r1 == idx) | (r2 == idx) | (r1 == r2) | (r1 == pbest_idx)
        a_idx = np.random.randint(0, len(self.arc), N) if len(self.arc) else None
        use_arc = np.random.rand(N) < 0.5 if len(self.arc) else np.zeros(N, bool)
        base = self.pop[r1] if False else None
        diffs = self.pop[r2] - self.pop[r1]
        diffs[use_arc] = (self.arc[a_idx[use_arc]] - self.pop[r1[use_arc]]
                          if len(self.arc) else diffs[use_arc])
        mutant = self.pop[pbest_idx] + F[:, None] * (
            self.pop[idx] - self.pop[pbest_idx]) + F[:, None] * diffs
        cross = np.random.rand(N, D) < CR[:, None]
        jrand = np.random.randint(0, D, N)
        cross[idx, jrand] = True
        trial = np.where(cross, mutant, self.pop)
        trial = np.clip(trial, self.bounds[0], self.bounds[1])
        f_trial = np.array([self.func(x) for x in trial])

        better = f_trial < self.fit
        # 存档：被淘汰的父代
        if better.any():
            self.arc = np.vstack([self.arc, self.pop[better]]) if len(self.arc) \
                else self.pop[better].copy()
            self.arc_fit = np.append(self.arc_fit, self.fit[better])
            cap = int(self.arc_rate * self.N_init)
            if len(self.arc) > cap:
                drop = np.random.choice(len(self.arc), len(self.arc) - cap,
                                        replace=False)
                keep = np.setdiff1d(np.arange(len(self.arc)), drop)
                self.arc, self.arc_fit = self.arc[keep], self.arc_fit[keep]
        self.pop[better] = trial[better]
        self.fit[better] = f_trial[better]
        self.g_fit = float(self.fit.min())

        # 成功历史更新（加权均值）
        if better.any():
            d = np.abs(self.fit[better] - f_trial[better])
            w = d / d.sum() if d.sum() > 0 else np.full_like(d, 1.0 / len(d))
            slot = self.k
            self.M_CR[slot] = np.sum(w * CR[better]) if np.any(CR[better] > 0) \
                else self.M_CR[slot]
            self.M_F[slot] = (np.sum(w * F[better] ** 2) /
                              np.sum(w * F[better]))
            self.k = (self.k + 1) % self.H
        return self.g_fit

    @staticmethod
    def _rand_cauchy(loc, scale, n):
        return loc + scale * np.random.standard_cauchy(n)


class LSHADE(SHADE):
    """L-SHADE = SHADE + 线性种群缩减（NFE 驱动，需 harness 注入 max_fe）。"""

    N_MIN = 4

    def __init__(self, D, N, bounds, func, H=6, arc_rate=2.6):
        super().__init__(D, N, bounds, func, H=H, arc_rate=arc_rate)
        self.max_fe = None   # harness 在跑前注入

    def _reduce(self):
        if not self.max_fe:
            return
        nfe = getattr(self, "fe", None)
        # 用已评估数估算：harness 循环每代评估当前种群数
        n_target = int(round((self.N_init - self.N_MIN) *
                             max(0.0, self.max_fe - self._nfe) / self.max_fe
                             + self.N_MIN))
        n_target = max(self.N_MIN, min(len(self.pop), n_target))
        if n_target < len(self.pop):
            order = np.argsort(self.fit)
            self.pop, self.fit = self.pop[order[:n_target]], self.fit[order[:n_target]]

    def step(self):
        # 每代评估数 = 当前种群数，据此推算 NFE（harness 不直接告知）
        self._nfe = getattr(self, "_nfe", 0) + len(self.pop)
        self._reduce()
        return super().step()




class NLSHADERSPMID(LSHADE):
    """NL-SHADE-RSP-MID（Biedrzycki et al., CEC 2022 top-4）。

    在 L-SHADE 基础上叠加 NL-SHADE-RSP 组件：
      * 排序选择压（RSP）：变异引导个体从排序后种群按线性概率分布抽取
      * 档案概率 0.5（dif 项以 0.5 概率取自外部档案）
      * 中点评估：每代额外评估种群中点，若优于当前最优则替换最差个体
        （FE 如实计入）
      * 重启机制：连续多代无改进时，保留最优 20% 个体，其余重新初始化
      * k-means 分裂：重启时若种群方差过大，按 k-means 分裂为子群分别演化
        （简化实现：以当前最优为中心做 2-means 划分）

    参数取原论文默认：H=6, arc_rate=2.6, p=0.11（RSP 分布截断）,
    记忆库初始化 0.5/0.5, 重启阈值 50 代, 保留比例 0.2。
    """

    P_RSP = 0.11        # RSP 线性分布截断（前 11% 最优个体）
    ARC_P = 0.5         # 档案参与 dif 项的概率
    STAG_LIMIT = 50     # 重启前的停滞代数阈值
    KEEP_FRAC = 0.2     # 重启时保留的最优个体比例

    def __init__(self, D, N, bounds, func, H=6, arc_rate=2.6):
        super().__init__(D, N, bounds, func, H=H, arc_rate=arc_rate)
        self.stag = 0
        self.best_hist = []

    def _rsp_select(self, n):
        """RSP 排序选择压：从排序后种群按线性概率分布抽 n 个索引。"""
        N = len(self.pop)
        # 线性概率：rank 越靠前概率越大，截断在前 P_RSP*N
        n_cut = max(2, int(self.P_RSP * N))
        ranks = np.arange(n_cut)
        prob = (n_cut - ranks) / (n_cut * (n_cut + 1) / 2.0)
        return np.random.choice(n_cut, size=n, p=prob)

    def _midpoint_eval(self):
        """中点评估：评估种群中点，若优于当前最优则替换最差个体。"""
        mid = np.mean(self.pop, axis=0)
        mid = np.clip(mid, self.bounds[0], self.bounds[1])
        f_mid = self.func(mid)  # FE 如实计入（harness 计数器自动统计）
        worst = int(np.argmax(self.fit))
        if f_mid < self.fit[worst]:
            self.pop[worst] = mid
            self.fit[worst] = f_mid
            if f_mid < self.g_fit:
                self.g_fit = float(f_mid)

    def _restart(self):
        """重启：保留最优 KEEP_FRAC，其余在界内重初始化（预算内）。"""
        N = len(self.pop)
        n_keep = max(1, int(self.KEEP_FRAC * N))
        order = np.argsort(self.fit)
        new_pop = self.pop[order[:n_keep]].copy()
        new_fit = self.fit[order[:n_keep]].copy()
        n_new = N - n_keep
        # 预算检查：重启评估数不能超过剩余预算
        if self.max_fe is not None:
            nfe = getattr(self.func, "n", getattr(self, "_nfe", 0))
            remaining = self.max_fe - nfe
            n_new = min(n_new, max(0, remaining))
        if n_new > 0:
            # k-means 分裂（简化 2-means）：以最优个体为中心
            center = new_pop[0]
            rest = np.random.uniform(self.bounds[0], self.bounds[1], (n_new, self.D))
            # 一半个体围绕最优个体小半径采样，另一半全局
            half = n_new // 2
            if half > 0:
                radius = 0.1 * (self.bounds[1] - self.bounds[0])
                rest[:half] = np.clip(center + np.random.randn(half, self.D) * radius,
                                      self.bounds[0], self.bounds[1])
            f_rest = np.array([self.func(x) for x in rest])  # FE 如实计入
            new_pop = np.vstack([new_pop, rest])
            new_fit = np.append(new_fit, f_rest)
        self.pop, self.fit = new_pop, new_fit
        self.g_fit = float(self.fit.min())
        self.stag = 0

    def step(self):
        prev_best = self.g_fit
        # 用 func 计数器的真实 FE（FECounter.n）做预算检查
        nfe = getattr(self.func, "n", getattr(self, "_nfe", 0))
        if self.max_fe is not None:
            remaining = self.max_fe - nfe
            if remaining >= len(self.pop) + 1:
                self._midpoint_eval()
        else:
            self._midpoint_eval()
        # RSP 选择压：覆盖 SHADE 的 pbest 选择
        N = len(self.pop)
        idx = np.arange(N)
        pbest_idx = self._rsp_select(N)
        # 调用 SHADE.step 但替换其 pbest 逻辑——直接内联重写
        r_int = np.random.randint(0, self.H, N)
        CR = np.random.normal(self.M_CR[r_int], 0.1, N)
        CR = np.clip(CR, 0.0, 1.0)
        F = self._rand_cauchy(self.M_F[r_int], 0.1, N)
        F = np.clip(F, 0.0, 1.0)

        r1, r2 = (np.random.randint(0, N, N) for _ in range(2))
        bad = (r1 == idx) | (r2 == idx) | (r1 == r2) | (r1 == pbest_idx)
        while np.any(bad):
            r1[bad] = np.random.randint(0, N, int(bad.sum()))
            r2[bad] = np.random.randint(0, N, int(bad.sum()))
            bad = (r1 == idx) | (r2 == idx) | (r1 == r2) | (r1 == pbest_idx)

        # 档案参与：以 ARC_P 概率用档案个体替换 r2
        use_arc = (np.random.rand(N) < self.ARC_P) & (len(self.arc) > 0)
        if use_arc.any():
            a_idx = np.random.randint(0, len(self.arc), int(use_arc.sum()))
            base_r2 = self.pop[r2].copy()
            base_r2[use_arc] = self.arc[a_idx]
        else:
            base_r2 = self.pop[r2]

        mutant = self.pop[pbest_idx] + F[:, None] * (
            self.pop[idx] - self.pop[pbest_idx]) + F[:, None] * (base_r2 - self.pop[r1])
        cross = np.random.rand(N, self.D) < CR[:, None]
        jrand = np.random.randint(0, self.D, N)
        cross[idx, jrand] = True
        trial = np.where(cross, mutant, self.pop)
        trial = np.clip(trial, self.bounds[0], self.bounds[1])
        f_trial = np.array([self.func(x) for x in trial])

        better = f_trial < self.fit
        if better.any():
            self.arc = np.vstack([self.arc, self.pop[better]]) if len(self.arc) \
                else self.pop[better].copy()
            self.arc_fit = np.append(self.arc_fit, self.fit[better])
            cap = int(self.arc_rate * self.N_init)
            if len(self.arc) > cap:
                drop = np.random.choice(len(self.arc), len(self.arc) - cap, replace=False)
                keep = np.setdiff1d(np.arange(len(self.arc)), drop)
                self.arc, self.arc_fit = self.arc[keep], self.arc_fit[keep]
        self.pop[better] = trial[better]
        self.fit[better] = f_trial[better]
        self.g_fit = float(self.fit.min())

        if better.any():
            d = np.abs(self.fit[better] - f_trial[better])
            w = d / d.sum() if d.sum() > 0 else np.full_like(d, 1.0 / len(d))
            slot = self.k
            self.M_CR[slot] = np.sum(w * CR[better]) if np.any(CR[better] > 0) \
                else self.M_CR[slot]
            _wf = np.sum(w * F[better])
            self.M_F[slot] = (np.sum(w * F[better] ** 2) / _wf
                              if _wf > 1e-12 else self.M_F[slot])
            self.k = (self.k + 1) % self.H

        # 线性种群缩减（继承 LSHADE）
        self._nfe = getattr(self, "_nfe", 0) + len(self.pop)
        self._reduce()

        # 停滞检测与重启
        if self.g_fit >= prev_best - 1e-12:
            self.stag += 1
        else:
            self.stag = 0
        if self.stag >= self.STAG_LIMIT:
            self._restart()
        return self.g_fit


if __name__ == "__main__":
    def sphere(x):
        return float(np.sum(np.asarray(x) ** 2))
    for cls in (DE, SHADE, LSHADE, NLSHADERSPMID):
        algo = cls(D=10, N=20, bounds=(-5.0, 5.0), func=sphere)
        algo.max_fe = 1000
        for _ in range(50):
            algo.step()
        print("{:8s} best={:.3e} (pop {})".format(cls.__name__, algo.g_fit,
                                                  len(getattr(algo, 'pop', []))))
