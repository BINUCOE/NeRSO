#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eio_v3.py — EIO 正典实现（唯一规格源：EIO_UNIFIED_SPEC.md）
本文件与规范逐行对应；任何偏差 = bug。【待G2】标记处为唯一待定开关。
"""
import numpy as np
from math import gamma as _gamma, sin as _sin, pi as _pi
# ── 规范 §2.5 冻结辅助函数（自包含，2026-10-13 收编——消灭未冻结外部依赖）──
def _eig_period(tau_E, tau_I):
    """【冻结】离散 Euler 线性化本征周期：a=1/τ_E, b=1/τ_I；
    迹 tr=2-(a+b)，行列式 dt=(1-a)(1-b)+ab；复根时 P=2π/atan2(√(4dt-tr²),tr)，
    否则 ∞（过阻尼）。所有频段经此导出，禁改。"""
    a, b = 1.0 / tau_E, 1.0 / tau_I
    tr, dt = 2.0 - (a + b), (1.0 - a) * (1.0 - b) + a * b
    disc = tr * tr - 4.0 * dt
    if disc >= 0:
        return float("inf")
    return 2.0 * np.pi / np.arctan2(np.sqrt(-disc), tr)


def levy_unit(rng, n, D, beta=1.5):
    """【冻结】Mantegna Lévy 采样的单位化方向：尺度因子
    su=(Γ(1+β)sin(πβ/2)/(Γ((1+β)/2)·β·2^((β-1)/2)))^(1/β)；
    u=rng.standard_normal((n,D))·su；v=rng.standard_normal((n,D))；
    steps=u/(|v|^(1/β)+1e-12)；返回 steps/||steps||。禁改。"""
    su = (_gamma(1 + beta) * _sin(_pi * beta / 2) /
          (_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.standard_normal((n, D)) * su
    v = rng.standard_normal((n, D))
    steps = u / (np.abs(v) ** (1 / beta) + 1e-12)
    return steps / (np.linalg.norm(steps, axis=1, keepdims=True) + 1e-12)


class EIO:
    def __init__(self, D, N, bounds, func,
                 tau=3.0, u_star=1.0, coupling=0.3, sigma0=0.05,
                 kappa=0.02, gamma_gain=0.0,          # γ 终裁=0（§0.5 长期注记）
                 levy_beta=1.5, a_max=1.2, c_guide=0.25, eps=1e-6,
                 homeo_off=False, noise_shape_off=False,   # LOO 消融开关（不进论文参数表）
                 esc_off=False, refine_off=False, noise_amp_off=False,
                 ar1_noise=False,      # 臂A：AR(1)相关噪声，ρ=1-1/τ_E 导出
                 theta_anneal=False,   # 臂B：θ驱动核幅，时钟退役
                 diff_coupling=False):  # 臂C(K1)：侧耦合差分项
        self.D, self.N, self.bounds, self.func = D, N, bounds, func
        self.tau, self.u_star = tau, u_star
        self.coupling, self.sigma0, self.kappa = coupling, sigma0, kappa
        self.gamma_gain = gamma_gain                   # 【待G2】
        self.levy_beta, self.a_max, self.c_guide, self.eps = levy_beta, a_max, c_guide, eps
        self.homeo_off = homeo_off
        self.noise_shape_off = noise_shape_off
        self.esc_off = esc_off
        self.refine_off = refine_off
        self.noise_amp_off = noise_amp_off
        self.ar1_noise = ar1_noise
        self.theta_anneal = theta_anneal
        self.diff_coupling = diff_coupling
        self._noise_prev = None
        self._rho_ar1 = 1.0 - 1.0 / tau       # ρ=1-1/τ_E（τ_E=tau），频段导出
        self.amp_ema = None

        tau_E, tau_I = tau, tau / 0.6
        P = _eig_period(tau_E, tau_I)
        self.tau_E, self.tau_I, self.P = tau_E, tau_I, P
        self.tau_slow = 1.5 * P
        T_SFA = 5.0 * P
        self.alpha_S, self.beta_S = 1.0 / T_SFA, 0.1 / T_SFA
        self.ref_interval = max(10, int(round(4.0 * P)))
        self.K = 7
        self.n_ref = max(1, int(0.1 * N))

        self.lb, self.ub = bounds
        self.search_range = self.ub - self.lb
        self.pos = np.random.uniform(self.lb, self.ub, (N, D))
        self.pb = self.pos.copy()
        self.pb_fit = np.full(N, np.inf)
        self.E = np.zeros(N); self.I = np.zeros(N)
        self.E_slow = np.zeros(N); self.I_slow = np.zeros(N)
        self.S = np.zeros(N)
        self.rms_in = 1.0
        self.sig_ref = None
        self.t = 0
        self.max_iter = 1000
        self.g_fit = np.inf
        self._arN = np.arange(N)
        self._rng = np.random

    def _lbest(self):
        h = self.K // 2
        off = np.arange(-h, h + 1)
        idx = (self._arN[:, None] + off) % self.N
        return self.pb[idx[self._arN, np.argmin(self.pb_fit[idx], axis=1)]]

    def step(self):
        N, D = self.N, self.D
        self.t += 1
        T = self.t / float(max(self.max_iter, 1))
        decay = 1.0 - T

        lb = self._lbest()
        diff = lb - self.pos
        dist_raw = np.linalg.norm(self.pos - self.pos.mean(axis=0), axis=1)
        spread = dist_raw.mean() + 1e-12
        L_env = max(spread, self.kappa * decay * self.search_range)      # 4.2
        ranks = np.empty(N)
        ranks[np.argsort(self.pb_fit)] = np.arange(N)
        fsig = 1.0 - 2.0 * ranks / max(N - 1, 1)
        prox = 1.0 / (1.0 + np.linalg.norm(diff, axis=1) / (self.search_range + 1e-12))
        ei_raw = fsig * prox + 0.5 * np.tanh(dist_raw / (spread + 1e-12) - 1.0)  # 4.3

        if not self.homeo_off:                                           # 4.4
            rms_now = np.sqrt((ei_raw ** 2).mean())
            self.rms_in += (1.0 / self.tau_slow) * (rms_now - self.rms_in)
        ei_in = self.u_star * ei_raw / (self.rms_in + 1e-12)

        self.E += (1.0 / self.tau_E) * (-self.E + np.tanh(ei_in - self.I))
        self.I += (1.0 / self.tau_I) * (-self.I + np.tanh(self.E))
        self.E_slow += (1.0 / self.tau_slow) * (-self.E_slow + np.tanh(ei_in - self.I_slow))
        self.I_slow += (1.0 / self.tau_slow) * (-self.I_slow + np.tanh(self.E_slow))  # 4.5

        d_ei = self.E - self.I
        theta = 1.0 + self.coupling * np.tanh(self.E_slow)
        amp = np.abs(d_ei) * theta                     # 4.6 θ 恰好一次
        g_mod = 1.0 + self.gamma_gain * np.tanh(2.0 * d_ei)   # γ 终裁=0 → g_mod≡1
        if self.noise_shape_off:
            amp_now = float(np.mean(amp))
            if self.amp_ema is None:
                self.amp_ema = amp_now
            self.amp_ema += (1.0 / self.tau_slow) * (amp_now - self.amp_ema)
            amp = np.full(N, self.amp_ema)

        if self.theta_anneal:
            a_now = self.a_max * float(1.0 + self.coupling * np.tanh(self.E_slow.mean()))
        else:
            a_now = self.a_max * decay
        trig = np.where(np.random.rand(N, 1) < 0.5,
                        np.sin(2 * np.pi * np.random.rand(N, 1)),
                        np.cos(2 * np.pi * np.random.rand(N, 1)))
        r2 = np.random.rand(N, 1)
        osc = a_now * trig * np.abs(r2 * self.pos - lb)
        guide = self.c_guide * diff
        if self.ar1_noise:
            xi = np.random.randn(N, D)
            if self._noise_prev is not None:
                xi = (self._rho_ar1 * self._noise_prev
                      + np.sqrt(1 - self._rho_ar1 ** 2) * xi)
            self._noise_prev = xi
        else:
            xi = np.random.randn(N, D)
        if self.noise_amp_off:
            # 【特异性裁决臂】噪声 = σ0·L·AMP_CONST·ξ（AMP_CONST=0.2 预置常数，
            # 不依赖任何 E/I 状态；对照组：noFastEI=无噪声、完整版=E-I 驱动幅度）
            noise = self.sigma0 * L_env * 0.2 * np.random.randn(N, D)
        else:
            noise = self.sigma0 * L_env * amp[:, None] * g_mod[:, None] * xi  # 4.7
        if self.diff_coupling:
            r1 = (self._arN + np.random.randint(1, self.N, N)) % self.N
            r2 = (self._arN + np.random.randint(1, self.N, N)) % self.N
            diff = 0.5 * a_now * (self.pos[r1] - self.pos[r2])
        else:
            diff = 0.0
        self.pos = np.clip(self.pos + osc + guide + noise + diff, self.lb, self.ub)

        fit = np.array([self.func(x) for x in self.pos])                 # 4.8

        imp = fit < self.pb_fit * (1.0 - self.eps)
        better = fit < self.pb_fit
        self.pb[better] = self.pos[better]
        self.pb_fit[better] = fit[better]
        if not self.esc_off:
            # esc_off 语义纯化（Kimi 审查）：S 完全不更新（恒 0），
            # 精炼 σ 的 S̄ 读出不受污染——"移除逃逸"不再混入"精炼加倍"
            self.S = np.where(imp, self.S * (1.0 - self.beta_S), self.S + self.alpha_S)
            esc = self.S > 1.0
        else:
            esc = np.zeros(N, bool)
        if np.any(esc):
            nd = int(esc.sum())
            self.pos[esc] += (self.S[esc, None] * L_env
                              * levy_unit(self._rng, nd, D, self.levy_beta))
            self.pos[esc] = np.clip(self.pos[esc], self.lb, self.ub)
            self.S[esc] = 0.0
            fit_esc = np.array([self.func(x) for x in self.pos[esc]])
            better2 = fit_esc < self.pb_fit[esc]
            if np.any(better2):
                idx = np.where(esc)[0]
                self.pb[idx[better2]] = self.pos[esc][better2]
                self.pb_fit[idx[better2]] = fit_esc[better2]

        self.g_fit = min(self.g_fit, float(self.pb_fit.min()))

        if (not self.refine_off) and self.t % self.ref_interval == 0:    # 4.11
            top = np.argsort(self.pb_fit)[:self.n_ref]
            s_norm = float(np.clip(self.S.mean(), 0.0, 1.0))
            if self.sig_ref is None:
                self.sig_ref = self.sigma0 * L_env
            sigma = self.sig_ref * (1.0 + s_norm)
            cand = np.clip(self.pb[top] + sigma * np.random.randn(len(top), D),
                           self.lb, self.ub)
            cf = np.array([self.func(x) for x in cand])
            good = cf < self.pb_fit[top]
            if np.any(good):
                acc = np.linalg.norm(cand[good] - self.pb[top][good], axis=1)
                self.sig_ref = max(float(np.sqrt((acc ** 2).mean())), 1e-3 * L_env)
            else:
                self.sig_ref = max(0.7 * self.sig_ref, 1e-3 * L_env)
            if np.any(good):
                self.pb[top[good]] = cand[good]
                self.pb_fit[top[good]] = cf[good]
                self.g_fit = min(self.g_fit, float(self.pb_fit.min()))
        return self.g_fit
