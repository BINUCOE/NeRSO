#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eio_v4.py — EIO v4：神经内核动力学区切换（Regime-Switching Neural Kernel）

主方程时变惯性质量 m(t)∈{0,1}：m=0 一阶振荡壳（L 模式，v3 核，rugged 突围），
m=1 二阶惯性核（H 模式，AD-PSO 速度积分核 verbatim：w=0.7, c1=c2=1.4，精密进给）。
差异化和先例（AdapSCA-PSO 等"SCA早PSO晚"切换器）钉在三根轴：
  ① 时钟自由：状态触发（散布轨迹）非迭代调度；② 统一 m(t) 主方程；③ P-E8 可检验。

spec 要点（Kimi 审查修正后）：
  * H 模式关闭 σ0·L_env 白噪声通道（AD-PSO 式无噪声+速度低通是深度机器的前提；
    保留噪声则深度验收 ≥3 量级构造上不可达）
  * 触发器主票=散布轨迹形态（单调坍缩+回弹否决）；逃逸率不入触发器
    （实测 F1≈0.89/代 > F9≈0.49/代，dwell 时钟驱动、无地貌判别力）
  * 外围全程 v3 配置（SFA/内稳态/环拓扑/E-I 状态连续），H 只换速度积分核
  * LOO 臂（common.py 注册，不进参数表）：h_noise=H 相重开 E-I 塑形白噪声
  * 联合稳定性无证明（两单模态各自有），论文措辞留余地
"""
import numpy as np

from algorithms.eio import EIO


class NeRSO(EIO):
    """动力学区切换：L(一阶) ↔ H(二阶)。触发器=散布轨迹+迟滞+回切+最小驻留。"""
    
    W_H, C_H = 0.7, 1.0  # H 模式：二阶惯性核（w=0.7 同 AD-PSO；c=1.0 简化取值，
    # φ=1.0 均值刚度仍在 TEVC 稳定域内——pilot 实测深度达标，
    # 不宣称 verbatim 1.4，见 ARCHIVE §35 审查修正）
    REBOUND = 1.5  # 回弹否决：近期散布从最小值回升 >50%
    
    def __init__(self, *a, clock_switch=False, r_lo=0.05, r_hi=0.10, t_min_frac=2.0, oneway=False, mode_lock=None,
                 h_noise=False, esc_l_gate=False, **kw):
        # oneway：只切不返（消极限环）；mode_lock："L"/"H"=锁定单模式（消融臂）
        super().__init__(*a, **kw)
        self.clock_switch = clock_switch  # 对照臂：True=纯时钟切换（新颖性裁决）
        self.R_LO, self.R_HI = r_lo, r_hi  # 散布比迟滞对（构造参数，v4s2=0.20/0.40）
        self.oneway = oneway
        self.h_noise = h_noise
        self.esc_l_gate = esc_l_gate  # 设计修订臂：逃逸仅 L 相开火（LOO 定价：H 相拆逃逸 F11 双维 +5 dex）  # LOO 臂 NeRSO-Hnoise：H 相重开与 L 同款
        # E-I 塑形白噪声（反事实："spec 关噪声"决策上秤）
        self.mode = mode_lock if mode_lock else "L"
        self.mode_lock = mode_lock
        self.vel = np.zeros((self.N, self.D))
        self._spread_max = 0.0
        self._ratio_hist = []
        self.mode_dwell = 0
        self.switch_log = []  # 审计：(gen, L→H / H→L)
        self.t_min = int(t_min_frac / self.alpha_S)  # t_min_frac·T_SFA，derived
    
    def _trigger(self, spread):
        if self.mode_lock:
            return None  # 锁定模式：永不切换（消融臂语义纯化）
        self._spread_max = max(self._spread_max, spread)
        r = spread / (self._spread_max + 1e-12)
        self._ratio_hist.append(r)
        W = max(10, int(round(self.tau_slow)))
        win = self._ratio_hist[-W:]
        collapse = r < self.R_LO
        rebound = (len(win) > 3 and spread > self.REBOUND * min([s * self._spread_max for s in win]))
        if self.clock_switch:  # 时钟臂：60% 预算点硬切
            frac = self.t / float(max(self.max_iter, 1))
            if self.mode == "L" and frac >= getattr(self, '_clock_frac', 0.6):
                return "L→H"
            return None
        if self.mode == "L":
            if (self.t > self.t_min and self.mode_dwell >= self.t_min and collapse and not rebound):
                return "L→H"
        elif not self.oneway:
            # 回切（双向版才有）：H 驻留 τ_slow 后代，散布破高阈或回弹才返 L。
            # 注意：H 速度自旋升会瞬时抬散布，双向切换存在自激极限环（v4s2 首诊），
            # oneway 版（只切不返）消除该环——时钟臂与 AdapSCA 先例均单向。
            if self.mode_dwell >= int(self.tau_slow) and (r > self.R_HI or rebound):
                return "H→L"
        return None
    
    def step(self):
        N, D = self.N, self.D
        self.t += 1
        self.mode_dwell += 1
        T = self.t / float(max(self.max_iter, 1))
        decay = max(0.0, 1.0 - T)
        
        lb = self._lbest()
        diff = lb - self.pos
        dist_raw = np.linalg.norm(self.pos - self.pos.mean(axis=0), axis=1)
        spread = float(dist_raw.mean())
        L_env = max(spread, self.kappa * decay * self.search_range)
        
        # 触发器（散布轨迹主票）
        sw = self._trigger(spread)
        if sw:
            self.mode = "H" if sw == "L→H" else "L"
            if self.mode == "H":
                self.vel[:] = 0.0  # v=0 起步，3.3 代弛豫
            self.mode_dwell = 0
            self.switch_log.append((self.t, sw))
        
        # 共享神经状态（E-I/慢包络/内稳态——双模式连续）
        ranks = np.empty(N)
        ranks[np.argsort(self.pb_fit)] = np.arange(N)
        fsig = 1.0 - 2.0 * ranks / max(N - 1, 1)
        prox = 1.0 / (1.0 + np.linalg.norm(diff, axis=1) / (self.search_range + 1e-12))
        ei_raw = fsig * prox + 0.5 * np.tanh(dist_raw / (spread + 1e-12) - 1.0)
        if not self.homeo_off:
            rms_now = np.sqrt((ei_raw ** 2).mean())
            self.rms_in += (1.0 / self.tau_slow) * (rms_now - self.rms_in)
        ei_in = self.u_star * ei_raw / (self.rms_in + 1e-12)
        self.E += (1.0 / self.tau_E) * (-self.E + np.tanh(ei_in - self.I))
        self.I += (1.0 / self.tau_I) * (-self.I + np.tanh(self.E))
        self.E_slow += (1.0 / self.tau_slow) * (-self.E_slow + np.tanh(ei_in - self.I_slow))
        self.I_slow += (1.0 / self.tau_slow) * (-self.I_slow + np.tanh(self.E_slow))
        d_ei = self.E - self.I
        theta = 1.0 + self.coupling * np.tanh(self.E_slow)
        amp = np.abs(d_ei) * theta
        g_mod = 1.0 + self.gamma_gain * np.tanh(2.0 * d_ei)
        
        # ── 模式特定位移 ──
        if self.mode == "L":
            a_now = self.a_max * decay
            trig = np.where(np.random.rand(N, 1) < 0.5, np.sin(2 * np.pi * np.random.rand(N, 1)),
                            np.cos(2 * np.pi * np.random.rand(N, 1)))
            r2 = np.random.rand(N, 1)
            osc = a_now * trig * np.abs(r2 * self.pos - lb)
            guide = self.c_guide * diff
            if self.noise_amp_off:
                noise = self.sigma0 * L_env * 0.2 * np.random.randn(N, D)
            else:
                noise = self.sigma0 * L_env * amp[:, None] * g_mod[:, None] * np.random.randn(N, D)
            self.pos = np.clip(self.pos + osc + guide + noise, self.lb, self.ub)
        else:
            # H：二阶惯性核（AD-PSO 配方）；白噪声通道关闭（spec 修正①）
            r1, r2 = np.random.rand(N, D), np.random.rand(N, D)
            self.vel = (self.W_H * self.vel + self.C_H * r1 * (self.pb - self.pos) + self.C_H * r2 * diff)
            self.pos = np.clip(self.pos + self.vel, self.lb, self.ub)
            if self.h_noise:  # NeRSO-Hnoise 臂：重开 σ0·L_env·amp·g_mod·ξ（无额外评估）
                self.pos = np.clip(
                    self.pos + self.sigma0 * L_env * amp[:, None] * g_mod[:, None] * np.random.randn(N, D), self.lb,
                    self.ub)
        
        fit = np.array([self.func(x) for x in self.pos])
        imp = fit < self.pb_fit * (1.0 - self.eps)
        better = fit < self.pb_fit
        self.pb[better] = self.pos[better]
        self.pb_fit[better] = fit[better]
        if not self.esc_off:
            self.S = np.where(imp, self.S * (1.0 - self.beta_S), self.S + self.alpha_S)
            esc = self.S > 1.0
            if self.esc_l_gate and self.mode == "H":   # NeRSO：H 相只积累不踢出
                esc = np.zeros(N, bool)
        else:
            esc = np.zeros(N, bool)
        if np.any(esc):
            nd = int(esc.sum())
            kick_gain = float(theta.mean()) if self.mode == "H" else 1.0
            # H 相：慢包络 θ 调制踢出幅度（神经状态在精修相的留存角色，Kimi 审查修正②）
            self.pos[esc] += (self.S[esc, None] * L_env * kick_gain * self._levy(nd, D))
            self.pos[esc] = np.clip(self.pos[esc], self.lb, self.ub)
            self.S[esc] = 0.0
            fe = np.array([self.func(x) for x in self.pos[esc]])
            b2 = fe < self.pb_fit[esc]
            if np.any(b2):
                idx = np.where(esc)[0]
                self.pb[idx[b2]] = self.pos[esc][b2]
                self.pb_fit[idx[b2]] = fe[b2]
        self.g_fit = min(self.g_fit, float(self.pb_fit.min()))
        return self.g_fit
    
    def _levy(self, n, D):
        from algorithms.eio import levy_unit
        return levy_unit(self._rng, n, D, self.levy_beta)
