# EIO 结果总表（RESULTS_SUMMARY.md，2026-10-14）

## A. 排位（CEC2022，19 算法，Friedman 平均秩）
D10：SHADE 2.21 > LSHADE 2.71 > NL-SHADE-RSP-MID 3.00 > DE 3.13 > AD-PSO 5.38
    > **EIO 8.75** > NPDOA 8.92 > EOL-v10 10.08 > AD-SCA 11.67 > PSO 11.83
D20：同序，**EIO 8.00**（维度越高越靠前）
v3 效率：D10 12.4s 全场最快 / D20 31.6s（SHADE 34.3 / NPDOA 69.5 / DE 91.5）

## B. v3 逐对 Wilcoxon（24 格，胜/平/负）
v10 17/7/0 | PSO 11/10/3 | AD-SCA 10/5/9 | NPDOA 6/9/9 | AD-PSO 1/2/21 | SHADE 0/0/24
实用等价口径（ε=1e-8 带内平局）：对 NPDOA 5/14/5，对 v10 13/8/3，无胜格变负。

## C. LOO 定价（Δlog10 中位，详见 loo_pricing.csv）
noFastEI +0.87 (24/0/0, 最差 F1D10 +5.8) | noEscape +0.02 (9/9/6, F9D10 +3.4)
| noSlow 0.00 (0/24/0, F9D10 +3.1) | noHomeo +0.03 (7/17/0) | noEInoiseShape +0.01
| noFloor 0.00 | noRefine 0.00 | noEInoiseAmp -0.00 (2/20/2)
叙事：内核全天候承重；逃逸/慢包络=组合地貌条件承重；地板/精炼/噪声三属性=实测惰性。
★ 归因：+0.87 归功"神经噪声通道存在"（noEInoiseAmp 裁决），E-I 统计形式不承重。

## D. 实用口径（practical_significance.py 预注册版）
F1：v3 248k FE(80%) vs NPDOA 2917k(7%) → 墙钟约 30×（必须与成功率成对引用）
F3：NPDOA 1 万 FE 更快（v3 19 万，双 100% 成功）
解决面：1e-2 档 v3 三格 vs NPDOA 四格；1e-8 档 v3 0 格 vs DE 系 8-9 格（精度深度=内核层短板）
F9：NPDOA 3% 彩票尾 vs v3 一致型 0.15 中位（互补画像：v3 赢 AD-PSO 3.4 dex 于 F9 类 rugged 格）

## E. v3.1 三臂 pilot 裁决（预注册，全部未过 → v3 定稿无改动）
A AR(1) 噪声(ρ=2/3 导出)：全格 ±0.2 dex → 噪声统计三属性惰性终局
B θ 时钟退役：F1/F3 恶化 +3.0~3.3 dex → 外生时钟承重
C K1 差分耦合(环邻对,加性)：F6 +0.4/+0.1 反劣 → 未探明域=全局对/替换式放置
（pilot_v31.npz 存原始目标值，分析减 f_global：F1=300,F3=600,F6=1800,F9=2300）

## F. 待跑可选件
- 常规基线 GA/BBO/WHO/RAO（已注册待命，一条命令）
- 插桩 144 全量（instrumented_runs.py，P-E 验证表 + 动力学六联图数据）
- C 变体续投（全局对/替换放置，480 runs）

# G. v4 时代正式结果（2026-10-14，e1_results_v4/，3600 行）

## G1. 排位（23 算法 Friedman 平均秩）
D10：DE 系 2.4-3.8 霸榜 | NeRSO-H 6.58 = AD-PSO 6.58（并列非 DE 第一档）
     | **NeRSO-fullEsc 9.00**（第 7）| NeRSO-clock 9.33 | NPDOA 11.83 = EIO 11.83 | NeRSO-L 12.42
D20：DE 2.3-4.6 | **NeRSO-H 5.75（非 DE 第一，压过 AD-PSO 6.17）**| NeRSO-fullEsc 7.08 | NeRSO-clock 8.25

## G2. NeRSO-fullEsc 逐对 Wilcoxon（24 格）
NPDOA 12/9/3（明确净胜，v3 时代是 6/9/9）| AD-PSO 4/9/11 | SHADE 1/6/17
| v3 祖先 19/5/0 | L-only 21/3/0 | H-only 1/15/8 | 时钟臂 3/19/2

## G3. ★外围贡献 head-to-head：NeRSO-H vs AD-PSO = 5/16/3
同核（二阶惯性）不同身体：我们的全神经外围净胜 TEVC 原配——外围价值终获直接证据。

## G4. 消融表（Δ中位 vs NeRSO-fullEsc）
L-only +0.14 | H-only −0.01（均值更优但 F9D10 爆 3.6e2）| 时钟 +0.00 | 祖先 v3 +0.14
切换律的定位 = 保险丝：均值与 H-only 打平（−0.01），F9D10 上救回 9 个数量级
（2.1e-7 vs 3.6e2）——"只在它消失时才被注意到的零件"。
叙事：H-only 是均值冠军（且靠外围打赢亲本），切换版是鲁棒冠军；论文双呈现。
