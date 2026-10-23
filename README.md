# NeRSO: Neural Regime-Switching Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CEC2022](https://img.shields.io/badge/benchmark-CEC2022-green.svg)](experiments/benchmark_cec2022)

**NeRSO lets the motion kernel of a metaheuristic optimizer *grow* a nervous system instead of grafting one onto it.**
In a unified master equation, the inertial mass *m* becomes a state variable (not a design constant), so that first-order oscillatory exploration and second-order inertial refinement become two switchable phases of one neural kernel. All governance time scales are derived from eigenvalue analysis of the kernel's own excitatory–inhibitory (E-I) population dynamics, and the switching timing is decided endogenously by the search's own dispersion — not by a preset schedule.

> The first kernel-level instance of neurodynamics-inspired metaheuristic optimization. Predecessor: [AD-PSO & AD-SCA, *One Nervous System, Two Bodies* (IEEE TEVC, 2026)] — the "one system, two bodies" controlled pair this work folds into "one kernel, two phases".

---

## Highlights

- **Kernel growth, not transplantation.** Neural population states constitute the kernel's temporal organization, stochastic amplitudes, phase gating, and dynamical order — the neural content lives *inside* the equation.
- **Endogenous regime switching.** A four-element trigger (minimum dwell, relative collapse, rebound veto, one-way hold) reads the population dispersion; switching times differentiate continuously with the landscape (**1.3%–95.5%** of budget) against a fixed 60% clock — a falsifiable prediction, registered before the experiments.
- **Self-derived time scales.** Every governance time scale is a multiple of the kernel's characteristic period *P* ≈ 19.2 generations; measured SFA residence (median **96.0** generations) matches the derived **95.8** over ~104,000 firing events.
- **Phase-gated escape.** Slow-adaptation (LIF/SFA) kicks are confined to the exploration phase; ablation pricing shows the escape channel is load-bearing only on rugged landscapes (+14.9 dex on F9-D10) and a pure burden in refinement.
- **Testable by construction.** Every ablation variant is a constructor switch of this single implementation; four quantitative predictions were verified in instrumented runs (7/7 quantities reported in the paper).

## Results at a glance (CEC2022, 12 functions × 2 dimensions)

| Comparison | NeRSO | Reference |
|---|---|---|
| Accuracy coverage @ 10⁻⁸ median error | **7 cells** | NPDOA: 2 cells (**3.5×**); AD-PSO: 5; AD-SCA: 6 |
| Wall-clock at equal accuracy | **9.4 s** | NPDOA: 47.6 s (**20%**); AD-SCA: 17.7 s; AD-PSO: 6.0 s |
| F9-D10 (rugged cell) median error | **4.5 × 10⁻¹³** | NPDOA: 4.1 × 10²; AD-PSO: 3.6 × 10²; AD-SCA: 2.1 × 10⁻⁷ |
| Run-level pass rate on F9-D10 | **18/30** | LSHADE: 29/30; all other non-DE competitors: 0/30 |
| Switching-time distribution | **1.3%–95.5%** of budget, landscape-clustered | fixed-schedule control: 60% in 48/48 runs |

Full benchmark tables (average ranks, Wilcoxon tallies, ERT/wall-clock Pareto across three accuracy tiers) are in the paper; raw experiment outputs are under [`experiments/benchmark_cec2022/`](experiments/benchmark_cec2022).

## Installation

```bash
git clone https://github.com/BINUCOE/NeRSO.git
cd NeRSO
pip install numpy opfunu==1.0.1 pandas matplotlib scipy
```

Only `numpy` is required to run the core algorithm; `opfunu` provides the CEC2022 functions, and the remaining packages are used by the experiment/analysis scripts.

## Quick start

A minimal run (30 lines of your own code, no CEC2022 needed):

```python
import numpy as np
from algorithms.nerso import NeRSO

D, N = 10, 50                       # dimension, population size
budget = 200_000                   # function evaluations (D=10 protocol)
lb, ub = np.full(D, -100.0), np.full(D, 100.0)
f = lambda x: float(np.sum(x**2))   # any callable: R^D -> float (minimization)

opt = NeRSO(D, N, (lb, ub), f, oneway=True, esc_l_gate=True)   # canonical NeRSO
opt.max_iter = budget // N
for _ in range(opt.max_iter):
    opt.step()
print("best fitness:", opt.g_fit)
```

Constructor flags for the ablation variants (all share this one implementation):

| Variant | Constructor flags |
|---|---|
| **NeRSO** (canonical) | `NeRSO(..., oneway=True, esc_l_gate=True)` |
| NeRSO-L (L locked) | `mode_lock="L"` |
| NeRSO-H (H locked) | `mode_lock="H"` |
| NeRSO-clock (fixed schedule) | `clock_switch=True` (switches hard at 60% of budget) |
| NeRSO-noEsc (escape off) | `esc_off=True` |
| NeRSO-Hnoise (noise reopened in H) | `h_noise=True` |
| NeRSO-H-noEsc (attribution variant) | `mode_lock="H", esc_off=True` |
| NeRSO-fullEsc (escape at all times) | `esc_l_gate=False` |

## Reproducing the paper

```bash
# Formal benchmark: 24 cells (12 functions × 2 dims) × 30 runs, paired seeds
python experiments/benchmark_cec2022/run/run_formal_cec2022.py

# Smoke mode (classic test functions instead of opfunu — no install needed):
python experiments/benchmark_cec2022/run/run_formal_cec2022.py --smoke

# Instrumented runs (internal-state logging for the four quantitative predictions):
python experiments/benchmark_cec2022/run/run_instrumented_nerso.py

# Regenerate figures / analysis tables:
python experiments/benchmark_cec2022/analysis/make_paper_figs.py
```

The expected-running-time (wall-clock) table and per-cell statistics are aggregated by the analysis scripts into `experiments/benchmark_cec2022/analysis/`.

## Repository layout

```
NeRSO/
├── algorithms/            # algorithm implementations
│   ├── nerso.py         # NeRSO (regime-switching neural kernel) — this paper
│   ├── eio.py           # EIO v3 base class (shared neural components, history)
│   ├── pso_family.py    # AD-PSO and baselines (predecessor family)
│   ├── sca_family.py    # AD-SCA and native SCA (predecessor family)
│   └── de_family.py     # LSHADE and DE baselines
├── core/
│   ├── common.py        # registry, run protocol, paired-seed machinery
│   └── counters.py      # unified function-evaluation counting
├── experiments/
│   └── benchmark_cec2022/
│       ├── run/           # formal runs, instrumented runs, LOO experiments
│       └── analysis/    # statistics, tables, paper figures
└── paper_figs/            # figure-generation scripts (paper style)
```

## Citation

If you use NeRSO in your research, please cite:

```bibtex
@article{he2026nerso,
  title   = {NeRSO: A Neural Regime-Switching Optimizer with Kernel-Grown Neurodynamics},
  author  = {He, Xu and Meng, Xiaolin and Zhang, Youdong and An, Xiangdong and Liu, Jingnan},
  journal = {IEEE Transactions on Evolutionary Computation},
  year    = {2026},
  note    = {under review}
}
```

Predecessor work:

```bibtex
@article{he2026onesystem,
  title   = {One Nervous System, Two Bodies: A Controlled Comparison of Inertial
             and Oscillatory Motion Kernels in Neurodynamics-Inspired
             Adaptive-Drift Swarm Optimization},
  author  = {He, Xu and Zhang, Youdong and Yin, Wenxuan and Liu, Fengyuan
             and Ma, Yuqing and An, Xiangdong and Liu, Jingnan and Meng, Xiaolin},
  journal = {IEEE Transactions on Evolutionary Computation},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgment

The development of NeRSO followed a registered-prediction discipline: all four quantitative claims were fixed before the benchmarking experiments, and negative results are reported in the paper. Supported by the Natural Science Foundation of Jiangsu Province (BK20243064) and the Postgraduate Research & Practice Innovation Program of Jiangsu Province (SJCX24_0067).
