#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches


# ---------- 路径 ----------
def _find_ws():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(d, "experiments", "benchmark_cec2022", "data")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("未找到工作区（需含 experiments/benchmark_cec2022/data）")


WS = _find_ws()
DATA = os.path.join(WS, "experiments", "benchmark_cec2022", "data")
TEST = os.path.join(WS, "tests", "data")
OUT = os.path.join(WS, "paper_figs", "generated")
os.makedirs(OUT, exist_ok=True)

# ---------- 样式 ----------
_avail = {f.name for f in font_manager.fontManager.ttflist}
SERIF = next((c for c in ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"] if c in _avail),
             "DejaVu Serif")
plt.rcParams.update({"font.family": "serif", "font.serif": [SERIF, "Liberation Serif"], "mathtext.fontset": "stix",
                     "axes.unicode_minus": False, "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.7})
COL = {"NeRSO": "#0072B2", "NeRSO-H": "#BBBBBB", "NeRSO-L": "#999999", "NeRSO-clock": "#E69F00",
       "NeRSO-noEsc": "#8B8B8B", "NeRSO-Hnoise": "#D0D0D0", "AD-PSO": "#009E73", "AD-SCA": "#56B4E9",
       "NPDOA": "#CC79A7", "LSHADE": "#000000", "PSO": "#DDDDDD", "wDecay": "#CCCCCC"}
FG = {1: 300., 2: 1100., 3: 600., 4: 1900., 5: 1700., 6: 1600., 7: 2100., 8: 2200., 9: 2300., 10: 2400., 11: 2600.,
      12: 2700.}


def grid(bud, n=121):
    g = np.unique(np.round(np.logspace(np.log10(50), np.log10(bud), n)).astype(int))
    g[-1] = bud
    return g


def fht(curve, fe, fg, target=1e-8):
    le = np.log10(np.maximum(np.abs(np.asarray(curve, float) - fg), 1e-300))
    hit = np.where(le <= np.log10(target))[0]
    if len(hit) == 0:
        return None
    k = max(int(hit[0]), 1)
    x0, x1 = np.log10(fe[k - 1]), np.log10(fe[k])
    y0, y1 = le[k - 1], le[k]
    return fe[k] if y1 == y0 else min(10 ** (x0 + (np.log10(target) - y0) * (x1 - x0) / (y1 - y0)), fe[-1])


def load():
    dff = pd.read_csv(f"{DATA}/formal/cec2022_raw_results.csv")
    df4 = pd.read_csv(f"{DATA}/v4/cec2022_raw_results.csv")
    zf = np.load(f"{DATA}/formal/conv_curves.npz")
    z4 = np.load(f"{DATA}/v4/conv_curves.npz")
    pv = np.load(f"{DATA}/p8_audit.npz", allow_pickle=True)
    pilot = np.load(f"{TEST}/pilot_v4.npz", allow_pickle=True)
    return dff, df4, zf, z4, pv, pilot


# ---------- Fig.1 所有权图 ----------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    BLUE, GRAY = "#0072B2", "#CCCCCC"
    ax = axes[0]
    ax.annotate("locked-L stalls at $6.6 \times 10^{-3}$\n(depth structurally capped)", xy=(8e4, 2e1),
                xytext=(1.2e3, 3e-9), fontsize=7, color="#56B4E9",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#56B4E9"))
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0.6, 1.6), 8.8, 6.6, boxstyle="round,pad=0.08", facecolor=BLUE, alpha=0.18,
                                         edgecolor=BLUE, lw=1.4))
    for (x0, lbl) in [(1.1, "L phase\n1st-order\noscillatory"), (5.0, "H phase\n2nd-order\ninertial")]:
        ax.add_patch(
            mpatches.FancyBboxPatch((x0, 3.1), 3.9, 3.4, boxstyle="round,pad=0.05", facecolor=GRAY, edgecolor="#666666",
                                    lw=1.0))
        ax.text(x0 + 1.95, 4.8, lbl, ha="center", va="center", fontsize=7, color="#333333")
    ax.annotate("", xy=(5.95, 4.8), xytext=(4.05, 4.8), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.3))
    ax.text(5.0, 5.7, "regime switch\n(state-governed)", ha="center", fontsize=6.5, color=BLUE)
    ax.text(5.0, 8.7, "NeRSO \u2014 one body, order is a state m(t)", ha="center", fontsize=8.5, fontweight="bold")
    for k, txt in enumerate(
            ["temporal organization:\nall timescales derived", "stochastic amplitude\n|E\u2212I|\u00b7\u03b8",
             "phase-gated escape\n(L-phase only)", "dynamical order\nconstant \u2192 state"]):
        ax.text(0.9 + k * 2.25, 2.5, txt, fontsize=5.6, color=BLUE, va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=BLUE, lw=0.6))
    ax.text(5, 0.7, "neural content constitutes the algorithm\u2019s organization", ha="center", fontsize=7.5,
            style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_ownership.png", dpi=300)
    plt.close(fig)


# ---------- Fig.2 rank_map ----------
def fig2(dfm):
    """Fig.2 rank_map 六变体全格结果（改版 v2，2026-10-17）。

    双栏 180mm，上下双子图 (a) D=10 / (b) D=20。行序按合并 24 格平均秩升序。
    单元格 = 该格 30 runs 中位误差（科学计数法，>=7pt）；格底按六臂名次着色
    （RdYlGn，1 绿 -> 6 红）；每格最优加粗白字；中位误差 <=1e-10 的地板格标 †
    （图注定义：达统计封底）。末列为该维平均秩；面板脚注给出合并 24 格口径。
    NeRSO 行以主色加粗标签 + 左侧色条标示。
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.cm import ScalarMappable
    ARMS = ["NeRSO", "NeRSO-H", "AD-PSO", "AD-SCA", "NPDOA", "LSHADE"]
    med = (dfm[dfm.algo.isin(ARMS)].groupby(["algo", "func_id", "dim"])["error"].median().reset_index())
    med["rank"] = med.groupby(["func_id", "dim"])["error"].rank(method="average")
    pooled = med.groupby("algo")["rank"].mean().sort_values()  # 合并 24 格平均秩
    ORDER = list(pooled.index)
    perdim = med.groupby(["dim", "algo"])["rank"].mean().unstack()  # 每维平均秩
    
    cmap = ListedColormap(plt.get_cmap("RdYlGn_r")(np.linspace(0.12, 0.88, 6)))
    norm = BoundaryNorm(np.arange(0.5, 7.5, 1), cmap.N)
    floor = 1e-10
    
    fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 110 / 25.4))
    for ax, dim, tag in zip(axes, (10, 20), ("(a) D = 10", "(b) D = 20")):
        R = np.zeros((6, 13))
        TXT = [[""] * 13 for _ in range(6)]
        for j in range(12):
            g = med[(med.func_id == j + 1) & (med.dim == dim)].set_index("algo")
            for i, a in enumerate(ORDER):
                v = g.loc[a, "error"]
                R[i, j] = g.loc[a, "rank"]
                TXT[i][j] = ("0" if v == 0 else "%.1e" % v) + ("†" if v <= floor else "")
        for i, a in enumerate(ORDER):
            R[i, 12] = perdim.loc[dim, a]
            TXT[i][12] = "%.2f" % perdim.loc[dim, a]
        ax.imshow(R, cmap=cmap, norm=norm, aspect="auto")
        for i in range(6):
            for j in range(13):
                tc = "white" if R[i, j] <= 1.5 else "black"
                ax.text(j, i, TXT[i][j], ha="center", va="center", fontsize=7, weight="bold",
                        color=tc)  # weight="bold" if R[i, j] == 1 else "normal", color=tc)
        ax.set_xticks(range(13), weight="bold")
        ax.set_xticklabels(["F%d" % (j + 1) for j in range(12)] + ["Rank"], fontsize=8.5, weight="bold")
        ax.set_yticks(range(6), weight="bold")
        labs = []
        for a in ORDER:
            lab = a if a != "NeRSO" else "NeRSO"
            labs.append(lab)
        ax.set_yticklabels(labs, fontsize=9, weight="bold")
        ax.get_yticklabels()[ORDER.index("NeRSO")].set_fontweight("bold")
        ax.get_yticklabels()[ORDER.index("NeRSO")].set_color(COL["NeRSO"])
        # NeRSO 行高亮：整行主色细边框（行内，不侵占标签区）
        rN = ORDER.index("NeRSO")
        ax.add_patch(
            mpatches.Rectangle((-0.5, rN - 0.5), 13.0, 1.0, fill=False, edgecolor=COL["NeRSO"], lw=1.8, clip_on=False))
        ax.set_title(tag, fontsize=10, fontweight="bold", pad=4)
        # ax.text(6.0, -0.88, "Pooled rank (24 cells): " + ",  ".join("%s %.2f" % (a, pooled[a]) for a in ORDER),
        #         ha="center", fontsize=6.8, style="italic", color="0.25")
        ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.6)
        ax.axvline(11.5, color="white", lw=5)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.955, bottom=0.18, hspace=0.30)
    cax = fig.add_axes([0.32, 0.055, 0.36, 0.026])
    cb = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation="horizontal", ticks=range(1, 7),
                      pad=0.085, shrink=0.55, aspect=28)
    cb.ax.tick_params(labelsize=8.5)
    for lbl in cb.ax.get_xticklabels():
        lbl.set_fontweight("bold")
    fig.text(0.5, 0.103, "per-cell rank (1 = best)", ha="center", fontsize=8.5, fontweight="bold")
    fig.savefig(f"{OUT}/fig2_rankmap.png", dpi=300)
    plt.close(fig)


# ---------- Fig.3 分位数剖面 + 通过率热力图（正式版，2026-10-18；替代旧箱体图） ----------
def fig3(dfm):
    """Fig.3 两张：（a/b）分位数剖面——每臂 median 线 + IQR 带，x = F1..F12 按地貌
    类聚（smooth/hybrid/composite 顶部括线 + 浅底纹）；（c）run 级通过率热力图——
    每格为 30 runs 中误差 <=1e-8 的比例。七臂口径与 Fig.2 一致。
    """
    ARMS3 = ["NeRSO", "NeRSO-H", "AD-PSO", "AD-SCA", "NPDOA", "LSHADE", "PSO"]
    FILL3 = {"NeRSO": "#0072B2", "NeRSO-H": "#BBBBBB", "AD-PSO": "#009E73", "AD-SCA": "#56B4E9", "NPDOA": "#CC79A7",
             "LSHADE": "#000000", "PSO": "#F0E442"}
    LW3 = {"NeRSO": 2.0, "NeRSO-H": 1.1, "AD-PSO": 1.1, "AD-SCA": 1.1, "NPDOA": 1.1, "LSHADE": 1.1, "PSO": 1.1}
    LS3 = {"NeRSO": "solid", "NeRSO-H": (0, (4, 2)), "AD-PSO": "solid", "AD-SCA": "solid", "NPDOA": "solid",
           "LSHADE": (0, (1, 1.2)), "PSO": "solid"}
    CLS3 = [("smooth", 0.96, 4.96), ("hybrid", 5.04, 7.96), ("composite", 8.04, 12.0)]
    d = dfm[dfm.algo.isin(ARMS3)].copy()
    d["err"] = np.clip(d["error"], 1e-14, None)
    
    # ---- (a)(b) 分位数剖面 ----
    for dim in (10, 20):
        fig, ax = plt.subplots(figsize=(180 / 25.4, 62 / 25.4))
        for name, x0, x1 in CLS3:
            ax.axvspan(x0, x1, color="0.94" if name != "composite" else "0.91", zorder=0)
            ax.plot([x0, x0, x1, x1], [1.03, 1.075, 1.075, 1.03], color="0.3", lw=0.8, clip_on=False,
                    transform=ax.get_xaxis_transform())
            ax.text((x0 + x1) / 2, 1.10, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.5, style="italic", color="0.15", fontweight="bold")
        for a in ARMS3:
            sub = d[(d.algo == a) & (d.dim == dim)]
            med, q25, q75 = [], [], []
            for f in range(1, 13):
                v = sub[sub.func_id == f]["err"].values
                med.append(np.median(v))
                q25.append(np.percentile(v, 25))
                q75.append(np.percentile(v, 75))
            ax.fill_between(range(1, 13), q25, q75, color=FILL3[a], alpha=0.13, lw=0)
            ax.plot(range(1, 13), med, color=FILL3[a], lw=LW3[a], ls=LS3[a], marker="o", ms=2.6, label=a)
        ax.set_yscale("log")
        ax.set_xlim(0.5, 11.5)
        ax.set_xticks(range(1, 13), fontweight="bold")
        ax.set_xticklabels(["F%d" % f for f in range(1, 13)], fontsize=8, fontweight="bold")
        ax.set_xlabel("Function  (grouped by landscape class)", fontsize=8.5, fontweight="bold")
        ax.set_ylabel("Error  (D=%d)" % dim, fontsize=8.5, fontweight="bold")
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, axis="y", lw=0.3, color="0.9")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=True, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.33), columnspacing=0.9,
                  handlelength=1.6, handletextpad=0.5, prop={"size": 6.7, "weight": "bold"})
        # ax.set_title("(D = %d)" % dim, fontsize=9, loc="left", fontweight="bold")
        fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.16)
        fig.savefig(f"{OUT}/fig3_profile_D%d.png" % dim, dpi=300)
        plt.close(fig)
    
    # ---- (c) 通过率热力图 ----
    ROWS3 = [(f, dd) for f in list(range(1, 6)) + list(range(6, 9)) + list(range(9, 13)) for dd in (10, 20)]
    M = np.zeros((7, 24))
    for j, a in enumerate(ARMS3):
        for i, (f, dd) in enumerate(ROWS3):
            v = d[(d.algo == a) & (d.func_id == f) & (d.dim == dd)]["error"].values
            M[j, i] = (v <= 1e-8).mean()
    fig, ax = plt.subplots(figsize=(180 / 25.4, 58 / 25.4))
    ax.imshow(M, cmap="GnBu", vmin=0, vmax=1, aspect="auto")
    for i in range(24):
        for j in range(7):
            v = M[j, i]
            ax.text(i, j, "" if v == 0 else ("%d" % round(100 * v)), ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else "black", fontweight="bold")
    ax.set_xticks(range(24))
    ax.set_xticklabels(["F%d\nD%d" % r for r in ROWS3], fontsize=7, fontweight="bold")
    ax.set_yticks(range(7))
    ax.set_yticklabels(ARMS3, fontsize=7.2, fontweight="bold")
    for i in (2, 4, 6, 8, 12, 14, 16, 18, 20, 22):
        ax.axvline(i - 0.5, color="white", lw=1.2)
    for j in range(1, 7):
        ax.axhline(j - 0.5, color="white", lw=0.9)
    for name, i0, i1 in [("smooth", -0.5, 9.5), ("hybrid", 9.5, 15.5), ("composite", 15.5, 23.5)]:
        ax.plot([i0, i0, i1, i1], [-0.55, -0.85, -0.85, -0.55], color="0.3", lw=0.8, clip_on=False)
        ax.text((i0 + i1) / 2, -0.95, name, ha="center", va="bottom", fontsize=7, style="italic", fontweight="bold")
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(6.5, -1.9)
    fig.subplots_adjust(left=0.07, right=0.99, top=1.05, bottom=0.245)
    cax = fig.add_axes([0.30, 0.08, 0.404, 0.024])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="GnBu", norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal",
                      ticks=[0, 0.5, 1])
    cb.ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7.5)
    for lbl in cb.ax.get_xticklabels():
        lbl.set_fontweight("bold")
    fig.savefig(f"{OUT}/fig3_passrate.png", dpi=300)
    plt.close(fig)


# ---------- Fig.4 Pareto（改版 v2，2026-10-17） ----------
def fig4():
    """Fig.4 分档 Pareto 散点：覆盖(1e-8 档) vs 墙钟，单栏 88mm。

    数据与 Table IX 同口径（coverage=中位<=1e-8 格数；墙钟=ERTx 中位 spf），
    已从 formal/v4 重算核验，与表内数值逐一相等；SCA 零覆盖不入图。
    阶梯虚线为 Pareto 前沿；NeRSO 星形主色强调并注 "fastest non-DE"。
    """
    tier = {}
    for csvp, npzp, pool in [(f"{DATA}/formal/cec2022_raw_results.csv", f"{DATA}/formal/conv_curves.npz",
                              ["LSHADE", "AD-SCA", "AD-PSO", "PSO", "wDecay", "NPDOA"]),
                             (f"{DATA}/v4/cec2022_raw_results.csv", f"{DATA}/v4/conv_curves.npz", ["NeRSO"])]:
        df_ = pd.read_csv(csvp)
        z_ = np.load(npzp)
        spf = (df_.runtime / df_.fe).groupby(df_.algo).median().to_dict()
        for a in pool:
            ert, cov = [], 0
            for f in range(1, 13):
                for d, bud in ((10, 200000), (20, 500000)):
                    if df_[(df_.algo == a) & (df_.func_id == f) & (df_.dim == d)].error.median() <= 1e-8:
                        cov += 1
                        fe = grid(bud)
                        fh = [fht(z_[f"{a}__{f}__{d}__{r}"], fe, FG[f]) for r in range(30)]
                        ert.append(np.mean([fe[-1] if h is None else h for h in fh]))
            tier[a] = (cov, np.mean(ert) * spf[a])
    OFF = {"AD-PSO": (-4, -14, "right"), "PSO": (0, -13, "center"), "NeRSO": (11, -2, "left"),
           "wDecay": (0, -13, "center"), "LSHADE": (7, 3, "left"), "AD-SCA": (7, -3, "left"), "NPDOA": (-7, 4, "right")}
    fig, ax = plt.subplots(figsize=(122 / 25.4, 54 / 25.4))
    for a, (c, t) in tier.items():
        main = (a == "NeRSO")
        ax.scatter(t, c, s=120 if main else 46, marker="*" if main else "o", color=COL[a], zorder=5 if main else 4,
                   edgecolors="white" if main else "none", linewidths=0.7 if main else 0)
        dx, dy, ha = OFF[a]
        ax.annotate(a, (t, c), textcoords="offset points", xytext=(dx, dy), fontsize=8.5 if main else 7.5, ha=ha,
                    fontweight="bold", color=COL["NeRSO"] if main else "black")
    ax.annotate("(LSHADE)", (13.1, 9), textcoords="offset points", xytext=(7, -9), fontsize=6.8, ha="left",
                color="0.35", style="italic", fontweight="bold")
    pts = sorted([(t, c) for c, t in tier.values()])
    front, best = [], -1
    for t, c in pts:
        if c > best:
            front.append((t, c))
            best = c
    ax.step([f[0] for f in front], [f[1] for f in front], where="post", ls=(0, (4, 2.5)), color="0.45", lw=1.0,
            zorder=2)
    ax.annotate("Pareto frontier", xy=(7.33, 5.3), fontsize=7.0, color="0.35", style="italic", ha="center",
                fontweight="bold")
    ax.annotate("fastest non-DE\nwidest coverage outside LSHADE", xy=(9.4, 7), xytext=(8.1, 8.75), fontsize=7.2,
                color=COL["NeRSO"], ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=COL["NeRSO"]))
    ax.set_xscale("log")
    ax.set_xlim(4.3, 55)
    ax.set_ylim(0, 10)
    ax.set_xticks([5, 10, 20, 50], fontweight="bold")
    ax.set_xticklabels(["5", "10", "20", "50"], fontweight="bold")
    ax.set_yticks(range(0, 10), fontweight="bold")
    ax.set_xlabel("Wall-clock time to $10^{-8}$ (s)", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Coverage (# of 24 cells)", fontsize=8.5, fontweight="bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(True, which="major", lw=0.3, color="0.9")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top=0.97, bottom=0.205)
    fig.savefig(f"{OUT}/fig4_pareto.png", dpi=300)
    plt.close(fig)


# ---------- Fig.5 四边形收敛小双联（改版 v3，2026-10-17） ----------
def fig5():
    """Fig.5 四边形收敛：F1(上,深度格) / F9(下,崎岖格)，D=10，中位曲线。

    四臂：NeRSO / NeRSO-L / NeRSO-H / NeRSO-clock。y 对数；曲线首点的
    inf 初始化哨兵置 NaN 断线；精确零画在 1e-14 底线。
    画法顺序 H -> clock -> NeRSO -> L（最上层）：L 与 NeRSO 在切换前
    轨迹重合，虚线置于顶层可区分二者；L 的 0.35 终值出现在预算末端，
    注记箭头指向曲线终点而非中段。
    终值已与正文核验（F1: L 止 6.6e-3 其余至零；F9: H 3.6e2, L 0.35,
    clock 2.1e-7, NeRSO 4.6e-13）。
    """
    z = np.load(f"{DATA}/v4/conv_curves.npz")
    ARMS5 = [("NeRSO-H", "#666666", (0, (1, 1.2)), 1.3), ("NeRSO-clock", "#E69F00", (0, (5, 1.5, 1, 1.5)), 1.3),
             ("NeRSO", COL["NeRSO"], "solid", 2.0), ("NeRSO-L", "#56B4E9", (0, (4, 2)), 1.6)]
    FLOOR5 = 1e-14
    fig, axes = plt.subplots(2, 1, figsize=(88 / 25.4, 104 / 25.4), sharex=True)
    fe = grid(200000)
    for ax, fid, tag in zip(axes, (1, 9), ("(a) F1, D = 10 (depth landscape)", "(b) F9, D = 10 (rugged landscape)")):
        for a, c, ls, lw in ARMS5:
            med = np.median(np.stack([np.abs(z[f"{a}__{fid}__10__{r}"] - FG[fid]) for r in range(30)]), axis=0)
            med = np.clip(med, FLOOR5, None)
            med[~np.isfinite(med)] = np.nan
            ax.plot(fe, med, color=c, ls=ls, lw=lw, label=a)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(1e-15, None)
        ax.grid(True, which="major", lw=0.3, color="0.9")
        ax.set_axisbelow(True)
        ax.set_title(tag, fontsize=8.5, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Error", fontsize=8.5, fontweight="bold")
    axes[1].set_ylabel("Error", fontsize=8.5, fontweight="bold")
    axes[1].set_xlabel("FE", fontsize=8.5, fontweight="bold")
    
    ax = axes[0]
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    ax.annotate("locked-L stalls at $6.6 \\times 10^{-3}$\n(depth structurally capped)", xy=(8e4, 2e1),
                xytext=(6.5e2, 3e-9), fontsize=7, color="#56B4E9", fontweight="bold", ha='center',
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#56B4E9"))
    ax = axes[1]
    for lbl in ax.get_yticklabels() + ax.get_xticklabels():
        lbl.set_fontweight("bold")
    ax.annotate("locked-H dies: $3.6 \\times 10^{2}$\n(inertial kernel without L-phase rescue)", xy=(3e4, 3.55e2),
                xytext=(6.7e2, 1e-2), fontsize=7, color="#666666", fontweight="bold", ha='center',
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#666666"))
    ax.annotate("locked-L survives: $3.5 \\times 10^{-1}$", xy=(1.8e5, 4e-1), xytext=(1.6e2, 6e-6), fontsize=7,
                color="#56B4E9", fontweight="bold", arrowprops=dict(arrowstyle="->", lw=0.7, color="#56B4E9"))
    ax.annotate("clock stops at $2.1 \\times 10^{-7}$", xy=(1.6e5, 2.3e-7), xytext=(1.6e2, 2e-9), fontsize=7,
                color="#E69F00", fontweight="bold", arrowprops=dict(arrowstyle="->", lw=0.7, color="#E69F00"))
    ax.annotate("NeRSO passes: $4.6 \\times 10^{-13}$", xy=(1.05e5, 5e-13), xytext=(1.5e2, 4e-13), fontsize=7,
                color=COL["NeRSO"], fontweight="bold", arrowprops=dict(arrowstyle="->", lw=0.8, color=COL["NeRSO"]))
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=lw, label=a) for a, c, ls, lw in
               [("NeRSO", COL["NeRSO"], "solid", 2.0), ("NeRSO-L", "#56B4E9", (0, (4, 2)), 1.6),
                ("NeRSO-H", "#666666", (0, (1, 1.2)), 1.3), ("NeRSO-clock", "#E69F00", (0, (5, 1.5, 1, 1.5)), 1.3)]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.00), columnspacing=1.6,
               handlelength=2.4, prop={"size": 6.7, "weight": "bold"})
    fig.subplots_adjust(left=0.155, right=0.975, top=0.95, bottom=0.165, hspace=0.3)
    fig.savefig(f"{OUT}/fig5_quad.png", dpi=300)
    plt.close(fig)


# ---------- Fig.6 消融定价热力图（改版 v3 转置，2026-10-17） ----------
def fig6(df4):
    """Fig.6 消融定价热力图（转置版）：x = 24 测试格（地貌类聚从左到右：
    光滑 F1-F5 / 混合 F6-F8 / 组合 F9-F12），y = 7 变体（正文全称，
    按 模式级/零件级/前身对照 分组）。单元格 Delta dex = log10(变体
    中位/NeRSO 中位)，红退蓝进（RdBu_r 截断 +-8）；双地板格灰底 dagger；
    Wilcoxon 显著格加粗 + 黑三角角标。
    """
    from scipy import stats as st
    from matplotlib.cm import ScalarMappable
    ARMS6 = [("NeRSO-L", "NeRSO-L"), ("NeRSO-H", "NeRSO-H"), ("NeRSO-clock", "NeRSO-clock"),
             ("NeRSO-noEsc", "NeRSO-noEsc"), ("NeRSO-Hnoise", "NeRSO-Hnoise"), ("NeRSO-H-noEsc", "NeRSO-H-noEsc"),
             ("NeRSO-fullEsc", "NeRSO-fullEsc")]
    GROUPS6 = [("mode-level", 0, 2), ("part-level", 3, 5), ("predecessor", 6, 6)]
    CLASSES = [("smooth", 1, 5), ("hybrid", 6, 8), ("composite", 9, 12)]
    ROWS6 = [(f, d) for f in list(range(1, 6)) + list(range(6, 9)) + list(range(9, 13)) for d in (10, 20)]
    basev = df4[df4.algo == "NeRSO"].groupby(["func_id", "dim"])["error"].median()
    M6 = np.zeros((24, 7))
    P6 = np.ones((24, 7))
    FL6 = np.zeros((24, 7), bool)
    for j, (a, _) in enumerate(ARMS6):
        for i, (f, d) in enumerate(ROWS6):
            v = df4[(df4.algo == a) & (df4.func_id == f) & (df4.dim == d)].sort_values("run_id").error.values
            n = df4[(df4.algo == "NeRSO") & (df4.func_id == f) & (df4.dim == d)].sort_values("run_id").error.values
            bm, am = basev.loc[(f, d)], float(np.median(v))
            if bm <= 1e-10 and am <= 1e-10:
                FL6[i, j] = True
                continue
            M6[i, j] = np.log10(max(am, 1e-300) / max(bm, 1e-300))
            if len(v) == 30 and not np.all(v == n):
                P6[i, j] = st.wilcoxon(v, n).pvalue
    cmap6 = plt.get_cmap("RdBu_r").copy()
    cmap6.set_bad("#f2f2f2")
    fig, ax = plt.subplots(figsize=(180 / 25.4, 80 / 25.4))
    ax.imshow(np.where(FL6, np.nan, np.clip(M6, -8, 8)).T, cmap=cmap6, vmin=-8, vmax=8, aspect="auto")
    for i in range(24):
        for j in range(7):
            if FL6[i, j]:
                ax.text(i, j, "\u2020", ha="center", va="center", fontsize=6.5, color="0.5")
            elif abs(M6[i, j]) >= 0.2:
                sig = P6[i, j] < 0.05
                ax.text(i, j, "%+.1f" % M6[i, j], ha="center", va="center", fontsize=6.2, weight="bold",
                        color="white" if abs(M6[i, j]) > 4.5 else "black")
            if P6[i, j] < 0.05 and not FL6[i, j]:
                ax.plot(i + 0.32, j - 0.32, marker="v", ms=2.2, color="black", mec="black", lw=0)
            if M6[i, j] > 16 and not FL6[i, j]:
                ax.plot(i - 0.30, j + 0.30, marker="^", ms=3.0, color="white", mec="0.15", mew=0.35, lw=0)
    ax.set_xticks(range(24))
    ax.set_xticklabels(["F%d\nD%d" % r for r in ROWS6], fontsize=6.3, fontweight="bold")
    ax.set_yticks(range(7), fontweight="bold")
    ax.set_yticklabels([a[1] for a in ARMS6], fontsize=7.5, fontweight="bold")
    for i in (2, 4, 6, 8, 12, 14, 16, 18, 20, 22):
        ax.axvline(i - 0.5, color="white", lw=1.1)
    for j in range(1, 7):
        ax.axhline(j - 0.5, color="white", lw=0.9)
    for name, j0, j1 in GROUPS6:  # 变体分组括注（右侧，与转置对称）
        y0, y1 = j0 - 0.5, j1 + 0.5
        ax.plot([23.55, 23.85, 23.85, 23.55], [y0, y0, y1, y1], color="0.3", lw=0.8, clip_on=False)
        ax.text(24.15, (y0 + y1) / 2, name, ha="center", va="center", rotation=90, fontsize=6.8, style="italic",
                clip_on=False, fontweight="bold")
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(6.5, -0.7)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.905, top=0.985, bottom=0.205)
    from matplotlib.lines import Line2D
    L6, R6, B6 = 0.128, 0.905, 0.235
    
    def _fx(xd):
        return L6 + (xd + 0.38) / 24.0 * (R6 - L6)
    
    for name, i0, i1 in [("smooth", 0, 9), ("hybrid", 10, 15), ("composite", 16, 23)]:
        x0, x1 = _fx(i0 - 0.5), _fx(i1 + 0.5)
        fig.add_artist(Line2D([x0, x0, x1, x1], [0.165, 0.118, 0.118, 0.165], color="0.3", lw=0.8))
        fig.text((x0 + x1) / 2, 0.108, name, ha="center", va="top", fontsize=7.5, style="italic", fontweight="bold")
    cax = fig.add_axes([0.952, 0.208, 0.012, 0.735])
    cb = fig.colorbar(ScalarMappable(cmap=cmap6, norm=plt.Normalize(-16, 16)), cax=cax, orientation="vertical",
                      ticks=[-16, -8, 0, 8, 16])
    cb.ax.tick_params(labelsize=7)
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    # fig.text(0.942, 0.602, "$\Delta$ dex (variant $-$ NeRSO)", rotation=90, ha="center", va="center", fontsize=7.2,
    #          fontweight="bold")
    fig.text(0.10, 0.025, "$\u2020$ both at floor;  $\\blacktriangledown$ $p<0.05$;  "
                          "$\\vartriangle$: beyond scale (value labeled);  "
                          "red = degradation, blue = improvement", fontsize=6.8, color="0.3", fontweight="bold")
    fig.savefig(f"{OUT}/fig6_ablation.png", dpi=300)
    plt.close(fig)


# ---------- Fig.7 机制验证六联图（v2 布局重写，2026-10-17；COMPAT_SW_v1 numpy1/2 兼容版，2026-10-18） ----------
def fig7(df4):
    """Fig.7（正文 Fig.8）mapping 核验六面板 —— 正典插桩批重出版（p9_canonical.npz）。

    2026-10-19 数据源切换：原版本面板的 S/散布/驻留/Fano 全部取自 NeRSO-fullEsc 臂
    （p8_audit + pilot + tests/data/p9_audit.npz），其中 fullEsc 臂 59.9% 的点火发生在
    相切换之后，与门控正典（H 相只积累不复位）的机制面相悖。切换时刻经 48/48 对运行
    验证与臂无关（由切换前轨迹逐位相同所致）。本版全部改由门控正典批取数：
      (a) S 积累锯齿：F1-D10 s1，gens 400–800（切换点 858 之前的 L 相窗口）；
      (b) 散布轨迹与切换点：F1-D10 s1（858 ≈ 21%）与 F9-D10 s1（2293 ≈ 57%）；
      (c) 驻留分布：24 格 × 3 种子，预测线 95.8 代；
      (d) 逐 run 切换时刻：70 次切换 + 2 次未触发（F6-D10 s2、F7-D10 s0），时钟线 0.60；
      (e) Fano 因子：100 代非重叠窗、窗口落于 L 相、窗均点火 > 0.5（与 Table IX 同口径）；
      (f) F9-D10 逃逸剂量收束：v4 CSV 正典定价，不涉及臂归属，保持原实现。
    """
    P9C = os.path.join(DATA, "p9_canonical.npz")
    if not os.path.exists(P9C):
        raise SystemExit("缺少正典插桩批 %s —— 请先运行 instrumented/p9_canonical_audit.py" % P9C)
    p9 = np.load(P9C, allow_pickle=True)
    ARM9 = "NeRSO"
    
    def _get9(prefix, f, d, sd):
        return p9["%s%s_F%d_D%d_s%d" % (prefix, ARM9, f, d, sd)]
    
    def _sw_gen9(f, d, sd):
        """返回首次切换代；-1 = 全程未切换。"""
        a = np.asarray(p9["SW_%s_F%d_D%d_s%d" % (ARM9, f, d, sd)])
        if a.size == 0:
            return -1
        if a.dtype.kind in "US":
            return int(str(a.reshape(-1)[0]).split(":")[0])
        return int(a.reshape(-1, 2)[0, 0])
    
    CELLS24 = [(f, d) for f in range(1, 13) for d in (10, 20)]
    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4))
    gs = fig.add_gridspec(2, 12, hspace=0.55, wspace=10.0, left=0.065, right=0.985, top=0.96, bottom=0.115)
    
    # ---------- (a) S 积累锯齿（正典：切换前窗口） ----------
    ax = fig.add_subplot(gs[0, 0:4])
    A = _get9("SJ_", 1, 10, 1).astype(float)
    g_sw = _sw_gen9(1, 10, 1)  # 858
    g0, g1 = 400, 800  # 切换前的 L 相窗口：积累-点火-复位可见
    for pp in range(0, 50, 10):
        ax.plot(range(g0, g1), A[g0:g1, pp], lw=0.8, color=COL["NeRSO"], alpha=0.85)
    ax.axhline(1.0, color="0.2", ls="--", lw=0.9)
    ax.text(g1 - 15, 1.05, "threshold $S$ = 1", fontsize=7.5, ha="right", fontweight="bold")
    ax.annotate("", xy=(700, 0.28), xytext=(604, 0.28), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(602, 0.13, "dwell time $\\approx$ $T_{SFA}$ = 95.8 gen", fontsize=7.5, ha="center", fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.85), zorder=5)
    ax.set_ylim(-0.03, 1.16)
    ax.set_title("(a) $S$ accumulation (LIF ramp), F1, D = 10", fontsize=8, fontweight="bold")
    ax.set_ylabel("$S$ (5 of 50 particles)", fontsize=8, fontweight="bold")
    ax.set_xlabel("generation (L phase; switch at %d)" % g_sw, fontsize=8, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    
    # ---------- (b) 散布轨迹 + 切换点（正典代表 run） ----------
    ax = fig.add_subplot(gs[0, 4:8])
    for f9, lab in [(1, "F1, D=10 (depth)"), (9, "F9, D=10 (rugged)")]:
        S = _get9("SP_", f9, 10, 1).astype(float)
        g = _sw_gen9(f9, 10, 1)
        c9 = "#0072B2" if f9 == 1 else "#b2182b"
        ax.plot(S, lw=1.1, color=c9, label=lab)
        ax.axvline(g, color=c9, ls="--", lw=1.2)
        ax.text(g - 840, S.max() * (0.05 if f9 == 1 else 0.007), "switch at %d gen (%d%%)" % (g, round(100 * g / 4000)),
                fontsize=6.6, color=c9, ha="left", va="bottom", fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.75))
    ax.set_xlim(-200, 3700)
    ax.set_yscale("log")
    ax.set_ylim(0.5e-1, S.max() * 3.0)
    ax.set_xlabel("generation", fontsize=8, fontweight="bold")
    ax.set_ylabel("population spread", fontsize=8, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95, prop={"size": 7, "weight": "bold"})
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_title("(b) Spread trajectories: F1 vs F9, D = 10", fontsize=8, fontweight="bold")
    
    # ---------- (c) 驻留分布（正典 24 格 × 3 种子） ----------
    ax = fig.add_subplot(gs[0, 8:12])
    dwells = []
    for (f, d) in CELLS24:
        for sd in range(3):
            sj = _get9("SJ_", f, d, sd).astype(float)
            fired = sj == 0
            for j in range(fired.shape[1]):
                dwells.extend(np.diff(np.where(fired[:, j])[0], prepend=-1).tolist())
    dwells = np.array(dwells)
    bins = np.arange(80, 165, 2)
    ax.hist(dwells, bins=bins, color="#0072B2", alpha=0.75, edgecolor="white", lw=0.4)
    ax.axvline(95.8, color="0.2", ls="--", lw=1.2)
    ax.text(97.2, ax.get_ylim()[1] * 0.85, "predicted $T_{SFA}$ = 95.8", fontsize=7.5, fontweight="bold")
    med = np.median(dwells)
    ax.text(0.87, 0.30, "median = %.1f gen\n($n$ = %s events)" % (med, "{:.1e}".format(len(dwells))),
            transform=ax.transAxes, ha="right", fontsize=7.5, fontweight="bold")
    ax.set_xlim(80, 165)
    ax.set_xlabel("dwell time (generations)", fontsize=8, fontweight="bold")
    ax.set_ylabel("count", fontsize=8, fontweight="bold")
    ax.set_yticklabels(["0", "20k", "40k", "60k", "80k"], fontweight="bold")
    ax.set_title("(c) Dwell time vs prediction", fontsize=8, fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    
    # ---------- (d) 逐 run 切换时刻（正典 70 点 + 2 未触发 + 时钟线） ----------
    ax = fig.add_subplot(gs[1, 0:6])
    rng = np.random.default_rng(0)
    xs_med, ys_med = [], []
    for i, (f, d) in enumerate(CELLS24):
        budg = 4000 if d == 10 else 10000
        fr = []
        for sd in range(3):
            g = _sw_gen9(f, d, sd)
            if g == -1:
                ax.plot(i, 1.035, marker="x", ms=5, color="#b2182b", mew=1.2)
            else:
                fr.append(g / budg)
                ax.plot(i + rng.uniform(-0.3, 0.3), g / budg, ".", ms=2.2, color="0.55", lw=0)
        if fr:
            xs_med.append(i)
            ys_med.append(np.median(fr))
    ax.plot(xs_med, ys_med, "_", ms=14, color="black", mew=1.6)
    ax.axhline(0.60, color="#E69F00", ls=(0, (4, 2.5)), lw=1.1)
    ax.text(5.2, 0.69, "clock variant: all runs at 60%", fontsize=7.5, color="#E69F00", fontweight="bold")
    ax.text(13.4, 0.95, "$\\times$ no switch in run\n(criterion never met)", fontsize=7.5, color="#b2182b",
            fontweight="bold")
    ax.set_xticks(range(24))
    ax.set_xticklabels(["F%d, D%d" % c for c in CELLS24], rotation=90, fontsize=7, fontweight="bold")
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("switch time (fraction of budget)", fontsize=8, fontweight="bold")
    ax.set_title("(d) Per-run switch times across 24 cells", fontsize=8, fontweight="bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    
    # ---------- (e) Fano（正典，同 Table IX 口径） ----------
    ax = fig.add_subplot(gs[1, 6:8])
    fanos = []
    for (f, d) in CELLS24:
        for sd in range(3):
            sj = _get9("SJ_", f, d, sd).astype(float)
            gsw = _sw_gen9(f, d, sd)
            if gsw == -1:
                gsw = sj.shape[0]
            fired = sj == 0
            rng_g = [g for g in range(0, sj.shape[0] - 100, 100) if g + 100 <= gsw]
            cs = np.array([fired[g:g + 100, :].sum() for g in rng_g])
            if len(cs) >= 5 and cs.mean() > 0.5:
                fv = cs.var(ddof=1) / cs.mean()
                if np.isfinite(fv):
                    fanos.append(fv)
    fanos = np.array(fanos)
    rng2 = np.random.default_rng(1)
    ax.scatter(rng2.uniform(-0.16, 0.16, len(fanos)), fanos, s=7, color=COL["NeRSO"], alpha=0.55, lw=0)
    ax.hlines(np.median(fanos), -0.25, 0.25, color="black", lw=1.2)
    ax.axhline(1.0, color="0.25", ls="--", lw=0.9)
    ax.text(-0.71, 0.38, "Poisson = 1", fontsize=6.8, fontweight="bold")
    ax.text(-0.85, -1.0, "regular clock\n$\\rightarrow$ Fano $\\approx$ 0", fontsize=7, color="0.1", fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels(["NeRSO\n(median %.2f)" % np.median(fanos)], fontsize=7.5, fontweight="bold")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.4, 7.3)
    ax.set_ylabel("Fano factor of kick counts\n(per 100-gen window)", fontsize=7.5, fontweight="bold")
    # ax.text(0.03, 0.98, "100-gen non-overlapping windows (L phase; mean count > 0.5)", transform=ax.transAxes,
    #         ha="left", va="top", fontsize=6.4, color="0.25")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_title("(e) Kick rhythm is Poisson-like", fontsize=8, fontweight="bold")
    
    # ---------- (f) F9-D10 剂量收束（v4 CSV 正典定价，保持原实现） ----------
    ax = fig.add_subplot(gs[1, 8:12])
    arms_f = ["NeRSO", "NeRSO-L", "NeRSO-H", "NeRSO-H-noEsc", "NeRSO-noEsc"]
    medf = [df4[(df4.algo == a) & (df4.func_id == 9) & (df4.dim == 10)].error.median() for a in arms_f]
    cols_f = [COL["NeRSO"], "#56B4E9", "#b2182b", "#b2182b", "#b2182b"]
    ax.bar(range(5), medf, color=cols_f, width=0.6, lw=0)
    for ii, v in enumerate(medf):
        ax.text(ii, v * 2.5, "%.1e" % v, ha="center", fontsize=6.7, va="bottom", fontweight="bold")
    ax.axhline(1e-8, color="0.25", ls="--", lw=0.9)
    ax.text(0.56, 2.3e-8, "$10^{-8}$ tier", fontsize=6.8, ha="right", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 1e6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(arms_f, fontsize=6.3, rotation=30, fontweight="bold")
    ax.set_ylabel("median error", fontsize=8, fontweight="bold")
    ax.set_title("(f) Escape-dose triangulation, F9-D10", fontsize=8, fontweight="bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontsize(7.5)
    fig.savefig(f"{OUT}/fig7_mapping.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    print("serif =", SERIF)
    dff, df4, zf, z4, pv, pilot = load()
    dfm = pd.concat([dff, df4], ignore_index=True)
    jobs = [("Fig.1", lambda: fig1()), ("Fig.2", lambda: fig2(dfm)), ("Fig.3", lambda: fig3(dfm)),
            ("Fig.4", lambda: fig4()), ("Fig.5", lambda: fig5()), ("Fig.6", lambda: fig6(df4)),
            ("Fig.7", lambda: fig7(df4))]
    for name, fn in jobs:
        fn()
        print(name, "完成")
    print("全部输出 ->", OUT)
