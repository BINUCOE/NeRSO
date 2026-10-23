# -*- coding: utf-8 -*-
# paper_fig_extra.py — Fig.3 替代方案试验场（2026-10-18）
# 三个候选：profile(分位数剖面) / passrate(通过率热力图) / strip(散点条图)
# 用法：IDE 直接运行本文件，产出三张 PNG 到 paper_figs/generated/。
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs


def _find_repo():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "core")) and os.path.isdir(os.path.join(d, "experiments")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise SystemExit("找不到仓库根（需含 core/ 与 experiments/）")


WS = _find_repo()
DATA = os.path.join(WS, "experiments/benchmark_cec2022/data")
TEST = os.path.join(WS, "tests/data")
OUT = os.path.join(WS, "paper_figs/generated")
os.makedirs(OUT, exist_ok=True)

ARMS3 = ["NeRSO", "NeRSO-H", "AD-PSO", "AD-SCA", "NPDOA", "LSHADE", "PSO"]
FILL = dict(fs.COL) if hasattr(fs, "COL") else {}
FILL = {"NeRSO": "#0072B2", "NeRSO-H": "#BBBBBB", "AD-PSO": "#009E73", "AD-SCA": "#56B4E9", "NPDOA": "#CC79A7",
        "LSHADE": "#000000", "PSO": "#F0E442"}
LW = {"NeRSO": 2.0, "NeRSO-H": 1.1, "AD-PSO": 1.1, "AD-SCA": 1.1, "NPDOA": 1.1, "LSHADE": 1.1, "PSO": 1.1}
LS = {"NeRSO": "solid", "NeRSO-H": (0, (4, 2)), "AD-PSO": "solid", "AD-SCA": "solid", "NPDOA": "solid",
      "LSHADE": (0, (1, 1.2)), "PSO": "solid"}
CLS = [("smooth", 0.5, 4.5), ("hybrid", 5.5, 7.5), ("composite", 8.5, 11.5)]


def _load():
    dff = pd.read_csv(os.path.join(DATA, "formal/cec2022_raw_results.csv"))
    df4 = pd.read_csv(os.path.join(DATA, "v4/cec2022_raw_results.csv"))
    return pd.concat([dff, df4], ignore_index=True)


# ---------------- 方案一：分位数剖面 ----------------
def profile(dfm):
    d = dfm[dfm.algo.isin(ARMS3)].copy()
    d["err"] = np.clip(d["error"], 1e-14, None)
    for dim in (10, 20):
        fig, ax = plt.subplots(figsize=(180 / 25.4, 62 / 25.4))
        for name, x0, x1 in CLS:
            ax.axvspan(x0, x1, color="0.94" if name != "composite" else "0.91", zorder=0)
            ax.plot([x0, x0, x1, x1], [1.03, 1.075, 1.075, 1.03], color="0.3", lw=0.8, clip_on=False,
                    transform=ax.get_xaxis_transform())
            ax.text((x0 + x1) / 2, 1.10, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6.8, style="italic", color="0.25")
        for a in ARMS3:
            sub = d[(d.algo == a) & (d.dim == dim)]
            med, q25, q75 = [], [], []
            for f in range(1, 13):
                v = sub[sub.func_id == f]["err"].values
                med.append(np.median(v))
                q25.append(np.percentile(v, 25))
                q75.append(np.percentile(v, 75))
            ax.fill_between(range(1, 13), q25, q75, color=FILL[a], alpha=0.13, lw=0)
            ax.plot(range(1, 13), med, color=FILL[a], lw=LW[a], ls=LS[a], marker="o", ms=2.6, label=a)
        ax.set_yscale("log")
        ax.set_xlim(0.5, 11.5)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["F%d" % f for f in range(1, 13)], fontsize=8)
        ax.set_xlabel("function (grouped by landscape class)", fontsize=8.5)
        ax.set_ylabel("error", fontsize=8.5)
        ax.grid(True, axis="y", lw=0.3, color="0.9")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=6.4, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.26),
                  columnspacing=0.9, handlelength=1.6, handletextpad=0.5)
        ax.set_title("(D = %d)" % dim, fontsize=9, loc="left", fontweight="bold")
        fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.15)
        fig.savefig(os.path.join(OUT, "fig3_alt_profile_D%d.png" % dim), dpi=300)
        plt.close(fig)
        print("fig3_alt_profile_D%d.png" % dim)


# ---------------- 方案二：通过率热力图 ----------------
def passrate(dfm):
    d = dfm[dfm.algo.isin(ARMS3)]
    ROWS = [(f, dd) for f in list(range(1, 6)) + list(range(6, 9)) + list(range(9, 13)) for dd in (10, 20)]
    M = np.zeros((7, 24))
    for j, a in enumerate(ARMS3):
        for i, (f, dd) in enumerate(ROWS):
            v = d[(d.algo == a) & (d.func_id == f) & (d.dim == dd)]["error"].values
            M[j, i] = (v <= 1e-8).mean()
    fig, ax = plt.subplots(figsize=(180 / 25.4, 58 / 25.4))
    ax.imshow(M, cmap="GnBu", vmin=0, vmax=1, aspect="auto")
    for i in range(24):
        for j in range(7):
            v = M[j, i]
            ax.text(i, j, "" if v == 0 else ("%d" % round(100 * v)), ha="center", va="center", fontsize=6.4,
                    color="white" if v > 0.6 else "black", fontweight="bold" if v == 1 else "normal")
    ax.set_xticks(range(24))
    ax.set_xticklabels(["F%d\nD%d" % r for r in ROWS], fontsize=5.2)
    ax.set_yticks(range(7))
    ax.set_yticklabels(ARMS3, fontsize=7.5)
    for i in (2, 4, 6, 8, 12, 14, 16, 18, 20, 22):
        ax.axvline(i - 0.5, color="white", lw=1.2)
    for j in range(1, 7):
        ax.axhline(j - 0.5, color="white", lw=0.9)
    for name, i0, i1 in [("smooth", -0.5, 9.5), ("hybrid", 9.5, 15.5), ("composite", 15.5, 23.5)]:
        ax.plot([i0, i0, i1, i1], [-0.55, -0.85, -0.85, -0.55], color="0.3", lw=0.8, clip_on=False)
        ax.text((i0 + i1) / 2, -0.95, name, ha="center", va="bottom", fontsize=7, style="italic", clip_on=False)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(6.5, -1.9)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.995, bottom=0.185)
    cax = fig.add_axes([0.30, 0.030, 0.40, 0.024])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="GnBu", norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal",
                      ticks=[0, 0.5, 1])
    cb.ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
    
    fig.savefig(os.path.join(OUT, "fig3_alt_passrate.png"), dpi=300)
    plt.close(fig)
    print("fig3_alt_passrate.png")


# ---------------- 方案三：散点条图 ----------------
def strip(dfm):
    d = dfm[dfm.algo.isin(ARMS3)].copy()
    W = 0.105
    for dim in (10, 20):
        sub = d[d.dim == dim]
        fig, axes = plt.subplots(3, 4, figsize=(180 / 25.4, 106 / 25.4))
        rng = np.random.default_rng(2)
        for fid, ax in enumerate(axes.flat, start=1):
            for j, a in enumerate(ARMS3):
                v = sub[(sub.func_id == fid) & (sub.algo == a)]["error"].values
                v = np.clip(v, 3e-11, None)
                x = 1 + (j - 3) * (W + 0.012) + rng.uniform(-W / 2.6, W / 2.6, len(v))
                main = (a == "NeRSO")
                ax.scatter(x, v, s=5 if main else 3.5, color=FILL[a], alpha=0.85 if main else 0.55, lw=0,
                           zorder=3 if main else 2)
            ax.set_yscale("log")
            ax.set_xticks([])
            ax.set_xlim(0.4, 1.6)
            ax.set_title("F%d" % fid, fontsize=9, fontweight="bold", loc="center",
                         color="#b2182b" if fid == 9 else "black")
            if fid == 9:
                for sp in ax.spines.values():
                    sp.set_edgecolor("#b2182b")
                    sp.set_linewidth(1.6)
            ax.grid(True, axis="y", which="major", lw=0.3, color="0.88")
            ax.set_axisbelow(True)
            if fid % 4 == 1:
                ax.set_ylabel("error", fontsize=9)
        handles = [mpatches.Patch(facecolor=FILL[a], lw=0, label=a) for a in ARMS3]
        fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.005),
                   columnspacing=1.0, handlelength=1.2)
        fig.suptitle("Per-run outcomes on CEC2022 (D = %d), 30 runs per cell" % dim, fontsize=10, fontweight="bold",
                     x=0.055, ha="left", y=0.978)
        fig.subplots_adjust(left=0.055, right=0.995, top=0.905, bottom=0.095, hspace=0.42, wspace=0.14)
        fig.savefig(os.path.join(OUT, "fig3_alt_strip_D%d.png" % dim), dpi=300)
        plt.close(fig)
        print("fig3_alt_strip_D%d.png" % dim)


if __name__ == "__main__":
    dfm = _load()
    profile(dfm)
    passrate(dfm)
    strip(dfm)
