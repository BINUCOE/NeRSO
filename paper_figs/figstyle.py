"""figstyle.py — 论文图件统一样式（2026-10-02 台账 T0）。
所有 make_paper_figs.py 出的图 import 本文件。沙箱无正版 Times New Roman，
用 Liberation Serif 顶替（与 Times 逐字宽一致）；本机有正版时把 FAMILY 改一行即可。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "STIXGeneral",
                                    "DejaVu Serif"],  # 按可用性自动回退
                     "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9, "mathtext.fontset": "stix",  # 数学字体与 Times 风格一致
                     "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
                     "axes.linewidth": 0.8, "lines.linewidth": 1.2, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "figure.dpi": 110, })

# 色板：两身各一主色；基线三色可区分（Okabe-Ito 色盲安全）。全论文图统一。
C_ADPSO = "#2166ac"  # AD-PSO 蓝
C_ADSCA = "#b2182b"  # AD-SCA 红
C_PSO = "#009e73"  # PSO 青绿
C_SCA = "#e69f00"  # SCA 橙
C_WDEC = "#8c8c8c"  # wDecay 灰
C_BASE = C_WDEC  # 兼容旧引用
# EIO 工程扩展：主角 EIO 系 + 同谱系/DE 基线（Okabe-Ito 色盲安全扩展）
C_EIO = "#7b3294"  # EIO 主色（紫，与 AD 蓝/红区分）
C_EIO_CHAIN = "#c2a5cf"  # EIO 消融链（浅紫）
C_NPD = "#4dac26"  # NPDOA（绿）
C_DE = "#636363"  # DE 族（深灰）
ALGO_STYLE = {  # 读者口径名 -> (颜色, 线宽)
    "AD-PSO": (C_ADPSO, 1.8), "AD-SCA": (C_ADSCA, 1.8), "PSO": (C_PSO, 1.0), "SCA": (C_SCA, 1.0),
    "wDecay": (C_WDEC, 1.0),  # EIO 家族（主角，加粗）
    "SCA-AD": ("#8c8c8c", 1.0), "EIO v0.4": (C_EIO_CHAIN, 1.0), "EIO v0.5": (C_EIO_CHAIN, 1.0),
    "EIO v0.6": (C_EIO_CHAIN, 1.0), "EIO v0.7": (C_EIO_CHAIN, 1.0), "EIO v0.8": (C_EIO_CHAIN, 1.0),
    "EIO v0.9": (C_EIO_CHAIN, 1.0), "EIO v1.0": (C_EIO, 2.2),  # 同谱系 SOTA
    "NPDOA": (C_NPD, 1.6),  # DE 族
    "DE": (C_DE, 1.0), "SHADE": (C_DE, 1.2), "L-SHADE": (C_DE, 1.4), "NL-SHADE-RSP-MID": (C_DE, 1.6), }


def newfig(width_mm, height_mm):
    """按论文栏宽开图：单栏 88，双栏 180（单位 mm）。"""
    return plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
