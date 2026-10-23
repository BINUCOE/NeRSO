#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_v4_loo.py — v4 零件级 LOO 三臂正式批（一键，约 1 小时，幂等续跑）

批：NeRSO-noEsc / NeRSO-Hnoise / NeRSO-H-noEsc，各 720（12 函数 × 2 维 × 30 runs），
落同一 data/v4（CSV 追加，runner 按 (algo,func,dim,run_id) 键跳过已完成行；
已落盘的 NeRSO-fullEsc 720 不受影响）。

三臂回答的问题（TECHNICAL §2 决策树）：
  NeRSO-noEsc    逃逸通道在切换核里还承重吗（预期复制 v3 的 F9 条件承重结论）
  NeRSO-Hnoise      "H 相关闭白噪声"spec 决策上秤（预期有害或中性 → 假设变登记）
  NeRSO-H-noEsc 锁 H 且无逃逸：精确分离 F9 灾难 = 核的惯性 还是 逃逸缺失

完毕后：stats_formal.py --flagships NeRSO-fullEsc 出合并排位；消融定价见 ARCHIVE §39 口径。
"""
import sys, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent
RUNNER = str(HERE / "run_formal_cec2022.py")
OUT = str(HERE.parent / "data" / "v4")

BATCHES = [
    ("LOO NeRSO-noEsc 720",    ["--algos", "NeRSO-noEsc"]),
    ("LOO NeRSO-Hnoise 720",      ["--algos", "NeRSO-Hnoise"]),
    ("LOO NeRSO-H-noEsc 720", ["--algos", "NeRSO-H-noEsc"]),
]

if __name__ == "__main__":
    for name, args in BATCHES:
        print("=" * 60, flush=True)
        print("[批次]", name, flush=True)
        cmd = [sys.executable, RUNNER] + args + ["--out-dir", OUT, "--workers", "12"]
        r = subprocess.run(cmd, cwd=str(WS))
        print("[批次]", name, "rc =", r.returncode, flush=True)
        if r.returncode != 0:
            break
    print("完成。下一步：stats_formal.py --flagships NeRSO-fullEsc → 消融表 v4 节成文。", flush=True)
