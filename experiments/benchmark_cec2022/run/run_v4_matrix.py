#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_v4_matrix.py — v4 时代正式矩阵（历史批次，数据已在 data/v4）（一键，约 1.5-2 小时，幂等续跑）

批 1：NeRSO-fullEsc 全量 720（12 函数 × 2 维 × 30 runs，主角正式数据）
批 2：消融臂 4×720 = 2880（NeRSO-L / NeRSO-H / NeRSO-clock 时钟臂 / EIO 祖先）
落同一 out-dir（CSV 追加不冲突，顺序执行）。完毕后：
  analysis/merge_batches.py 合并 → analysis/stats_formal.py --flagships NeRSO-fullEsc → 图件。
"""
import sys, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent
RUNNER = str(HERE / "run_formal_cec2022.py")
OUT = str(HERE.parent / "data" / "v4")

BATCHES = [
    ("前身 NeRSO-fullEsc 720", ["--algos", "NeRSO-fullEsc"]),
    ("消融臂 4×720", ["--algos", "NeRSO-L,NeRSO-H,NeRSO-clock,EIO"]),
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
    print("完成。下一步：merge_batches.py（把 data/v4 加进合并列表）"
          " → analysis/stats_formal.py --flagships NeRSO", flush=True)
