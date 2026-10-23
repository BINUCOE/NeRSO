#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_v4_lgate.py — NeRSO 设计修订臂正式批（一键，约 1 小时，幂等续跑）

背景（LOO_RESULTS_2026-10-15.md §三/§四）：LOO 定价推导出的设计修订——
逃逸仅 L 相开火（H 相踢出 = 精细收敛毒剂，F11 双维 −5 dex；L 相踢出 = F9 唯一救援，+9.23 dex）。

预注册预测（跑前钉死，出数后逐格对照）：
  F9-D10  打平（≈2.14e-7，救援来自 L 相，门控不影响）
  F11-D10/D20  改善至 4.5e-13 / 精确零量级（−5 dex 级）
  F6  小改善（H 相踢出在该格亦无益）
  其余 20 格 |Δ|<0.2 噪声带
  对 NeRSO-fullEsc 的 Wilcoxon 预期：≈4–6 胜 / 18–20 平 / 0 负
若兑现：正典候选从 NeRSO-fullEsc 移名 NeRSO（同触发器同内核，逃逸按模态门控）。

落同一 data/v4（CSV 追加，runner 按 (algo,func,dim,run_id) 键跳过已完成行）。
完毕后：stats_formal.py --flagships NeRSO-fullEsc,NeRSO 出合并排位。
"""
import sys, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent
RUNNER = str(HERE / "run_formal_cec2022.py")
OUT = str(HERE.parent / "data" / "v4")

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("[批次] NeRSO 720（设计修订臂，预注册见 LOO_RESULTS 档案）", flush=True)
    cmd = [sys.executable, RUNNER, "--algos", "NeRSO",
           "--out-dir", OUT, "--workers", "12"]
    r = subprocess.run(cmd, cwd=str(WS))
    print("[批次] rc =", r.returncode, flush=True)
    print("完成。下一步：stats_formal.py --flagships NeRSO-fullEsc,NeRSO → 正典移名裁决。", flush=True)
