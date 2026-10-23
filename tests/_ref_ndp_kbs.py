# KEEP：NPDOA 等价性锚；tests/test_regression.py 第 3 节依赖本文件，勿归档（三司会审终裁）。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npd.py
NPD（Neural Population Dynamics，神经种群动力学优化算法，KBS 2024）独立实现

来源说明：
  从 baseline_optimizers.py 中提取 NPD 类（严格复现 KBS 2024 论文及 MATLAB
  官方代码的 NPDOA 算法），并整合 NPDOptimizer 包装器的功能（最大化适配、
  随机种子控制），使其脱离基准测试框架，可单独导入/运行。

算法特点（忠于原版）：
  - 内部统一按最小化处理（maximize=True 时自动取负适配最大化问题）
  - 无精英保留，子代完全替换父代
  - 三大策略：
      策略1  吸引子趋势（Attractor trending）
      策略2  耦合干扰（Coupling disturbance）
      策略3  信息投影（Information projection）

用法示例：
    import numpy as np
    from npd import NPD

    def sphere(x):
        return np.sum(x ** 2)

    solver = NPD(num_populations=60, dimensions=10, max_fe=6000,
                 obj_func=sphere, bounds=(-100.0, 100.0), seed=42)
    best_pos, best_fit, info = solver.optimize(verbose=True)
"""

import numpy as np
import random
from typing import Callable, Dict, Optional, Tuple


class NPD:
    """
    严格复现 KBS 2024 论文及 MATLAB 官方代码的 NPDOA 算法。

    参数：
        num_populations:    种群大小 N
        dimensions:         优化维度 D
        max_fe:             最大评估次数（终止条件）
        obj_func:           目标函数，输入 shape (D,) 的 np.ndarray，输出 float
        params:             算法参数字典，可含：
                              'AT': 吸引子比例（默认 0.1）
                              'Nd': 噪声缩放因子（默认 0.1）
                              'l' : 吸引子缩放因子（默认 0.7，兼容键名 'a'）
                              'd' : 耦合缩放因子（默认 0.1）
        bounds:             值域边界 (low, high)
        maximize:           True 表示最大化问题（内部取负按最小化求解，
                            返回值与 best_history 均转回原始视角）
        seed:               随机种子（None 则不固定）
        initial_population: 可选，shape (N, D) 的外部初始种群
    """

    def __init__(self, num_populations: int, dimensions: int, max_fe: int,
                 obj_func: Callable, params: Optional[Dict] = None,
                 bounds: Tuple[float, float] = (0.0, 1.0),
                 maximize: bool = False, seed: Optional[int] = None,
                 initial_population: Optional[np.ndarray] = None):
        self.N = num_populations
        self.D = dimensions
        self.max_fe = max_fe
        self.obj_func = obj_func
        self.bounds = np.array(bounds, dtype=float)
        self.maximize = maximize
        # 内部统一最小化：最大化问题取负
        self._sense = -1.0 if maximize else 1.0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        params = params or {}
        self.AT = params.get('AT', 0.1)
        self.Nd = params.get('Nd', 0.1)
        self.l = params.get('l', params.get('a', 0.7))
        self.d = params.get('d', 0.1)

        # 支持外部传入初始种群（统一起跑线）
        if initial_population is not None and len(initial_population) >= self.N:
            self.population = np.asarray(initial_population[:self.N], dtype=float).copy()
        else:
            self.population = np.random.uniform(bounds[0], bounds[1], (self.N, self.D))

        self.fitness = np.full(self.N, np.inf)   # 内部最小化视角
        self.fe = 0

        # 全局最优跟踪（内部最小化视角）
        self.gbest_fitness = np.inf
        self.gbest_position = None
        self.best_history = []                    # 记录原始视角的全局最优

    def _evaluate(self, pop: np.ndarray) -> np.ndarray:
        """评估种群适应度（转换为内部最小化视角），并累计评估次数"""
        fitness = np.array([self._sense * self.obj_func(ind) for ind in pop])
        self.fe += len(pop)
        return fitness

    def optimize(self, verbose: bool = False):
        """
        运行 NPD 优化。

        返回：
            best_position:  最优解，shape (D,)
            best_fitness:   最优适应度（原始视角，最大化问题为越大越好）
            info:           {'best_history': [...], 'eval_count': int}
        """
        self.fitness = self._evaluate(self.population)

        # 初始化全局最优
        best_idx = np.argmin(self.fitness)
        self.gbest_fitness = self.fitness[best_idx]
        self.gbest_position = self.population[best_idx].copy()
        self.best_history.append(self._sense * self.gbest_fitness)

        while self.fe < self.max_fe:
            # 按适应度升序排序（最小化视角，最优在前）
            rank = np.argsort(self.fitness)
            self.population = self.population[rank]
            self.fitness = self.fitness[rank]

            n_attractor = max(1, int(self.N * self.AT))
            attractors = self.population[:n_attractor]

            T = self.fe / self.max_fe
            offspring = self.population.copy()

            # ── 策略1：吸引子趋势 ──
            # 吸引子本身不动，其余个体向随机吸引子趋向，并叠加噪声
            for i in range(n_attractor, self.N):
                newone = self.population[i].copy()
                base = self.population[i].copy()

                for j in range(n_attractor):
                    A = np.random.randint(0, 2, self.D)
                    SA = np.random.randint(0, n_attractor)
                    bestdec = attractors[SA]
                    L = np.random.rand(self.D)
                    newone = newone + self.l * L * L * A * (bestdec - base)

                Lower = np.min(self.population, axis=0)
                Upper = np.max(self.population, axis=0)
                noise_std = self.Nd * (Upper - Lower)
                noise = np.random.randn() * noise_std
                newone = newone + noise
                offspring[i] = newone

            # ── 策略2&3：耦合干扰 + 信息投影 ──
            meanN = np.mean(offspring, axis=0)

            for i in range(self.N):
                newone = offspring[i].copy()

                l1 = 0.5 * self.d * np.random.rand(self.D)
                l2 = 0.5 * self.d * np.random.rand(self.D)
                L = np.random.randint(-1, 2, self.D)   # 原版代码保留（未直接使用）
                L1 = np.random.randint(-1, 2, self.D)
                L2 = np.random.randint(-1, 2, self.D)
                A1 = np.random.randint(0, 2, self.D)
                A2 = np.random.randint(0, 2, self.D)

                term_dif = L1 * l1 * A1 * (newone - meanN)
                term_add = L2 * A2 * l2 * meanN
                newone = newone + (term_dif + term_add) * np.random.rand() * (1 - T)
                offspring[i] = newone

            offspring = np.clip(offspring, self.bounds[0], self.bounds[1])
            offspring_fitness = self._evaluate(offspring)

            # 完全替换（无精英保留，忠于原版）
            self.population = offspring
            self.fitness = offspring_fitness

            # 更新全局最优
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.gbest_fitness:
                self.gbest_fitness = self.fitness[current_best_idx]
                self.gbest_position = self.population[current_best_idx].copy()

            self.best_history.append(self._sense * self.gbest_fitness)

            if verbose and (self.fe // self.N) % 10 == 0:
                print(f"[NPD] Gen={self.fe // self.N}, FE={self.fe}/{self.max_fe}, "
                      f"Best={self._sense * self.gbest_fitness:.6f}")

        best_fitness = self._sense * self.gbest_fitness
        info = {'best_history': self.best_history, 'eval_count': self.fe}
        return self.gbest_position, best_fitness, info


# ═══════════════════════════════════════════════════════════
#  独立测试入口（基站定位示例：最小化 + 最大化双向验证）
# ═══════════════════════════════════════════════════════════

def _test_objective_function(x: np.ndarray, A: np.ndarray, d: np.ndarray):
    """三维基站定位：最小化测距误差平方和（测试用）"""
    return np.sum((np.linalg.norm(A - x, axis=1) - d) ** 2)


def _generate_toa_instance(num_base_stations=6, space_size=100.0, noise_std=0.1):
    A = np.random.uniform(-space_size, space_size, (num_base_stations, 3))
    true_tag = np.random.uniform(-space_size / 2, space_size / 2, 3)
    true_distances = np.linalg.norm(A - true_tag, axis=1)
    d = np.maximum(true_distances + np.random.normal(0, noise_std, num_base_stations), 0.01)
    return A, true_tag, d


if __name__ == "__main__":
    # ── 测试1：基站定位（最小化模式）──
    A, true_tag, d = _generate_toa_instance(6, 100.0, noise_std=0.15)
    print("=" * 60)
    print("基站定位测试 (NPD 最小化模式)")
    print("=" * 60)
    print(f"真实标签位置: {true_tag}")

    solver = NPD(num_populations=60, dimensions=3, max_fe=6000,
                 obj_func=lambda x: _test_objective_function(x, A, d),
                 params={'AT': 0.1, 'Nd': 0.1, 'l': 0.7, 'd': 0.1},
                 bounds=(-100.0, 100.0), maximize=False, seed=42)
    best_pos, best_fit, info = solver.optimize(verbose=True)

    print("\n" + "=" * 60)
    print("结果")
    print("=" * 60)
    print(f"真实位置: {true_tag}")
    print(f"估计位置: {best_pos}")
    print(f"定位误差: {np.linalg.norm(best_pos - true_tag):.6f}")
    print(f"最终适应度: {best_fit:.8f}")
    print(f"评估次数: {info['eval_count']}")

    final_errors = info['best_history'][-10:]
    print("\n最后10代最优值:")
    for v in final_errors:
        print(f"  {v:.8f}")
    print(f"收敛差（最后5代极差）: {max(final_errors[-5:]) - min(final_errors[-5:]):.2e}")

    # ── 测试2：Sphere 函数（最大化模式验证，最优值为 0）──
    print("\n" + "=" * 60)
    print("Sphere 函数测试 (NPD 最大化模式，目标 max -sum(x^2) = 0)")
    print("=" * 60)

    solver2 = NPD(num_populations=60, dimensions=10, max_fe=6000,
                  obj_func=lambda x: -np.sum(x ** 2),
                  bounds=(-100.0, 100.0), maximize=True, seed=42)
    best_pos2, best_fit2, info2 = solver2.optimize(verbose=True)
    print(f"最优适应度: {best_fit2:.8f}（理论最优 0）")
    print(f"评估次数: {info2['eval_count']}")
    