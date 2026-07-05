"""
问题2：FY1 投放 1 枚烟幕干扰弹对 M1 实施干扰，
优化无人机飞行方向 heading、速度 speed、投放延迟 t_drop、引信延迟 t_fuze，
使遮蔽时长尽可能长。

背景（几次迭代踩过的坑，决定了这里的做法）：
1. 直接对整个4维控制空间 (speed, heading, t_drop, t_fuze) 跑差分进化、用单点播种，
   收敛到一个局部最优 (heading≈176.65°, duration≈4.543s)，但漏掉了另一个完全独立、
   更优的解 (heading≈5°, duration≈4.588s)。原因：均匀随机采样命中非零解的概率只有
   约0.015%，可行域是被大片"全零死区"分隔的稀疏孤岛，单点播种的种群只覆盖了其中
   一个孤岛。
2. 改成按 heading 分网格扫描，虽然能同时找到两个孤岛，但这是"按方向分类"的笨办法：
   如果孤岛更多、或者两个孤岛 heading 接近但起爆时刻差很远，网格法可能漏掉或混淆。

真正的改进（本版做法）：
"孤岛"是控制空间 (speed, heading, t_drop, t_fuze) 参数化方式造成的假象——
bomb_detonation 这个映射把"起爆点+起爆时刻"这个物理上连续的空间，非线性地折叠成了
控制空间里分得很开的几块。而"是否被遮蔽"这个条件，本质上是"起爆点要靠近导弹→目标
这条随时间扫过的视线"，这是一个可以直接构造的几何对象。

所以候选生成不再靠瞎撞或按方向分网格，而是：
  1. 随机取一个起爆时刻，算出导弹当时的位置；
  2. 沿"导弹位置→目标圆柱边缘某点"这条连线随机取一点、加小扰动，作为候选起爆点
     （直接对准"应该起效"的区域，而不是瞎猜）；
  3. 用运动学关系解析反解出所需的 (speed, heading, t_drop, t_fuze)；
  4. 检查解出来的速度/时间是否是 FY1 实际可达的（speed∈[70,140], t_drop>=0）。
这套采样不依赖"预先知道有几个孤岛、分别在哪个方向"——不管背后有多少个互相独立的
可行区域，采样都会按各自在这个"起爆时刻×位置比例×目标角度"空间里的体积占比自动
覆盖到。找到候选后再聚类（避免同一个孤岛内的候选互相干扰），每个类簇分别精修，
最后横向比较。
"""

import numpy as np
from scipy.optimize import minimize

from model import (
    F_single_bomb,
    UAV_SPEED_MIN,
    UAV_SPEED_MAX,
    UAV_INIT,
    G,
    REAL_TARGET_CENTER,
    REAL_TARGET_RADIUS,
    REAL_TARGET_HEIGHT,
    missile_position,
)

UAV_NAME = "FY1"
MISSILE_NAME = "M1"

# 决策变量 X = (speed, heading_rad, t_drop, t_fuze)
BOUNDS = [
    (UAV_SPEED_MIN, UAV_SPEED_MAX),  # 无人机速度 m/s
    (0.0, 2 * np.pi),  # 飞行方向角 弧度
    (0.0, 60.0),  # 投放延迟 s
    (
        0.001,
        20.0,
    ),  # 引信延迟 s（下界只要求为正；20s 是 1800m 高度自由落体到地面所需时间上限的近似）
]

# 起爆时刻的搜索上限：宽松地覆盖导弹飞抵假目标所需时间量级即可，超过这个时间再拦截
# 已经没有意义（missile_position 会验证：duration 自然趋于 0，不需要精确卡死上限）。
DET_TIME_MAX = 70.0


def objective(x: np.ndarray, n_coarse: int = 150) -> float:
    """负的遮蔽时长（供最小化算法使用）。"""
    speed, heading, t_drop, t_fuze = x
    duration, _, _, _ = F_single_bomb(
        UAV_NAME, MISSILE_NAME, speed, heading, t_drop, t_fuze, n_coarse=n_coarse
    )
    return -duration


def geometric_candidates(
    n_samples: int = 200_000,
    perturb_scale: float = 8.0,
    rng: np.random.Generator = np.random.default_rng(0),
) -> np.ndarray:
    """
    几何构造候选起爆配置，并反解为 FY1 实际可达的控制量。
    返回形状 (n_feasible, 4) 的数组，每行是一个可行的 (speed, heading, t_drop, t_fuze)。
    """
    p0 = UAV_INIT[UAV_NAME]

    det_times = rng.uniform(0.01, DET_TIME_MAX, n_samples)
    s = rng.uniform(0.0, 1.0, n_samples)  # 沿"导弹->目标边缘点"连线的位置比例
    theta = rng.uniform(0.0, 2 * np.pi, n_samples)  # 目标圆柱边缘的角度
    target_z = rng.uniform(0.0, REAL_TARGET_HEIGHT, n_samples)
    perturb = rng.normal(scale=perturb_scale, size=(n_samples, 3))

    m = missile_position(MISSILE_NAME, det_times)  # (n_samples, 3)
    tgt = np.stack(
        [
            REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(theta),
            REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(theta),
            REAL_TARGET_CENTER[2] + target_z,
        ],
        axis=-1,
    )
    det_points = m + s[:, None] * (tgt - m) + perturb  # (n_samples, 3)

    # 反解控制量：由 det_z 解 t_fuze，由 det_time 解 t_drop，由起爆点水平位移解 heading/speed
    dz = p0[2] - det_points[:, 2]
    t_fuze = np.sqrt(np.maximum(dz, 0.0) / (0.5 * G))
    t_drop = det_times - t_fuze
    dxy = det_points[:, :2] - p0[:2]
    dist = np.linalg.norm(dxy, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = dist / det_times
    heading = np.arctan2(dxy[:, 1], dxy[:, 0]) % (2 * np.pi)

    feasible = (
        (dz >= 0) & (t_drop >= 0) & (speed >= UAV_SPEED_MIN) & (speed <= UAV_SPEED_MAX)
    )
    return np.stack([speed, heading, t_drop, t_fuze], axis=1)[feasible]


def cluster_candidates(candidates: np.ndarray, gap: float = 0.08) -> list[np.ndarray]:
    """
    对候选点按控制空间里的归一化距离做简单聚类（并查集），
    分开互相独立的孤岛，避免不同孤岛的候选互相污染局部精修的结果。
    """
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    normed = (candidates - lo) / (hi - lo)

    n = len(candidates)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        diff = np.abs(normed[i + 1 :] - normed[i])
        diff[:, 1] = np.minimum(
            diff[:, 1], 1.0 - diff[:, 1]
        )  # heading 是周期量，用最短角距离而不是直接差值
        dist = np.linalg.norm(diff, axis=1)
        close = np.nonzero(dist < gap)[0] + (i + 1)
        for j in close:
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [candidates[idx] for idx in groups.values()]


def polish(x0: np.ndarray, n_coarse: int = 800) -> tuple[np.ndarray, float]:
    """对单个候选点做带边界约束的 Nelder-Mead 局部精修。"""
    result = minimize(
        objective,
        x0,
        args=(n_coarse,),
        method="Nelder-Mead",
        bounds=BOUNDS,
        options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 3000},
    )
    return result.x, -result.fun


if __name__ == "__main__":
    # rng = np.random.default_rng(42)

    # ---- 第一阶段：几何构造候选起爆配置，反解为可行控制量 ----
    feasible = geometric_candidates()
    print("采样得到 %d 个可行候选（速度/时间约束均满足）" % len(feasible))

    # 用较低精度快速评分，先过滤掉零遮蔽的候选
    scores = np.array([-objective(x, n_coarse=60) for x in feasible])
    nonzero = feasible[scores > 0]
    print("其中 %d 个候选有非零遮蔽" % len(nonzero))

    # ---- 第二阶段：聚类成互相独立的"孤岛"，每个孤岛取最优代表点 ----
    clusters = cluster_candidates(nonzero)
    print("聚类为 %d 个互相独立的孤岛" % len(clusters))

    seeds = []
    for cluster in clusters:
        cluster_scores = np.array([-objective(x, n_coarse=150) for x in cluster])
        seeds.append(cluster[np.argmax(cluster_scores)])

    # ---- 第三阶段：对每个孤岛的代表点分别做局部精修，互不干扰 ----
    polished = [polish(x0) for x0 in seeds]
    polished.sort(key=lambda item: item[1], reverse=True)

    print("\n各孤岛精修后的遮蔽时长（按从优到劣排序）：")
    for x, d in polished:
        print(
            "  duration=%.4f  speed=%.3f heading=%.4f(%.2fdeg) t_drop=%.4f t_fuze=%.4f"
            % (d, x[0], x[1], np.degrees(x[1]) % 360, x[2], x[3])
        )

    best_x, _ = polished[0]

    # ---- 最终高精度评估 ----
    speed, heading, t_drop, t_fuze = best_x
    duration, drop_point, det_point, det_time = F_single_bomb(
        UAV_NAME, MISSILE_NAME, speed, heading, t_drop, t_fuze, n_coarse=8000
    )

    heading_deg = np.degrees(heading) % 360

    print("\n===== 问题2 最优投放策略 =====")
    print("飞行方向: %.4f rad (%.2f deg)" % (heading, heading_deg))
    print("飞行速度: %.4f m/s" % speed)
    print("投放延迟: %.4f s" % t_drop)
    print("引信延迟: %.4f s" % t_fuze)
    print("投放点:", drop_point)
    print("起爆点:", det_point)
    print("起爆时刻(相对任务下达): %.4f s" % det_time)
    print("有效遮蔽时长: %.4f s" % duration)
