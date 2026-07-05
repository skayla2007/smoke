"""
2025 CUMCM A题 - 烟幕干扰弹投放策略
核心物理模型：给定一枚烟幕干扰弹的投放/起爆参数，计算其对真目标的有效遮蔽时长。

设计原则：
- 把运动学（导弹直线运动、弹丸抛体运动、云团匀速下沉）写成解析函数，这部分是已知物理，不是黑箱。
- 把"遮蔽时长"这个标量输出，作为后续优化算法要最大化的目标函数 F(X)，从优化的角度当黑箱使用。
"""

import numpy as np
from scipy.optimize import brentq

# ---------------- 常量 ----------------
G = 9.8  # 重力加速度 m/s^2
MISSILE_SPEED = 300.0  # 导弹速度 m/s
CLOUD_SINK_SPEED = 3.0  # 云团下沉速度 m/s
CLOUD_RADIUS = 10.0  # 有效遮蔽半径 m
CLOUD_LIFE = 20.0  # 起爆后有效时长 s
MIN_BOMB_INTERVAL = 1.0  # 同机相邻两枚弹最小投放间隔 s
UAV_SPEED_MIN, UAV_SPEED_MAX = 70.0, 140.0

FAKE_TARGET = np.array([0.0, 0.0, 0.0])
REAL_TARGET_CENTER = np.array([0.0, 200.0, 0.0])  # 圆柱下底面圆心
REAL_TARGET_RADIUS = 7.0
REAL_TARGET_HEIGHT = 10.0

MISSILE_INIT = {
    "M1": np.array([20000.0, 0.0, 2000.0]),
    "M2": np.array([19000.0, 600.0, 2100.0]),
    "M3": np.array([18000.0, -600.0, 1900.0]),
}

UAV_INIT = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0]),
}


def missile_position(name: str, t: float | np.ndarray) -> np.ndarray:
    """
    导弹 t 时刻位置：从初始点匀速直线飞向假目标（原点）。
    t 可为标量或任意形状的数组，返回形状为 t.shape + (3,)。
    """
    p0 = MISSILE_INIT[name]
    direction = (direction := FAKE_TARGET - p0) / np.linalg.norm(direction)
    t = np.asarray(t, dtype=float)
    return p0 + MISSILE_SPEED * t[..., None] * direction


def target_sample_points(n_theta: int = 36) -> np.ndarray:
    """
    对真目标（圆柱）采样若干代表点，用于近似判断"整体遮蔽"。
    只需采样上下底面圆周：圆柱体恰好是这两个圆周的凸包，而遮蔽区域（固定视点下）
    是凸集，圆周之外的点（侧面、底面内部、轴线）都必然已被圆周的遮蔽条件所覆盖，
    无需单独采样。
    """
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    pts = []
    for z in (0.0, REAL_TARGET_HEIGHT):
        for th in thetas:
            pts.append(
                [
                    REAL_TARGET_CENTER[0] + REAL_TARGET_RADIUS * np.cos(th),
                    REAL_TARGET_CENTER[1] + REAL_TARGET_RADIUS * np.sin(th),
                    REAL_TARGET_CENTER[2] + z,
                ]
            )
    return np.array(pts)


TARGET_POINTS = target_sample_points()


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """点 p 到线段 ab 的最短距离（支持向量化：p, a, b 可为 (...,3) 数组，广播）。"""
    ab = b - a
    ab_len2 = np.sum(ab * ab, axis=-1)
    ab_len2 = np.where(ab_len2 == 0, 1e-12, ab_len2)
    t = np.sum((p - a) * ab, axis=-1) / ab_len2
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1)


def bomb_detonation(
    uav_name: str, uav_speed: float, heading_rad: float, t_drop: float, t_fuze: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    根据无人机初始位置、飞行方向(水平面内角度，弧度，0=+x轴方向, 逆时针)、速度、
    投放延迟 t_drop、引信延迟 t_fuze，计算投放点与起爆点。
    无人机等高度飞行，弹丸脱离后做平抛运动。
    返回: drop_point, det_point, det_time (相对任务下达时刻 t=0)
    """
    p0 = UAV_INIT[uav_name]
    vel_xy = uav_speed * np.array([np.cos(heading_rad), np.sin(heading_rad)])
    drop_xy = p0[:2] + vel_xy * t_drop
    drop_point = np.array([drop_xy[0], drop_xy[1], p0[2]])

    det_xy = drop_xy + vel_xy * t_fuze
    det_z = p0[2] - 0.5 * G * t_fuze**2
    det_point = np.array([det_xy[0], det_xy[1], det_z])
    det_time = t_drop + t_fuze
    return drop_point, det_point, det_time


def cloud_center(
    det_point: np.ndarray, det_time: float, t: float | np.ndarray
) -> np.ndarray:
    """
    t 时刻云团中心位置（不做有效期截断）。t 可为标量或数组，返回形状 t.shape + (3,)。
    """
    t = np.asarray(t, dtype=float)
    tau = t - det_time
    xy = np.broadcast_to(np.array([det_point[0], det_point[1]]), tau.shape + (2,))
    z = det_point[2] - CLOUD_SINK_SPEED * tau
    return np.concatenate([xy, z[..., None]], axis=-1)


def shielding_margin(
    missile_name: str, det_point: np.ndarray, det_time: float, t: float | np.ndarray
) -> np.ndarray:
    """
    g(t) = max_i dist(cloud(t), segment(missile(t), target_point_i)) - CLOUD_RADIUS
    g(t) <= 0 <=> 所有目标采样点都被云团遮蔽（t 需已限制在 [det_time, det_time+20] 内）。
    t 可为标量或一维数组，向量化对 72 个目标采样点同时计算。
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    m = missile_position(missile_name, t)  # (N,3)
    c = cloud_center(det_point, det_time, t)  # (N,3)
    dists = point_segment_distance(
        c[:, None, :], m[:, None, :], TARGET_POINTS[None, :, :]
    )  # (N, n_target)
    return dists.max(axis=1) - CLOUD_RADIUS  # (N,)


def is_shielded_at(
    missile_name: str, det_point: np.ndarray, det_time: float, t: float
) -> bool:
    """判断 t 时刻，单枚烟幕弹是否让真目标对该导弹"整体不可见"。"""
    tau = t - det_time
    if tau < 0 or tau > CLOUD_LIFE:
        return False
    return bool(shielding_margin(missile_name, det_point, det_time, t)[0] <= 0)


def shielding_intervals(
    missile_name: str,
    det_point: np.ndarray,
    det_time: float,
    n_coarse: int = 400,
    xtol: float = 1e-9,
) -> list[tuple[float, float]]:
    """
    精确（到 xtol）计算单枚烟幕弹在有效期 [det_time, det_time+20] 内对目标的
    遮蔽时间区间列表（可能不止一段）。

    做法：先用向量化的粗网格找出 g(t)=0 的变号区间（覆盖状态切换点），
    再用 brentq 对每个变号区间做二分求根，得到覆盖区间的精确边界 —— 而不是
    用固定步长做黎曼和近似。这样既避免了 Python 级别的逐点循环（快），
    精度也不再受 dt 网格粗细限制（准）。
    """
    t0, t1 = det_time, det_time + CLOUD_LIFE
    ts = np.linspace(t0, t1, n_coarse)
    g = shielding_margin(missile_name, det_point, det_time, ts)

    g_scalar = lambda t: shielding_margin(missile_name, det_point, det_time, t)[0]

    signs = np.sign(g)
    signs[signs == 0] = 1.0
    roots = [t0]
    for i in range(len(ts) - 1):
        if signs[i] != signs[i + 1]:
            roots.append(brentq(g_scalar, ts[i], ts[i + 1], xtol=xtol))
    roots.append(t1)

    intervals = []
    for a, b in zip(roots[:-1], roots[1:]):
        if g_scalar(0.5 * (a + b)) <= 0:
            intervals.append((a, b))
    return intervals


def shielding_duration(
    missile_name: str,
    det_point: np.ndarray,
    det_time: float,
    n_coarse: int = 400,
    xtol: float = 1e-9,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    精确计算单枚烟幕弹在有效期内对目标的总遮蔽时长（对 shielding_intervals 的区间求和）。
    """
    intervals = shielding_intervals(missile_name, det_point, det_time, n_coarse, xtol)
    total = sum(b - a for a, b in intervals)
    ts = np.linspace(det_time, det_time + CLOUD_LIFE, n_coarse)
    g = shielding_margin(missile_name, det_point, det_time, ts)
    return total, ts, g <= 0


def union_length(interval_groups: list[list[tuple[float, float]]]) -> float:
    """
    把多枚弹各自的遮蔽区间列表合并求并集总长度（重叠部分不重复计入）。
    题目明确说"不同烟幕干扰弹的遮蔽可不连续"，所以多弹总遮蔽时长不能简单把各自
    时长相加——若两枚弹的区间有重叠，需要按并集去重。
    """
    all_intervals = sorted(interval for group in interval_groups for interval in group)
    if not all_intervals:
        return 0.0
    total = 0.0
    cur_start, cur_end = all_intervals[0]
    for a, b in all_intervals[1:]:
        if a > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = a, b
        else:
            cur_end = max(cur_end, b)
    total += cur_end - cur_start
    return total


def F_single_bomb(
    uav_name: str,
    missile_name: str,
    uav_speed: float,
    heading_rad: float,
    t_drop: float,
    t_fuze: float,
    n_coarse: int = 400,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    通用黑箱目标函数：X = (uav_speed, heading_rad, t_drop, t_fuze) -> 有效遮蔽时长(秒)
    """
    drop_point, det_point, det_time = bomb_detonation(
        uav_name, uav_speed, heading_rad, t_drop, t_fuze
    )
    if det_point[2] < 0:
        return 0.0, drop_point, det_point, det_time  # 起爆点在地面以下，不合理
    duration, _, _ = shielding_duration(
        missile_name, det_point, det_time, n_coarse=n_coarse
    )
    return duration, drop_point, det_point, det_time


def batch_quick_score(
    uav_name: str,
    missile_name: str,
    speeds: np.ndarray,
    headings: np.ndarray,
    t_drops: np.ndarray,
    t_fuzes: np.ndarray,
    n_t: int = 30,
) -> np.ndarray:
    """
    向量化粗筛：一批 (speed, heading, t_drop, t_fuze) 候选（四个参数都可以是标量或
    可广播的数组），估计一个与遮蔽时长成正比的粗略分数（有效期内被完全遮蔽的采样
    时刻个数）。不做精确求根，只用来快速排序/筛选，代价远低于调用 F_single_bomb。
    """
    speeds, headings, t_drops, t_fuzes = np.broadcast_arrays(
        speeds, headings, t_drops, t_fuzes
    )
    p0 = UAV_INIT[uav_name]
    vel_xy = speeds[:, None] * np.stack([np.cos(headings), np.sin(headings)], axis=-1)
    drop_xy = p0[:2] + vel_xy * t_drops[:, None]
    det_xy = drop_xy + vel_xy * t_fuzes[:, None]
    det_z = p0[2] - 0.5 * G * t_fuzes**2
    det_time = t_drops + t_fuzes

    M = len(speeds)
    taus = np.linspace(0.0, CLOUD_LIFE, n_t)
    ts = det_time[:, None] + taus[None, :]  # (M, n_t)

    m = missile_position(missile_name, ts)  # (M, n_t, 3)
    c_xy = np.broadcast_to(det_xy[:, None, :], (M, n_t, 2))
    c_z = det_z[:, None] - CLOUD_SINK_SPEED * taus[None, :]
    c = np.concatenate([c_xy, c_z[..., None]], axis=-1)  # (M, n_t, 3)

    dists = point_segment_distance(
        c[:, :, None, :], m[:, :, None, :], TARGET_POINTS[None, None, :, :]
    )  # (M, n_t, n_target)
    covered = dists.max(axis=-1) <= CLOUD_RADIUS  # (M, n_t)
    return covered.sum(axis=-1)  # (M,)
