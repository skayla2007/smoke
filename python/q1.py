"""
问题1：给定具体参数，计算 FY1 投放的 1 枚烟幕干扰弹对 M1 的有效遮蔽时长。
FY1 以 120 m/s 朝假目标方向飞行，受领任务 1.5s 后投放，间隔 3.6s 后起爆。
"""

import numpy as np

from model import F_single_bomb

if __name__ == "__main__":
    # FY1 在 (17800,0)，假目标在原点 -> 方向沿 -x 轴，heading = pi
    heading = np.pi
    speed = 120.0
    t_drop = 1.5
    t_fuze = 3.6

    duration, drop_point, det_point, det_time = F_single_bomb(
        "FY1", "M1", speed, heading, t_drop, t_fuze
    )

    print("投放点:", drop_point)
    print("起爆点:", det_point)
    print("起爆时刻(相对任务下达): %.3f s" % det_time)
    print("有效遮蔽时长: %.4f s" % duration)
