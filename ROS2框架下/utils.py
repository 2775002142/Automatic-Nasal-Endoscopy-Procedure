# 文件路径: ~/FR5/src/fr5_vision_control/fr5_vision_control/utils.py

import numpy as np
from enum import Enum, auto

class SystemState(Enum):
    """系统状态枚举 (合并了 Outside 和 Inside 的所有状态)"""
    IDLE = auto()              # 空闲状态
    
    # === 体外部分 (Outside) ===
    ALIGN_XY = auto()          # XY对齐状态
    APPROACH_Z = auto()        # Z轴逼近状态
    TRANSITION_TO_APPROACH = auto()   # 从对齐向进近过渡
    TRANSITION_TO_ALIGN = auto()      # 从进近向对齐过渡
    TARGET_LOST = auto()       # 目标丢失状态
    RETREAT = auto()           # 后退重试状态
    TARGET_REACHED = auto()    # 目标到达状态
    
    # === 体内部分 (Inside) ===
    BLIND_ENTRY = auto()       # 盲走进洞状态
    ROTATE_ALIGN = auto()      # 旋转对准状态
    ADVANCE_Z = auto()         # 沿Z轴前进状态
    TRANSITION_TO_ADVANCE = auto()  # 新增：从旋转到前进的平滑过渡
    TRANSITION_TO_ROTATE = auto()   # 新增：从前进到旋转的平滑过渡
    BLOCKED = auto()           # 遇阻状态
    MAX_DEPTH_REACHED = auto() # 达到最大深度限制
    
    # === 通用部分 ===
    SYSTEM_ERROR = auto()      # 系统错误状态
    TERMINATED = auto()        # 程序终止状态


def clamp(val, vmin, vmax):
    return max(vmin, min(vmax, val))


class EMAFilter:
    """简单指数滑动平均滤波器"""
    def __init__(self, alpha=0.3):
        self.alpha = float(alpha)
        self.inited = False
        self.value = None

    def reset(self):
        self.inited = False
        self.value = None

    def update(self, x):
        x = np.array(x, dtype=np.float32)
        if not self.inited:
            self.value = x
            self.inited = True
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * x
        return self.value
    
class KalmanFilter2D:
    """
    用于二维点(x, y)的简单卡尔曼滤波器。
    状态向量: [x, y, vx, vy] (位置和速度)
    观测向量: [x, y]
    """
    def __init__(self, dt=1.0, process_noise_cov=1e-2, measurement_noise_cov=1e0):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]) # 状态转移矩阵
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # 观测矩阵
        self.Q = process_noise_cov * np.eye(4) # 过程噪声协方差
        self.R = measurement_noise_cov * np.eye(2) # 测量噪声协方差
        self.P = np.eye(4) * 1.0 # 状态协方差
        self.x_hat = np.zeros((4, 1)) # 状态向量
        self.initialized = False

    def reset(self):
        self.initialized = False
        self.x_hat = np.zeros((4, 1))
        self.P = np.eye(4) * 1.0

    def update(self, measurement):
        measurement = np.array(measurement).reshape(2, 1)
        if not self.initialized:
            self.x_hat[0:2] = measurement
            self.initialized = True
            return self.x_hat[0:2].flatten()

        # 预测
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q

        # 更新
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = measurement - (self.H @ self.x_hat)
        self.x_hat = self.x_hat + (K @ y)
        I = np.eye(self.A.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.x_hat[0:2].flatten()