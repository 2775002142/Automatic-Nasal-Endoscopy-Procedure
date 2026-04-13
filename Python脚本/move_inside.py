import sys
import os

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import math
import time
from windows.fairino import Robot
from enum import Enum, auto


# ==========================================
# 共享工具类
# ==========================================
class SystemState(Enum):
    """系统状态枚举 (体内导航 - 旋转优先版)"""
    IDLE = auto()
    BLIND_ENTRY = auto()
    ROTATE_ALIGN = auto()
    ADVANCE_Z = auto()
    TRANSITION_TO_ADVANCE = auto()  # 新增：从旋转到前进的平滑过渡
    TRANSITION_TO_ROTATE = auto()   # 新增：从前进到旋转的平滑过渡
    BLOCKED = auto()
    RETREAT = auto()
    MAX_DEPTH_REACHED = auto()
    SYSTEM_ERROR = auto()


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

# ==========================================
# 视觉处理系统 (基于APF算法) - 保持不变
# ==========================================
class APFVisionSystem:
    """
    基于人工势场法(APF)的视觉处理系统
    """
    def __init__(self, debug=False):
        self.K_ATT = 5.0
        self.K_REP = 3.0
        self.REP_RANGE = 80.0
        self.MIN_AREA_RATIO = 0.005

        self.BASE_DARK_PERCENTILE = 8
        self.BASE_BRIGHT_PERCENTILE = 80

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.debug = debug

        self.center = (0, 0)
        self.crop_offset = (0, 0)
        self.width = 0
        self.height = 0

    def process_frame(self, image):
        if image is None:
            return image, np.array([0.0, 0.0]), None

        image_cropped, self.crop_offset = self._crop_black_borders(image)
        self.height, self.width = image_cropped.shape[:2]
        self.center = (self.width // 2, self.height // 2)

        gray_image = self._preprocess_image(image_cropped)
        dark_regions, bright_regions = self._find_regions(gray_image)
        goal = self._select_goal(dark_regions)

        if goal:
            attractive_force = self._calculate_attractive_force(goal)
            repulsive_force = self._calculate_repulsive_force(bright_regions)
            total_force = attractive_force + repulsive_force
        else:
            total_force = np.array([0.0, 0.0])

        vis_image = self._visualize_result(image, total_force, goal, dark_regions, bright_regions)
        return vis_image, total_force, goal

    def _crop_black_borders(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, (0, 0)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        if cv2.contourArea(max_contour) / (image.shape[0] * image.shape[1]) < 0.1:
            return image, (0, 0)
        return image[y:y+h, x:x+w], (x, y)

    def _preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return cv2.GaussianBlur(enhanced, (7, 7), 0)

    def _find_regions(self, gray_img):
        dark_thresh = np.percentile(gray_img, self.BASE_DARK_PERCENTILE)
        bright_thresh = np.percentile(gray_img, self.BASE_BRIGHT_PERCENTILE)

        dark_mask = cv2.inRange(gray_img, 0, int(dark_thresh))
        bright_mask = cv2.inRange(gray_img, int(bright_thresh), 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        min_area = self.MIN_AREA_RATIO * self.height * self.width
        d_cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        b_cnts, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return ([c for c in d_cnts if cv2.contourArea(c) > min_area],
                [c for c in b_cnts if cv2.contourArea(c) > min_area])

    def _select_goal(self, regions):
        if not regions:
            return None
        best_goal, max_score = None, -1
        for region in regions:
            M = cv2.moments(region)
            if M['m00'] == 0:
                continue
            area = M['m00']
            cx, cy = int(M['m10'] / area), int(M['m01'] / area)
            score = (cy / self.height * 0.2) + (area / (self.width * self.height) * 0.8)
            if score > max_score:
                max_score, best_goal = score, (cx, cy)
        return best_goal

    def _calculate_attractive_force(self, goal):
        dx, dy = goal[0] - self.center[0], goal[1] - self.center[1]
        return self.K_ATT * np.array([dx, dy])

    def _calculate_repulsive_force(self, obstacles):
        total_rep = np.array([0.0, 0.0])
        center_pt = (float(self.center[0]), float(self.center[1]))

        for obs in obstacles:
            if cv2.pointPolygonTest(obs, center_pt, False) > 0:
                M = cv2.moments(obs)
                if M['m00'] == 0:
                    continue
                ox, oy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                vec = np.array([self.center[0] - ox, self.center[1] - oy])
                norm = np.linalg.norm(vec)
                if norm > 0:
                    total_rep += self.K_REP * 100 * (vec / norm)
                continue

            dist_sq_min = float('inf')
            closest_pt = None
            for pt_wrapper in obs:
                pt = pt_wrapper[0]
                d_sq = (self.center[0] - pt[0]) ** 2 + (self.center[1] - pt[1]) ** 2
                if d_sq < dist_sq_min:
                    dist_sq_min, closest_pt = d_sq, pt

            if closest_pt is None:
                continue
            dist = math.sqrt(dist_sq_min)
            if 0 < dist < self.REP_RANGE:
                vec = np.array([self.center[0] - closest_pt[0], self.center[1] - closest_pt[1]])
                mag = self.K_REP * ((1.0 / dist) - (1.0 / self.REP_RANGE)) * (1.0 / dist ** 2)
                total_rep += mag * (vec / dist)
        return total_rep

    def _visualize_result(self, original_img, force, goal, darks, brights, filtered_goal=None):
        vis = original_img.copy()
        ox, oy = self.crop_offset
        # 确保 center 不为 (0,0)
        if self.center[0] == 0 and self.center[1] == 0:
            return vis
            
        cx, cy = self.center[0] + ox, self.center[1] + oy

        # 中心点 (黄色)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

        # 原始检测目标点 (橙色空心圆)
        if goal is not None:
            gx_abs = goal[0] + ox
            gy_abs = goal[1] + oy
            cv2.circle(vis, (gx_abs, gy_abs), 8, (0, 165, 255), 2)

        # 卡尔曼滤波后的目标点 (绿色实心圆) - 这是我们主要跟随的点
        if filtered_goal is not None:
            fgx_abs = int(filtered_goal[0] + ox)
            fgy_abs = int(filtered_goal[1] + oy)
            cv2.circle(vis, (fgx_abs, fgy_abs), 7, (0, 255, 0), -1)

        darks_abs = [c + (ox, oy) for c in darks]
        brights_abs = [c + (ox, oy) for c in brights]
        cv2.drawContours(vis, darks_abs, -1, (255, 0, 0), 1)
        cv2.drawContours(vis, brights_abs, -1, (0, 0, 255), 1)

        # 力场向量
        if np.linalg.norm(force) > 1e-4:
            unit_vec = force / np.linalg.norm(force)
            end_pt = (int(cx + unit_vec[0] * 50), int(cy + unit_vec[1] * 50))
            cv2.arrowedLine(vis, (cx, cy), end_pt, (0, 255, 0), 2)

        return vis


# ==========================================
# 机械臂控制系统
# ==========================================
class RobotController:
    def __init__(self, ip='192.168.58.2', simulate=False):
        self.ip = ip
        self.simulate = simulate
        self.robot = None
        self.connected = False
        self.tool_id = 1
        self.user_id = 0
        self.get_pose_fail_count = 0
        self.MAX_GET_POSE_FAILURES = 5

    def connect(self):
        if self.simulate:
            print("[Robot] 模拟模式")
            self.connected = True
            return True
        try:
            self.robot = Robot.RPC(self.ip)
            print(f"[Robot] 已连接: {self.ip}")
            self.connected = True
            return True
        except Exception as e:
            print(f"[Robot] 连接异常: {e}")
            self.connected = False
            return False

    def _get_current_pose(self):
        """获取当前TCP位姿"""
        raw_result = self.robot.GetActualTCPPose(0)
        if isinstance(raw_result, int):
            return None
        ret, current_pose = raw_result
        if ret != 0 or current_pose is None or len(current_pose) != 6:
            return None
        try:
            valid_pose = [float(v) for v in current_pose]
            if any(math.isnan(v) or math.isinf(v) for v in valid_pose):
                return None
            return valid_pose
        except (TypeError, ValueError):
            return None

    def move_z_only(self, dz, max_retries=3):
        """纯Z轴平移"""
        if not self.connected:
            return False
        if self.simulate:
            time.sleep(0.05)
            print(f"[Sim] Z轴移动 -> dz:{dz:.2f}mm")
            return True

        for attempt in range(max_retries):
            try:
                current_pose = self._get_current_pose()
                if current_pose is None:
                    self.get_pose_fail_count += 1
                    time.sleep(0.5)
                    continue
                self.get_pose_fail_count = 0
                offset_pos = [0.0, 0.0, float(dz), 0.0, 0.0, 0.0]
                ret = self.robot.MoveL(
                    desc_pos=current_pose,
                    tool=self.tool_id,
                    user=self.user_id,
                    vel=5.0,
                    acc=20.0,
                    offset_flag=2,
                    offset_pos=offset_pos
                )
                if ret == 0:
                    return True
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

        print(f"[Err] Z轴移动失败，已重试{max_retries}次")
        return False

    def move_xy(self, dx_mm, dy_mm, max_retries=3):
        """
        【新增】纯XY平移（工具坐标系）
        用于将旋转与平移解耦，确保各步骤失败可独立记录。

        Args:
            dx_mm: 沿X轴平移(mm), +X=右
            dy_mm: 沿Y轴平移(mm), +Y=下
        """
        if not self.connected:
            return False
        if abs(dx_mm) < 1e-6 and abs(dy_mm) < 1e-6:
            return True  # 无需移动，直接成功
        if self.simulate:
            time.sleep(0.05)
            print(f"[Sim] XY平移 -> dx:{dx_mm:.3f}mm dy:{dy_mm:.3f}mm")
            return True

        for attempt in range(max_retries):
            try:
                current_pose = self._get_current_pose()
                if current_pose is None:
                    time.sleep(0.5)
                    continue
                offset_pos = [float(dx_mm), float(dy_mm), 0.0, 0.0, 0.0, 0.0]
                ret = self.robot.MoveL(
                    desc_pos=current_pose,
                    tool=self.tool_id,
                    user=self.user_id,
                    vel=5.0,
                    acc=20.0,
                    offset_flag=2,
                    offset_pos=offset_pos
                )
                if ret == 0:
                    return True
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

        print(f"[Err] XY平移失败，已重试{max_retries}次")
        return False

    def rotate_tool_frame(self, rx_deg, ry_deg, max_retries=3):
        """纯姿态旋转"""
        if not self.connected:
            return False
        if self.simulate:
            time.sleep(0.05)
            print(f"[Sim] 旋转对准 -> Rx:{rx_deg:.3f}° Ry:{ry_deg:.3f}°")
            return True

        for attempt in range(max_retries):
            try:
                current_pose = self._get_current_pose()
                if current_pose is None:
                    time.sleep(0.5)
                    continue
                offset_pos = [0.0, 0.0, 0.0, float(rx_deg), float(ry_deg), 0.0]
                ret = self.robot.MoveL(
                    desc_pos=current_pose,
                    tool=self.tool_id,
                    user=self.user_id,
                    vel=5.0,
                    acc=20.0,
                    offset_flag=2,
                    offset_pos=offset_pos
                )
                if ret == 0:
                    return True
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

        print(f"[Err] 旋转失败，已重试{max_retries}次")
        return False

    def rotate_and_translate(self, rx_deg, ry_deg, dx_mm, dy_mm, max_retries=3):
        """
        旋转+平移联合接口（保留向后兼容，内部调用独立方法）
        注意：主循环中推荐直接调用 rotate_tool_frame + move_xy 以获得
        更精确的失败分离记录，此方法仅供外部调用或向后兼容。
        """
        if not self.connected:
            return False
        if self.simulate:
            time.sleep(0.05)
            print(f"[Sim] 旋转+平移 -> Rx:{rx_deg:.3f}° Ry:{ry_deg:.3f}° "
                  f"dx:{dx_mm:.3f}mm dy:{dy_mm:.3f}mm")
            return True

        rot_ok = True
        if abs(rx_deg) > 1e-6 or abs(ry_deg) > 1e-6:
            rot_ok = self.rotate_tool_frame(rx_deg, ry_deg, max_retries)
            if rot_ok:
                time.sleep(0.05)

        trans_ok = True
        if abs(dx_mm) > 1e-6 or abs(dy_mm) > 1e-6:
            trans_ok = self.move_xy(dx_mm, dy_mm, max_retries)

        return rot_ok and trans_ok

    def disconnect(self):
        if self.simulate:
            return
        if self.robot is not None:
            try:
                self.robot.ResetAllError()
                self.robot = None
            except:
                pass
        self.connected = False


# ==========================================
# 力场向量 → 旋转角度 + 平移量 转换器
# ==========================================
class ForceToMotionConverter:
    """
    将APF输出的像素级力场向量转换为工具坐标系下的旋转角度和平移量
    
    核心原理：
    APF力场输出的 force = [fx, fy] 表示"目标在画面中心的哪个方向"
      - fx > 0: 目标偏右 → 镜头需要向右转 → Ry 取负值；同时向右平移 → dx > 0
      - fy > 0: 目标偏下 → 镜头需要向下俯 → Rx 取正值；同时向下平移 → dy > 0
    
    旋转：将像素偏差映射到极小的旋转角度
    平移：将力场方向映射到固定步长的XY平移（辅助校准）
    """
    def __init__(self,
                 force_deadzone=10.0,
                 max_force_for_scale=1150.0, # 假设力场幅度在此范围内达到最大增益
                 min_rotation_gain=0.001,
                 max_rotation_gain=0.012,
                 rotation_gain_curve_factor=0.6,
                 max_rotation_deg=0.5,
                 min_translate_step_mm=0.02,
                 max_translate_step_mm=0.15,
                 translate_step_curve_factor=0.7,
                 max_translate_per_phase_mm=1.0):
        """
        Args:
            force_deadzone: 力场死区，低于此值不动作（防抖）
            max_rotation_deg: 单次最大旋转角度（度），安全限幅
            gain: 力场到角度的转换增益 (度/力场单位)
            translate_step_mm: 单次平移步长(mm)
            max_translate_per_phase_mm: 每个旋转阶段最大平移累计(mm)
        """
        self.force_deadzone = force_deadzone
        self.max_force_for_scale = max_force_for_scale
        self.min_rotation_gain = min_rotation_gain
        self.max_rotation_gain = max_rotation_gain
        self.rotation_gain_curve_factor = rotation_gain_curve_factor
        self.max_rotation_deg = max_rotation_deg
        self.min_translate_step_mm = min_translate_step_mm
        self.max_translate_step_mm = max_translate_step_mm
        self.translate_step_curve_factor = translate_step_curve_factor
        self.max_translate_per_phase_mm = max_translate_per_phase_mm

    def _calculate_dynamic_params(self, force_mag):
        if force_mag < self.force_deadzone:
            return 0.0, 0.0

        norm_force = clamp((force_mag - self.force_deadzone) / (self.max_force_for_scale - self.force_deadzone), 0.0, 1.0)
        
        rot_gain = self.min_rotation_gain + (self.max_rotation_gain - self.min_rotation_gain) * (norm_force ** self.rotation_gain_curve_factor)
        trans_step = self.min_translate_step_mm + (self.max_translate_step_mm - self.min_translate_step_mm) * (norm_force ** self.translate_step_curve_factor)
        
        return rot_gain, trans_step

    def convert(self, force_vector, current_phase_dx, current_phase_dy):
        fx, fy = float(force_vector[0]), float(force_vector[1])
        force_mag = math.sqrt(fx**2 + fy**2)

        dyn_rot_gain, dyn_trans_step = self._calculate_dynamic_params(force_mag)
        if dyn_rot_gain == 0.0 and dyn_trans_step == 0.0:
            return 0.0, 0.0, 0.0, 0.0

        # !! 重要: 旋转正负号需根据实际机器人和相机安装测试 !!
        # 假设: +fy(目标偏下) -> -Rx(下俯), +fx(目标偏右) -> +Ry(右转)
        rx_deg = -fy * dyn_rot_gain
        ry_deg = fx * dyn_rot_gain
        rx_deg = clamp(rx_deg, -self.max_rotation_deg, self.max_rotation_deg)
        ry_deg = clamp(ry_deg, -self.max_rotation_deg, self.max_rotation_deg)

        dx_mm, dy_mm = 0.0, 0.0
        if force_mag > 1e-6:
            dir_x, dir_y = fx / force_mag, fy / force_mag
            dx_cand = dir_x * dyn_trans_step
            dy_cand = dir_y * dyn_trans_step
            if abs(current_phase_dx + dx_cand) <= self.max_translate_per_phase_mm:
                dx_mm = dx_cand
            if abs(current_phase_dy + dy_cand) <= self.max_translate_per_phase_mm:
                dy_mm = dy_cand
        
        return rx_deg, ry_deg, dx_mm, dy_mm


# ==========================================
# 主程序逻辑
# ==========================================
# ==========================================
# 主程序逻辑
# ==========================================
def main():
    # -------- 核心调优参数 --------
    VIDEO_SOURCE = 1
    MAX_DEPTH_MM = 50.0
    SIMULATE_ROBOT = False

    # -------- 状态切换判据 (滞回防抖) --------
    ALIGN_DIST_START_PX = 150.0
    ALIGN_DIST_STOP_PX  = 130.0
    MIN_ROTATE_FRAMES   = 2
    MIN_ADVANCE_FRAMES  = 5
    TRANSITION_FRAMES   = 10

    # -------- Z轴前进/后退参数 --------
    Z_ADVANCE_STEP_MM = 2.0
    RETREAT_STEP_MM   = -2.0   # 负值 = 后退

    # -------- 丢失与防撞保护 --------
    BLOCKED_TIMEOUT_SEC = 1.5
    EMA_ALPHA           = 0.25
    KALMAN_PROCESS_NOISE  = 0.05
    KALMAN_MEASURE_NOISE  = 4.0

    # -------- 盲走阶段参数 --------
    BLIND_ENTRY_STEP_MM        = 0.5
    BLIND_ENTRY_MAX_MM         = 20.0
    BLIND_ENTRY_INTERVAL_SEC   = 0.15
    BLIND_GOAL_CONFIRM_FRAMES  = 20

    # -------- 单阶段旋转安全限制 --------
    MAX_TOTAL_ROTATION_DEG = 1.0

    # -------- 初始化 --------
    vision    = APFVisionSystem(debug=False)
    bot       = RobotController(simulate=SIMULATE_ROBOT)
    converter = ForceToMotionConverter()

    if not bot.connect():
        print("无法连接机械臂，程序退出。")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("无法打开摄像头。")
        bot.disconnect()
        return

    cv2.namedWindow("Internal Endoscope Control", cv2.WINDOW_NORMAL)

    # -------- 状态与计数器 --------
    auto_run      = False
    current_state = SystemState.IDLE
    current_depth = 0.0

    goal_kalman_filter = KalmanFilter2D(
        process_noise_cov=KALMAN_PROCESS_NOISE,
        measurement_noise_cov=KALMAN_MEASURE_NOISE
    )
    force_x_filter = EMAFilter(alpha=EMA_ALPHA)
    force_y_filter = EMAFilter(alpha=EMA_ALPHA)
    dist_filter    = EMAFilter(alpha=EMA_ALPHA)
    blocked_start_time = None

    # 盲走变量
    blind_entry_completed   = False
    blind_entry_distance    = 0.0
    blind_goal_consecutive  = 0
    blind_last_step_time    = 0.0

    # 累计量（旋转 + 平移）
    total_rx_deg          = 0.0
    total_ry_deg          = 0.0
    total_translation_dx  = 0.0
    total_translation_dy  = 0.0

    # 当前旋转阶段累计（用于旋转安全限制）
    current_rotation_rx   = 0.0
    current_rotation_ry   = 0.0
    current_translation_dx = 0.0
    current_translation_dy = 0.0

    state_frame_counter = 0

    print("=" * 60)
    print(">>> 体内导航系统 (优化版：卡尔曼 + 动态控制 + 平滑切换) <<<")
    print(f"  [旋转增益] {converter.min_rotation_gain:.4f} ~ {converter.max_rotation_gain:.4f} | "
          f"单次上限={converter.max_rotation_deg}°")
    print(f"  [平移步长] {converter.min_translate_step_mm:.3f} ~ {converter.max_translate_step_mm:.3f}mm | "
          f"每阶段上限={converter.max_translate_per_phase_mm}mm")
    print(f"  [切换] 偏离>{ALIGN_DIST_START_PX}px开始校准 | 偏离<{ALIGN_DIST_STOP_PX}px允许直行")
    print(f"  [防抖] 旋转最少{MIN_ROTATE_FRAMES}帧 | 前进最少{MIN_ADVANCE_FRAMES}帧 | 过渡{TRANSITION_FRAMES}帧")
    print(f"  [安全] 单阶段旋转上限={MAX_TOTAL_ROTATION_DEG}° | 最大深度={MAX_DEPTH_MM}mm")
    print("  [操作] 空格: 启动/暂停 | R: 重置深度上限 | Q/ESC: 退出")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ==================================================
            # 1. 视觉处理 & 卡尔曼滤波目标点
            # ==================================================
            vis_img, raw_force, raw_goal = vision.process_frame(frame)

            goal = None
            if raw_goal is not None:
                filtered_pos = goal_kalman_filter.update(raw_goal)
                goal = (int(filtered_pos[0]), int(filtered_pos[1]))
            else:
                goal_kalman_filter.reset()

            # ==================================================
            # 2. 力场与距离计算（基于滤波后目标）
            # ==================================================
            pixel_dist = 0.0
            if goal is not None:
                # 用滤波后目标点重建吸引力，叠加原始排斥力
                att_filtered = vision._calculate_attractive_force(goal)
                if raw_goal is not None:
                    att_raw = vision._calculate_attractive_force(raw_goal)
                    rep_force = raw_force - att_raw
                else:
                    rep_force = np.array([0.0, 0.0])
                combined_force = att_filtered + rep_force

                fx = float(force_x_filter.update(combined_force[0]))
                fy = float(force_y_filter.update(combined_force[1]))
                filtered_force = np.array([fx, fy])

                raw_dist  = math.hypot(
                    goal[0] - vision.center[0],
                    goal[1] - vision.center[1]
                )
                pixel_dist = float(dist_filter.update(raw_dist))
                blocked_start_time = None
            else:
                filtered_force = np.array([0.0, 0.0])
                force_x_filter.reset()
                force_y_filter.reset()

            force_mag = np.linalg.norm(filtered_force)

            # ==================================================
            # 3. 状态机评估
            # ==================================================
            new_state   = current_state
            action_type = None
            rx_cmd = ry_cmd = dx_cmd = dy_cmd = dz_cmd = 0.0
            status_msg  = ""

            if not auto_run:
                new_state  = SystemState.IDLE
                status_msg = "IDLE (已暂停)"

            elif not blind_entry_completed:
                # ===== 盲走阶段 =====
                new_state = SystemState.BLIND_ENTRY

                if goal is not None:
                    blind_goal_consecutive += 1
                else:
                    blind_goal_consecutive = 0

                if blind_goal_consecutive >= BLIND_GOAL_CONFIRM_FRAMES:
                    blind_entry_completed = True
                    print(f"[盲走完成] 连续{BLIND_GOAL_CONFIRM_FRAMES}帧检测到目标，切换正常导航。")
                    force_x_filter.reset()
                    force_y_filter.reset()
                    current_state = SystemState.TRANSITION_TO_ROTATE
                    new_state     = SystemState.TRANSITION_TO_ROTATE
                    state_frame_counter = 0
                    current_rotation_rx = current_rotation_ry = 0.0
                    current_translation_dx = current_translation_dy = 0.0
                    status_msg = "Blind->Transition to Rotate"
                    continue

                elif blind_entry_distance >= BLIND_ENTRY_MAX_MM:
                    blind_entry_completed = True
                    print(f"[盲走上限] 已盲走{blind_entry_distance:.1f}mm，强制切换。")
                    force_x_filter.reset()
                    force_y_filter.reset()
                    current_state = SystemState.TRANSITION_TO_ROTATE
                    new_state     = SystemState.TRANSITION_TO_ROTATE
                    state_frame_counter = 0
                    current_rotation_rx = current_rotation_ry = 0.0
                    current_translation_dx = current_translation_dy = 0.0
                    status_msg = "Blind Max->Transition to Rotate"
                    continue

                else:
                    now = time.time()
                    if now - blind_last_step_time >= BLIND_ENTRY_INTERVAL_SEC:
                        action_type = 'blind_z'
                        dz_cmd = BLIND_ENTRY_STEP_MM
                        blind_last_step_time = now
                    status_msg = f"BLIND_ENTRY {blind_entry_distance:.1f}/{BLIND_ENTRY_MAX_MM}mm"

            else:
                # ===== 正常导航阶段 =====

                # [FIX 问题2] rotation_safe 仅在旋转相关状态下限制
                current_rot = math.sqrt(current_rotation_rx**2 + current_rotation_ry**2)
                if current_state in [SystemState.ROTATE_ALIGN, SystemState.TRANSITION_TO_ROTATE]:
                    rotation_safe = current_rot < MAX_TOTAL_ROTATION_DEG
                else:
                    rotation_safe = True  # 非旋转状态始终允许重新启动旋转

                # --- 最高优先级判断 ---

                # (a) 达到最大深度
                if current_depth >= MAX_DEPTH_MM:
                    new_state  = SystemState.MAX_DEPTH_REACHED
                    status_msg = f"MAX DEPTH {current_depth:.1f}mm (按R重置)"

                # (b) 目标丢失 → BLOCKED / RETREAT
                elif goal is None:
                    if blocked_start_time is None:
                        blocked_start_time = time.time()
                    elapsed = time.time() - blocked_start_time
                    if elapsed > BLOCKED_TIMEOUT_SEC:
                        new_state   = SystemState.RETREAT
                        action_type = 'retreat'
                        dz_cmd      = RETREAT_STEP_MM
                        status_msg  = f"RETREAT elapsed={elapsed:.1f}s"
                    else:
                        new_state  = SystemState.BLOCKED
                        status_msg = f"BLOCKED {elapsed:.1f}/{BLOCKED_TIMEOUT_SEC}s"

                # (c) 正常滞回切换
                else:
                    if current_state in [SystemState.ADVANCE_Z, SystemState.IDLE]:
                        if (pixel_dist > ALIGN_DIST_START_PX
                                and rotation_safe
                                and state_frame_counter >= MIN_ADVANCE_FRAMES):
                            new_state = SystemState.TRANSITION_TO_ROTATE
                        else:
                            new_state   = SystemState.ADVANCE_Z
                            action_type = 'advance_z'
                            dz_cmd      = Z_ADVANCE_STEP_MM
                            status_msg  = f"ADVANCE D={pixel_dist:.1f} F={force_mag:.1f}"

                    elif current_state == SystemState.ROTATE_ALIGN:
                        if ((pixel_dist < ALIGN_DIST_STOP_PX or not rotation_safe)
                                and state_frame_counter >= MIN_ROTATE_FRAMES):
                            new_state = SystemState.TRANSITION_TO_ADVANCE
                        else:
                            new_state   = SystemState.ROTATE_ALIGN
                            action_type = 'rotate_translate'
                            rx_cmd, ry_cmd, dx_cmd, dy_cmd = converter.convert(
                                filtered_force,
                                current_translation_dx,
                                current_translation_dy
                            )
                            status_msg = (f"ROTATE D={pixel_dist:.1f} "
                                          f"Rx={rx_cmd:.3f}° Ry={ry_cmd:.3f}° "
                                          f"dx={dx_cmd:.3f} dy={dy_cmd:.3f}")

                    elif current_state == SystemState.TRANSITION_TO_ROTATE:
                        pass  # 由下方过渡逻辑处理

                    elif current_state == SystemState.TRANSITION_TO_ADVANCE:
                        pass  # 由下方过渡逻辑处理

                    else:
                        # 从 BLOCKED/RETREAT/其他状态恢复
                        new_state = SystemState.TRANSITION_TO_ROTATE

            # ==================================================
            # 状态切换：重置计数器与累积量
            # ==================================================
            if new_state != current_state:
                current_state       = new_state
                state_frame_counter = 0

                # [FIX 问题3] 进入旋转状态时重置旋转/平移累积
                if current_state in [SystemState.TRANSITION_TO_ROTATE,
                                     SystemState.ROTATE_ALIGN]:
                    current_rotation_rx    = 0.0
                    current_rotation_ry    = 0.0
                    current_translation_dx = 0.0
                    current_translation_dy = 0.0

                # [FIX 问题3] 正式进入前进状态时也清空（过渡期保留）
                if current_state == SystemState.ADVANCE_Z:
                    current_rotation_rx    = 0.0
                    current_rotation_ry    = 0.0
                    current_translation_dx = 0.0
                    current_translation_dy = 0.0
            else:
                state_frame_counter += 1

            # ==================================================
            # 过渡状态的运动指令计算
            # ==================================================
            if current_state == SystemState.TRANSITION_TO_ROTATE:
                ratio = min(state_frame_counter / max(TRANSITION_FRAMES, 1), 1.0)
                t_rx, t_ry, t_dx, t_dy = converter.convert(
                    filtered_force,
                    current_translation_dx,
                    current_translation_dy
                )
                rx_cmd = t_rx * ratio
                ry_cmd = t_ry * ratio
                dx_cmd = t_dx * ratio
                dy_cmd = t_dy * ratio
                action_type = 'rotate_translate'
                status_msg  = f"Transition→ROTATE ({state_frame_counter}/{TRANSITION_FRAMES})"
                if state_frame_counter >= TRANSITION_FRAMES:
                    current_state       = SystemState.ROTATE_ALIGN
                    state_frame_counter = 0

            elif current_state == SystemState.TRANSITION_TO_ADVANCE:
                ratio = min(state_frame_counter / max(TRANSITION_FRAMES, 1), 1.0)
                t_rx, t_ry, t_dx, t_dy = converter.convert(
                    filtered_force,
                    current_translation_dx,
                    current_translation_dy
                )
                # 旋转/平移线性淡出，Z轴线性淡入
                rx_cmd = t_rx * (1.0 - ratio)
                ry_cmd = t_ry * (1.0 - ratio)
                dx_cmd = t_dx * (1.0 - ratio)
                dy_cmd = t_dy * (1.0 - ratio)
                dz_cmd = Z_ADVANCE_STEP_MM * ratio
                action_type = 'all'
                status_msg  = f"Transition→ADVANCE ({state_frame_counter}/{TRANSITION_FRAMES})"
                if state_frame_counter >= TRANSITION_FRAMES:
                    current_state       = SystemState.ADVANCE_Z
                    state_frame_counter = 0
                    # [FIX 问题3] 过渡完成正式进入前进，清空累积量
                    current_rotation_rx    = 0.0
                    current_rotation_ry    = 0.0
                    current_translation_dx = 0.0
                    current_translation_dy = 0.0

            # ==================================================
            # 4. 执行动作  [FIX 问题4：拆分旋转与平移，独立更新累积]
            # ==================================================
            if auto_run:

                if action_type == 'blind_z':
                    if bot.move_z_only(dz_cmd):
                        blind_entry_distance += abs(dz_cmd)
                        current_depth        += abs(dz_cmd)

                elif action_type == 'rotate_translate':
                    # 旋转：独立执行，成功后记录
                    if abs(rx_cmd) > 1e-6 or abs(ry_cmd) > 1e-6:
                        if bot.rotate_tool_frame(rx_cmd, ry_cmd):
                            current_rotation_rx += abs(rx_cmd)
                            current_rotation_ry += abs(ry_cmd)
                            total_rx_deg        += abs(rx_cmd)
                            total_ry_deg        += abs(ry_cmd)
                    # 平移：独立执行，成功后记录
                    if abs(dx_cmd) > 1e-6 or abs(dy_cmd) > 1e-6:
                        if bot.move_xy(dx_cmd, dy_cmd):
                            current_translation_dx += dx_cmd
                            current_translation_dy += dy_cmd
                            total_translation_dx   += dx_cmd
                            total_translation_dy   += dy_cmd

                elif action_type == 'advance_z':
                    if bot.move_z_only(dz_cmd):
                        current_depth += abs(dz_cmd)

                elif action_type == 'retreat':
                    if bot.move_z_only(dz_cmd):
                        current_depth      = max(0.0, current_depth + dz_cmd)
                        blocked_start_time = None

                elif action_type == 'all':
                    # 过渡状态：旋转平移 + Z轴分别执行，独立记录
                    if abs(rx_cmd) > 1e-6 or abs(ry_cmd) > 1e-6:
                        if bot.rotate_tool_frame(rx_cmd, ry_cmd):
                            current_rotation_rx += abs(rx_cmd)
                            current_rotation_ry += abs(ry_cmd)
                            total_rx_deg        += abs(rx_cmd)
                            total_ry_deg        += abs(ry_cmd)
                    if abs(dx_cmd) > 1e-6 or abs(dy_cmd) > 1e-6:
                        if bot.move_xy(dx_cmd, dy_cmd):
                            current_translation_dx += dx_cmd
                            current_translation_dy += dy_cmd
                            total_translation_dx   += dx_cmd
                            total_translation_dy   += dy_cmd
                    if abs(dz_cmd) > 1e-6:
                        if bot.move_z_only(dz_cmd):
                            current_depth += abs(dz_cmd)

            # ==================================================
            # 5. 可视化叠加
            # ==================================================
            if vis_img is not None:
                vis_img = vision._visualize_result(
                    vis_img, filtered_force,
                    raw_goal, [], [],
                    filtered_goal=goal
                )
                h_img, w_img = vis_img.shape[:2]

                # 状态与深度
                cv2.putText(vis_img,
                            f"State: {current_state.name}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis_img,
                            f"Depth: {current_depth:.1f}/{MAX_DEPTH_MM}mm",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis_img,
                            f"Dist: {pixel_dist:.1f}px  F: {force_mag:.1f}",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis_img,
                            status_msg,
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
                cv2.putText(vis_img,
                            status_msg,
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

                # 旋转累积信息
                cv2.putText(vis_img,
                            f"Rot: Rx={current_rotation_rx:.3f}° Ry={current_rotation_ry:.3f}° "
                            f"(max={MAX_TOTAL_ROTATION_DEG}°)",
                            (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

                # 平移累积信息
                cv2.putText(vis_img,
                            f"Trans: dx={current_translation_dx:.3f}mm "
                            f"dy={current_translation_dy:.3f}mm",
                            (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

                # 总累积信息
                cv2.putText(vis_img,
                            f"Total: Rx={total_rx_deg:.2f}° Ry={total_ry_deg:.2f}° "
                            f"dx={total_translation_dx:.2f}mm dy={total_translation_dy:.2f}mm",
                            (10, 171), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)

                # 运行状态指示
                run_color = (0, 255, 0) if auto_run else (0, 0, 255)
                run_text  = "AUTO" if auto_run else "PAUSED"
                cv2.putText(vis_img,
                            run_text,
                            (w_img - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, run_color, 2)

                # 深度进度条
                bar_x, bar_y, bar_w, bar_h = 10, h_img - 20, w_img - 20, 12
                cv2.rectangle(vis_img, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
                fill_w = int(bar_w * min(current_depth / max(MAX_DEPTH_MM, 1e-6), 1.0))
                bar_color = (0, 200, 255) if current_depth < MAX_DEPTH_MM * 0.8 else (0, 80, 255)
                cv2.rectangle(vis_img, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
                cv2.rectangle(vis_img, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

                # MAX_DEPTH_REACHED 警告
                if current_state == SystemState.MAX_DEPTH_REACHED:
                    cv2.putText(vis_img,
                                "MAX DEPTH REACHED  Press R to reset",
                                (w_img // 2 - 200, h_img // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Internal Endoscope Control", vis_img)

            # ==================================================
            # 6. 键盘处理
            # ==================================================
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # Q 或 ESC：退出
                print("用户退出。")
                break

            elif key == ord(' '):  # 空格：启动 / 暂停
                auto_run = not auto_run
                if auto_run:
                    # 启动时重置所有累积量与状态
                    current_state          = SystemState.IDLE
                    state_frame_counter    = 0
                    current_depth          = 0.0
                    blind_entry_completed  = False
                    blind_entry_distance   = 0.0
                    blind_goal_consecutive = 0
                    blind_last_step_time   = 0.0
                    total_rx_deg           = 0.0
                    total_ry_deg           = 0.0
                    total_translation_dx   = 0.0
                    total_translation_dy   = 0.0
                    current_rotation_rx    = 0.0
                    current_rotation_ry    = 0.0
                    current_translation_dx = 0.0
                    current_translation_dy = 0.0
                    blocked_start_time     = None
                    goal_kalman_filter.reset()
                    force_x_filter.reset()
                    force_y_filter.reset()
                    dist_filter.reset()
                    print(">>> 自动导航已启动 <<<")
                else:
                    print(">>> 自动导航已暂停 <<<")

            elif key == ord('r') or key == ord('R'):
                # [FIX 问题5] R 键：重置深度上限，允许继续导航
                if current_state == SystemState.MAX_DEPTH_REACHED:
                    current_depth  = 0.0
                    current_state  = SystemState.IDLE
                    state_frame_counter = 0
                    print(">>> 深度上限已手动重置，系统回到 IDLE <<<")
                else:
                    # 非上限状态下按 R：仅重置深度计数（方便调试）
                    current_depth = 0.0
                    print(f">>> 深度计数已重置为 0 <<<")

    except KeyboardInterrupt:
        print("\n键盘中断，正在退出...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        bot.disconnect()
        print("系统已安全关闭。")
        print(f"[统计] 总深度={current_depth:.1f}mm | "
              f"总旋转 Rx={total_rx_deg:.3f}° Ry={total_ry_deg:.3f}° | "
              f"总平移 dx={total_translation_dx:.3f}mm dy={total_translation_dy:.3f}mm")


if __name__ == "__main__":
    main()