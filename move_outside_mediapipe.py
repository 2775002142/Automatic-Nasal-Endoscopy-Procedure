
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import mediapipe as mp
import time
import math
import numpy as np
from windows.fairino import Robot
from enum import Enum, auto

class SystemState(Enum):
    """系统状态枚举"""
    IDLE = auto()              # 空闲状态
    ALIGN_XY = auto()          # XY对齐状态
    APPROACH_Z = auto()        # Z轴逼近状态
    TARGET_LOST = auto()       # 目标丢失状态
    RETREAT = auto()           # 后退重试状态
    TARGET_REACHED = auto()    # 目标到达状态
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


class VisionSystem:
    """
    视觉处理系统
    功能：去除黑边 -> 提取有效画面 -> 识别鼻孔目标
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[Vision] MediaPipe 模型已加载")

    def crop_effective_area(self, image):
        """裁切黑边，获取有效视野"""
        if image is None:
            return None, (0, 0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image, (0, 0)

        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # 过滤噪点：有效面积太小则不裁剪
        if w * h < (image.shape[0] * image.shape[1] * 0.05):
            return image, (0, 0)

        cropped_image = image[y:y + h, x:x + w]
        return cropped_image, (x, y)

    def detect_nose_target(self, effective_image, target_side='center'):
        """
        寻找目标：基于几何插值定位精确鼻孔中心
        target_side: 'left' (人脸左鼻孔), 'right' (人脸右鼻孔), 'center' (中间)
        """
        if effective_image is None:
            return None, 0.0, 0.0

        h, w = effective_image.shape[:2]
        rgb_image = cv2.cvtColor(effective_image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # === 1. 获取基准点 (MediaPipe原始点位) ===
            # 279: 左鼻翼底部基准点 (Left Alar Base)
            # 49:  右鼻翼底部基准点 (Right Alar Base)
            pt_base_left = landmarks[279]
            pt_base_right = landmarks[49]

            # === 2. 计算几何中间点 (鼻小柱位置) ===
            mid_x = (pt_base_left.x + pt_base_right.x) * 0.5
            mid_y = (pt_base_left.y + pt_base_right.y) * 0.5

            # === 3. 根据需求计算最终目标点 ===
            target_norm_x = 0.0
            target_norm_y = 0.0

            target_side_str = str(target_side).strip().lower()
            if target_side_str == 'left':
                # 逻辑：左鼻翼点 和 中间点 的中间 = 左鼻孔中心
                target_norm_x = (pt_base_left.x + mid_x) * 0.5
                target_norm_y = (pt_base_left.y + mid_y) * 0.5
                
            elif target_side_str == 'right':
                # 逻辑：右鼻翼点 和 中间点 的中间 = 右鼻孔中心
                target_norm_x = (pt_base_right.x + mid_x) * 0.5
                target_norm_y = (pt_base_right.y + mid_y) * 0.5
                
            else:
                # 默认为中间点
                target_norm_x = mid_x
                target_norm_y = mid_y

            # 转换为像素坐标
            tx = int(target_norm_x * w)
            ty = int(target_norm_y * h)

            # === 4. 计算辅助数据 (保持原逻辑用于测距和判定) ===
            # 鼻孔基准间距 (用于 Z 轴深度估算)
            nostril_distance_px = math.hypot(
                (pt_base_left.x - pt_base_right.x) * w,
                (pt_base_left.y - pt_base_right.y) * h
            )

            # 鼻翼总宽度 (用于到达判定) - 使用更外侧的点 358(左外) 和 129(右外)
            left_wing_outer = landmarks[358]
            right_wing_outer = landmarks[129]
            feature_width = math.hypot(
                (left_wing_outer.x - right_wing_outer.x) * w,
                (left_wing_outer.y - right_wing_outer.y) * h
            )

            return (tx, ty), float(feature_width), float(nostril_distance_px)

        return None, 0.0, 0.0

    def release(self):
        """释放MediaPipe资源"""
        if hasattr(self, 'mp_face_mesh'):
            self.mp_face_mesh.close()
            print("[Vision] MediaPipe 资源已释放")


class RobotController:
    """
    机械臂控制系统
    功能：基于工具坐标系(摄像头视角)执行相对移动
    offset_flag=2: 明确基于工具坐标系进行偏移
    """
    def __init__(self, ip='192.168.58.2', simulate=False):
        self.ip = ip
        self.simulate = simulate
        self.robot = None
        self.connected = False

        # 已标定的工具坐标系和用户坐标系
        self.tool_id = 1
        self.user_id = 0
        
        # 错误计数器（用于状态机错误检测）
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

    def move_offset_tool_frame(self, dx, dy, dz, max_retries=3):
        """
        执行工具坐标系下的相对偏移
        dx, dy, dz: 毫米(mm)
        offset_flag=2: 明确基于工具坐标系
        增加重试机制
        """
        if not self.connected:
            return False

        if self.simulate:
            print(f"[Sim] 移动向量 -> X:{dx:.2f}, Y:{dy:.2f}, Z:{dz:.2f}")
            return True

        # 重试机制
        for attempt in range(max_retries):
            try:
                # 1) 读当前位置 - 增加兼容性处理
                raw_result = self.robot.GetActualTCPPose(0)
                
                # 检查返回值类型
                if isinstance(raw_result, int):
                    # 如果只返回了一个整数，说明出错了（或者是错误码）
                    print(f"[Debug] GetActualTCPPose 异常返回，仅返回整数: {raw_result}")
                    ret = raw_result
                    current_pose = None
                else:
                    # 正常解包
                    ret, current_pose = raw_result

                if ret != 0:
                    self.get_pose_fail_count += 1
                    print(f"[Err] GetPose失败: 返回码={ret}, 尝试 {attempt + 1}/{max_retries}, 累计失败 {self.get_pose_fail_count}/{self.MAX_GET_POSE_FAILURES}")
                    time.sleep(0.5)
                    continue
                else:
                    # 重置错误计数器
                    self.get_pose_fail_count = 0

                # 严格检查返回的姿态数据有效性
                if current_pose is None or not isinstance(current_pose, (list, tuple)):
                    print(f"[Err] GetPose返回空数据, 尝试 {attempt + 1}/{max_retries}")
                    time.sleep(0.5)
                    continue

                # 检查长度是否为6 (x, y, z, rx, ry, rz)
                if len(current_pose) != 6:
                    print(f"[Err] GetPose返回数据长度异常: {len(current_pose)}, 尝试 {attempt + 1}/{max_retries}")
                    time.sleep(0.5)
                    continue

                # 检查是否有NaN或无效值
                try:
                    valid_pose = [float(v) for v in current_pose]
                    if any(math.isnan(v) or math.isinf(v) for v in valid_pose):
                        print(f"[Err] GetPose返回NaN/Inf数据, 尝试 {attempt + 1}/{max_retries}")
                        time.sleep(0.5)
                        continue
                    current_pose = valid_pose
                except (TypeError, ValueError) as e:
                    print(f"[Err] GetPose数据转换异常: {e}, 尝试 {attempt + 1}/{max_retries}")
                    time.sleep(0.5)
                    continue

                # 2) 拼装偏移向量
                offset_pos = [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0]

                # 3) 执行移动
                # offset_flag=2: 明确基于工具坐标系姿态进行偏移
                ret = self.robot.MoveL(
                    desc_pos=current_pose,
                    tool=self.tool_id,
                    user=self.user_id,
                    vel=20.0,
                    acc=50.0,
                    offset_flag=2,  # 工具坐标系
                    offset_pos=offset_pos
                )
                
                if ret == 0:
                    return True
                else:
                    print(f"[Err] MoveL失败: 返回码={ret}, 尝试 {attempt + 1}/{max_retries}")
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"[Err] 移动异常: {e}, 尝试 {attempt + 1}/{max_retries}")
                time.sleep(0.5)

        print(f"[Err] 移动失败，已重试{max_retries}次")
        return False

    def disconnect(self):
        """断开机械臂连接"""
        if self.simulate:
            print("[Robot] 模拟模式，无需断开")
            return
        
        if self.robot is not None:
            try: 
                # 清除机械臂错误
                try:
                    self.robot.ResetAllError()
                    print("[Robot] 已清除机械臂错误")
                except Exception as e:
                    print(f"[Robot] 清除错误异常: {e}")
                
                # 尝试释放Robot对象引用
                self.robot = None
                print("[Robot] 机械臂对象已释放")
            except Exception as e:
                print(f"[Robot] 断开异常: {e}")
        else:
            print("[Robot] 机械臂未连接")
        
        self.connected = False


def main():
    # -------- 系统参数 --------
    VIDEO_IDX = 1

    # 动态比例参数（基于鼻孔间距）
    NOSTRIL_DISTANCE_MM = 12.0   # 鼻孔中心间距经验值：12mm
    
    XY_TOLERANCE = 20            # 平面容差（像素）
    TARGET_WIDTH_THRESHOLD = 300 # 鼻翼宽度阈值（像素）超过认为到达
    Z_APPROACH_STEP = 15         # 每次向前（mm）
    
    # 深度移动安全限制
    MAX_Z_TOTAL_MM = 150.0       # Z轴总前进距离不超过150mm
    RETREAT_STEP_MM = 20.0       # 后退距离

    # 滤波 / 限幅 / 异常帧拒绝参数
    EMA_ALPHA_POS = 0.35
    EMA_ALPHA_W = 0.25

    MAX_JUMP_PX = 90             # 异常帧：目标点跳变超过此像素 -> 拒绝
    MAX_WIDTH_JUMP_RATIO = 0.40  # 异常帧：宽度相对跳变超过此比例 -> 拒绝

    XY_MAX_STEP_MM = 10.0        # 限幅：单次XY最大移动(mm)
    Z_MAX_STEP_MM = 20.0         # 限幅：单次Z最大移动(mm)

    LOST_TIMEOUT_SEC = 3.0       # 丢失目标超时：3秒
    FILTER_RESET_TIMEOUT = 2.0   # 丢失超过2秒重置滤波器
    
    # 可选值: 'left' (人脸左鼻孔), 'right' (人脸右鼻孔), 'center' (中间)
    # 注意: 'left' 通常对应屏幕画面中的右侧(镜像关系)
    TARGET_SELECTION = 'left'

    # -------- 初始化 --------
    vision = VisionSystem()
    bot = RobotController(simulate=False)  # 改为True可在无机械臂时测试
    
    if not bot.connect():
        print("[错误] 无法连接机械臂，程序终止")
        vision.release()
        return

    cap = cv2.VideoCapture(VIDEO_IDX)
    if not cap.isOpened():
        print("[错误] 摄像头打开失败")
        bot.disconnect()
        vision.release()
        return

    cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)

    auto_run = False
    finished = False
    system_terminated = False  # 系统终止标志
    
    # 状态机变量
    current_state = SystemState.IDLE

    # 目标滤波器
    pos_filter = EMAFilter(alpha=EMA_ALPHA_POS)
    w_filter = EMAFilter(alpha=EMA_ALPHA_W)
    nostril_filter = EMAFilter(alpha=EMA_ALPHA_W)  # 鼻孔间距滤波

    # 上一帧有效观测（用于异常帧拒绝）
    last_raw_pos = None
    last_raw_w = None

    # 丢失目标计时
    lost_start_time = None
    had_target_before = False
    filter_reset_done = False  # 标记滤波器是否已重置

    # 深度移动计数器
    z_total_moved = 0.0        # 累计向前移动距离
    retreat_attempted = False  # 是否已尝试后退
    
    # 控制频率限制变量
    last_move_time = 0.0
    MIN_MOVE_INTERVAL = 0.1  # 最小指令下发间隔（秒），100ms对应10Hz

    print("=" * 60)
    print(">>> 程序运行中 <<<")
    print("  [空格键] 启动/暂停自动运动")
    print("  [Q键]    退出程序")
    print("=" * 60)

    try:
        while True:
            ret, frame_raw = cap.read()
            if not ret:
                print("[警告] 无法读取摄像头帧")
                break

            # 裁剪黑边
            eff_frame, _ = vision.crop_effective_area(frame_raw)
            if eff_frame is None:
                eff_frame = frame_raw

            h, w = eff_frame.shape[:2]
            cam_center_x, cam_center_y = w // 2, h // 2

            # 1) 检测目标
            nose_pos, nose_width, nostril_px = vision.detect_nose_target(eff_frame, target_side=TARGET_SELECTION)

            # 2) 异常帧拒绝（跳变过大则当作"本帧无效目标"）
            valid_obs = False
            if nose_pos is not None and nose_width > 1.0 and nostril_px > 1.0:
                if last_raw_pos is not None:
                    jump = math.hypot(nose_pos[0] - last_raw_pos[0], nose_pos[1] - last_raw_pos[1])
                else:
                    jump = 0.0

                if last_raw_w is not None and last_raw_w > 1.0:
                    w_jump_ratio = abs(nose_width - last_raw_w) / last_raw_w
                else:
                    w_jump_ratio = 0.0

                if (jump <= MAX_JUMP_PX) and (w_jump_ratio <= MAX_WIDTH_JUMP_RATIO):
                    valid_obs = True

            # 3) 如果观测有效，更新滤波
            filtered_pos = None
            filtered_w = None
            filtered_nostril = None
            dynamic_pixel_to_mm = 0.05  # 默认值

            if valid_obs:
                last_raw_pos = nose_pos
                last_raw_w = nose_width
                filtered_pos = pos_filter.update([nose_pos[0], nose_pos[1]])
                filtered_w = float(w_filter.update([nose_width])[0])
                filtered_nostril = float(nostril_filter.update([nostril_px])[0])
                
                # 动态计算像素-毫米比例
                if filtered_nostril > 5.0:  # 避免除零
                    dynamic_pixel_to_mm = NOSTRIL_DISTANCE_MM / filtered_nostril
                
                # 重置丢失相关状态
                lost_start_time = None
                filter_reset_done = False
                had_target_before = True
            else:
                nose_pos = None
                nose_width = 0.0
                nostril_px = 0.0

            # 4) 控制逻辑 - 状态机实现
            # 计算偏差（必须在状态判断之前）
            if filtered_pos is not None:
                nose_x, nose_y = int(filtered_pos[0]), int(filtered_pos[1])
                err_x = nose_x - cam_center_x
                err_y = nose_y - cam_center_y
            else:
                err_x, err_y = 0, 0
            
            # 更新状态
            if system_terminated:
                new_state = SystemState.TERMINATED
            elif finished:
                new_state = SystemState.TARGET_REACHED
            elif filtered_pos is None:
                new_state = SystemState.TARGET_LOST
            elif auto_run:
                # 根据当前情况决定状态
                if abs(err_x) > XY_TOLERANCE or abs(err_y) > XY_TOLERANCE:
                    new_state = SystemState.ALIGN_XY
                elif filtered_w < TARGET_WIDTH_THRESHOLD:
                    new_state = SystemState.APPROACH_Z
                else:
                    new_state = SystemState.TARGET_REACHED
            else:
                new_state = SystemState.IDLE
            
            # 绘制UI：准星和目标点
            # 准星
            cv2.line(eff_frame, (cam_center_x - 10, cam_center_y), 
                    (cam_center_x + 10, cam_center_y), (0, 255, 0), 2)
            cv2.line(eff_frame, (cam_center_x, cam_center_y - 10), 
                    (cam_center_x, cam_center_y + 10), (0, 255, 0), 2)
            
            # 目标点（如果检测到目标）
            if filtered_pos is not None:
                cv2.circle(eff_frame, (nose_x, nose_y), 5, (0, 0, 255), -1)
                cv2.line(eff_frame, (cam_center_x, cam_center_y), 
                        (nose_x, nose_y), (255, 255, 0), 1)
            
            # 状态转换处理
            if new_state != current_state:
                current_state = new_state
            
            # 执行当前状态的操作
            move_x, move_y, move_z = 0.0, 0.0, 0.0
            
            if current_state == SystemState.IDLE:
                status_msg = "Idle"
            elif current_state == SystemState.ALIGN_XY:
                status_msg = "Aligning XY"
                move_x = float(err_x) * dynamic_pixel_to_mm
                move_y = float(err_y) * dynamic_pixel_to_mm          
                
                # 限幅
                move_x = clamp(move_x, -XY_MAX_STEP_MM, XY_MAX_STEP_MM)
                move_y = clamp(move_y, -XY_MAX_STEP_MM, XY_MAX_STEP_MM)
                
            elif current_state == SystemState.APPROACH_Z:
                # 动态减速：根据距离目标的远近调整步长
                width_diff = TARGET_WIDTH_THRESHOLD - filtered_w
                
                # 差得多就走 15mm，差得少就只走 5mm 或 2mm
                if width_diff > 30:
                    actual_step = Z_APPROACH_STEP  # 15mm
                elif width_diff > 10:
                    actual_step = 5.0  # 5mm
                else:
                    actual_step = 2.0  # 2mm
                
                # 检查深度安全限制
                if z_total_moved + actual_step >= MAX_Z_TOTAL_MM:
                    status_msg = "Target Reached (Z Limit)"
                    finished = True
                    print(f"[完成] 深度移动已达限制: {z_total_moved:.1f}mm")
                else:
                    status_msg = f"Approaching Z ({int(filtered_w)}px, {z_total_moved:.1f}mm)"
                    move_z = float(actual_step)
                    move_z = clamp(move_z, 0.0, Z_MAX_STEP_MM)
                    
                    # 确保不超过限制
                    if z_total_moved + move_z > MAX_Z_TOTAL_MM:
                        move_z = MAX_Z_TOTAL_MM - z_total_moved
                        
            elif current_state == SystemState.TARGET_REACHED:
                status_msg = "Target Reached (Width)"
                finished = True
                print(f"[完成] 目标宽度达标，深度移动: {z_total_moved:.1f}mm")
                
            elif current_state == SystemState.TARGET_LOST:
                status_msg = "No Target"
                
                if auto_run and had_target_before:
                    # 启动丢失计时
                    if lost_start_time is None:
                        lost_start_time = time.time()
                        print("[警告] 目标丢失，开始计时...")

                    lost_dt = time.time() - lost_start_time
                    status_msg = f"Lost Target... {lost_dt:.1f}s"

                    # 超过2秒重置滤波器（避免跳变）
                    if lost_dt >= FILTER_RESET_TIMEOUT and not filter_reset_done:
                        pos_filter.reset()
                        w_filter.reset()
                        nostril_filter.reset()
                        filter_reset_done = True
                        print("[Info] 滤波器已重置")

                    # 超过3秒处理
                    if lost_dt >= LOST_TIMEOUT_SEC:
                        # 判断深度移动距离
                        if z_total_moved < MAX_Z_TOTAL_MM and not retreat_attempted:
                            # 后退重新检测
                            print(f"[策略] 深度移动{z_total_moved:.1f}mm < 150mm，后退{RETREAT_STEP_MM}mm重新检测")
                            retreat_attempted = True
                            success = bot.move_offset_tool_frame(0, 0, -RETREAT_STEP_MM)
                            if success:
                                z_total_moved -= RETREAT_STEP_MM
                                if z_total_moved < 0:
                                    z_total_moved = 0
                                # 后退成功后立即清理异常帧拒绝和滤波器
                                last_raw_pos = None
                                last_raw_w = None
                                pos_filter.reset()
                                w_filter.reset()
                                nostril_filter.reset()
                                print("[后退完成] 已重置滤波器和异常帧拒绝")
                                # 重置计时，给后退后检测一次机会
                                lost_start_time = time.time()
                        else:
                            # 后退后仍检测不到，或已达深度限制
                            print("[错误] 无法检测到人脸，终止运行")
                            status_msg = "Terminated: No Face"
                            system_terminated = True
                            auto_run = False
                            
            elif current_state == SystemState.TERMINATED:
                status_msg = "Terminated"
            else:
                status_msg = "Unknown State"
            
            # 执行移动（除了IDLE和TARGET_LOST状态）
            if current_state in [SystemState.ALIGN_XY, SystemState.APPROACH_Z]:
                if (move_x != 0.0) or (move_y != 0.0) or (move_z != 0.0):
                    current_time = time.time()
                    # 检查是否满足最小指令下发间隔
                    if current_time - last_move_time >= MIN_MOVE_INTERVAL:
                        success = bot.move_offset_tool_frame(move_x, move_y, move_z)
                        if not success:
                            print("[错误] 机械臂移动失败")
                            # 可选：标记系统错误
                        else:
                            # 更新深度计数器
                            if move_z > 0:
                                z_total_moved += move_z
                            # 记录最后一次移动时间
                            last_move_time = current_time
                    else:
                        # 在间隔期内，只更新视觉滤波结果但不调用MoveL
                        pass
            elif current_state == SystemState.TERMINATED:
                status_msg = "Terminated"
            elif current_state == SystemState.TARGET_REACHED:
                status_msg = "Target Reached"
            elif current_state == SystemState.IDLE:
                status_msg = "Idle"
            elif current_state == SystemState.TARGET_LOST:
                status_msg = "No Target"
            else:
                status_msg = "Unknown State"

            # 5) 显示信息
            cv2.putText(eff_frame, f"Target: {TARGET_SELECTION.upper()}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            cv2.putText(eff_frame, f"State: {status_msg}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(eff_frame, f"Auto: {auto_run}  Finished: {finished}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(eff_frame, f"Z Moved: {z_total_moved:.1f}mm / {MAX_Z_TOTAL_MM}mm", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
            
            if filtered_pos is not None:
                cv2.putText(eff_frame, f"Err(px): X={err_x}, Y={err_y}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(eff_frame, f"Width: {int(filtered_w)}px", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(eff_frame, f"Pixel->MM: {dynamic_pixel_to_mm:.4f}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
            else:
                cv2.putText(eff_frame, f"Width: 0px", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow("Robot View", eff_frame)

            # 6) 按键控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Info] 用户请求退出")
                break
            if key == ord(' '):
                if not finished and not system_terminated:
                    auto_run = not auto_run
                    print(f"[Info] 自动运行: {auto_run}")
                    if auto_run:
                        # 重新开始时重置状态
                        lost_start_time = None
                        retreat_attempted = False
                else:
                    if system_terminated:
                        print("[Info] 系统已终止，按Q退出")
                    else:
                        print("[Info] 运动已完成，按Q退出")

    except KeyboardInterrupt:
        print("\n[Info] 用户中断程序")
    except Exception as e:
        print(f"[错误] 程序异常: {e}")
    finally:
        # 资源释放
        print("\n" + "=" * 60)
        print(">>> 正在释放资源 <<<")
        cap.release()
        cv2.destroyAllWindows()
        bot.disconnect()
        vision.release()
        print(">>> 资源释放完成 <<<")
        print("=" * 60)


if __name__ == "__main__":
    main()
