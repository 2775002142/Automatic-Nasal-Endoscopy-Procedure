# 文件路径: ~/FR5/src/fr5_vision_control/fr5_vision_control/move_inside_node.py

import sys
import threading
import time
import math
import cv2
import numpy as np
import concurrent.futures
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

# 导入修改后的模块
from fr5_vision_control.robot_controller import RobotController
from fr5_vision_control.utils import SystemState, EMAFilter, KalmanFilter2D # 导入 KalmanFilter2D
from fr5_vision_control.vision_inside import APFVisionSystem, ForceToMotionConverter # 导入 ForceToMotionConverter

class MoveInsideNode(Node):
    def __init__(self):
        super().__init__('move_inside_node')
        
        # 1. 参数声明与获取
        # 视觉与运动控制相关参数
        self.declare_parameter('video_idx', '0')  # 摄像头索引，可以是数字或设备路径
        self.declare_parameter('simulate', True)  # 是否在仿真模式下运行
        self.declare_parameter('max_depth_mm', 50.0)  # 最大插入深度（毫米）
        self.declare_parameter('align_dist_start_px', 150.0)  # 开始对齐的距离阈值（像素）
        self.declare_parameter('align_dist_stop_px', 130.0)  # 停止对齐的距离阈值（像素）
        
        # 运动状态转换参数
        self.declare_parameter('min_rotate_frames', 2)  # 最小旋转帧数，确保旋转动作稳定
        self.declare_parameter('min_advance_frames', 5)  # 最小前进帧数，确保前进动作稳定
        self.declare_parameter('transition_frames', 10)  # 状态转换所需的最大帧数
        
        # 移动步长参数
        self.declare_parameter('z_advance_step_mm', 0.5)  # Z轴方向前进步长（毫米）
        self.declare_parameter('retreat_step_mm', -0.5)  # 后退步长（毫米）
        
        # 超时与滤波参数
        self.declare_parameter('blocked_timeout_sec', 1.5)  # 阻塞超时时间（秒）
        self.declare_parameter('ema_alpha', 0.25)  # 指数移动平均滤波器的平滑系数
        self.declare_parameter('kalman_process_noise', 0.05)  # 卡尔曼滤波器过程噪声
        self.declare_parameter('kalman_measure_noise', 4.0)  # 卡尔曼滤波器测量噪声
        
        # 盲插策略参数
        self.declare_parameter('blind_entry_step_mm', 2.0)  # 盲插阶段每步移动距离（毫米）
        self.declare_parameter('blind_entry_max_mm', 20.0)  # 盲插阶段最大移动距离（毫米）
        self.declare_parameter('blind_entry_interval_sec', 0.15)  # 盲插阶段步进间隔（秒）
        self.declare_parameter('blind_goal_confirm_frames', 20)  # 确认到达目标所需的帧数
        
        # 旋转限制参数
        self.declare_parameter('max_total_rotation_deg', 2.0)  # 总体最大旋转角度（度）
        
        # 力到运动转换参数
        self.declare_parameter('converter_force_deadzone', 10.0)  # 力传感器死区阈值
        self.declare_parameter('converter_max_force_for_scale', 1150.0)  # 用于缩放的最大力值
        self.declare_parameter('converter_min_rotation_gain', 0.001)  # 最小旋转增益
        self.declare_parameter('converter_max_rotation_gain', 0.012)  # 最大旋转增益
        self.declare_parameter('converter_rotation_gain_curve_factor', 0.6)  # 旋转增益曲线因子
        self.declare_parameter('converter_max_rotation_deg', 0.2) # 单次最大旋转
        self.declare_parameter('converter_min_translate_step_mm', 0.02)  # 最小平移步长（毫米）
        self.declare_parameter('converter_max_translate_step_mm', 0.15)  # 最大平移步长（毫米）
        self.declare_parameter('converter_translate_step_curve_factor', 0.7)  # 平移步长曲线因子
        self.declare_parameter('converter_max_translate_per_phase_mm', 1.0) # 单次最大平移


        video_param = self.get_parameter('video_idx').get_parameter_value().string_value
        self.video_idx = int(video_param) if video_param.isdigit() else video_param
        self.simulate = self.get_parameter('simulate').get_parameter_value().bool_value
        
        # 加载所有参数
        self.MAX_DEPTH_MM = self.get_parameter('max_depth_mm').get_parameter_value().double_value
        self.ALIGN_DIST_START_PX = self.get_parameter('align_dist_start_px').get_parameter_value().double_value
        self.ALIGN_DIST_STOP_PX = self.get_parameter('align_dist_stop_px').get_parameter_value().double_value
        self.MIN_ROTATE_FRAMES = self.get_parameter('min_rotate_frames').get_parameter_value().integer_value
        self.MIN_ADVANCE_FRAMES = self.get_parameter('min_advance_frames').get_parameter_value().integer_value
        self.TRANSITION_FRAMES = self.get_parameter('transition_frames').get_parameter_value().integer_value
        self.Z_ADVANCE_STEP_MM = self.get_parameter('z_advance_step_mm').get_parameter_value().double_value
        self.RETREAT_STEP_MM = self.get_parameter('retreat_step_mm').get_parameter_value().double_value
        self.BLOCKED_TIMEOUT_SEC = self.get_parameter('blocked_timeout_sec').get_parameter_value().double_value
        self.EMA_ALPHA = self.get_parameter('ema_alpha').get_parameter_value().double_value
        self.KALMAN_PROCESS_NOISE = self.get_parameter('kalman_process_noise').get_parameter_value().double_value
        self.KALMAN_MEASURE_NOISE = self.get_parameter('kalman_measure_noise').get_parameter_value().double_value
        self.BLIND_ENTRY_STEP_MM = self.get_parameter('blind_entry_step_mm').get_parameter_value().double_value
        self.BLIND_ENTRY_MAX_MM = self.get_parameter('blind_entry_max_mm').get_parameter_value().double_value
        self.BLIND_ENTRY_INTERVAL_SEC = self.get_parameter('blind_entry_interval_sec').get_parameter_value().double_value
        self.BLIND_GOAL_CONFIRM_FRAMES = self.get_parameter('blind_goal_confirm_frames').get_parameter_value().integer_value
        self.MAX_TOTAL_ROTATION_DEG = self.get_parameter('max_total_rotation_deg').get_parameter_value().double_value


        self.get_logger().info(f'启动内部导航 -> 视频源: {self.video_idx}, 模拟: {self.simulate}')

        # 2. 初始化子系统
        self.vision = APFVisionSystem(debug=False)
        self.robot = RobotController(self, simulate=self.simulate) # 传入 ROS Node 实例

        # 初始化转换器 (从参数服务器获取)
        self.converter = ForceToMotionConverter(
            force_deadzone=self.get_parameter('converter_force_deadzone').get_parameter_value().double_value,
            max_force_for_scale=self.get_parameter('converter_max_force_for_scale').get_parameter_value().double_value,
            min_rotation_gain=self.get_parameter('converter_min_rotation_gain').get_parameter_value().double_value,
            max_rotation_gain=self.get_parameter('converter_max_rotation_gain').get_parameter_value().double_value,
            rotation_gain_curve_factor=self.get_parameter('converter_rotation_gain_curve_factor').get_parameter_value().double_value,
            max_rotation_deg=self.get_parameter('converter_max_rotation_deg').get_parameter_value().double_value,
            min_translate_step_mm=self.get_parameter('converter_min_translate_step_mm').get_parameter_value().double_value,
            max_translate_step_mm=self.get_parameter('converter_max_translate_step_mm').get_parameter_value().double_value,
            translate_step_curve_factor=self.get_parameter('converter_translate_step_curve_factor').get_parameter_value().double_value,
            max_translate_per_phase_mm=self.get_parameter('converter_max_translate_per_phase_mm').get_parameter_value().double_value
        )

        # 3. 状态变量和计数器
        self.auto_run = False
        self.system_terminated = False
        self.current_state = SystemState.IDLE
        self.current_depth = 0.0
        self.state_frame_counter = 0
        self.moving_lock = threading.Lock()
        self.motion_in_progress = False

        # 滤波和计时器
        self.goal_kalman_filter = KalmanFilter2D(
            process_noise_cov=self.KALMAN_PROCESS_NOISE,
            measurement_noise_cov=self.KALMAN_MEASURE_NOISE
        )
        self.force_x_filter = EMAFilter(alpha=self.EMA_ALPHA)
        self.force_y_filter = EMAFilter(alpha=self.EMA_ALPHA)
        self.dist_filter = EMAFilter(alpha=self.EMA_ALPHA)
        self.blocked_start_time = None
        self.last_move_time = 0.0 # 用于控制机器人动作频率，防止发送指令过快

        # 盲走变量
        self.blind_entry_completed = False
        self.blind_entry_distance = 0.0
        self.blind_goal_consecutive = 0
        self.blind_last_step_time = 0.0

        # 累积量（用于可视化和安全限制）
        self.total_rx_deg = 0.0
        self.total_ry_deg = 0.0
        self.total_translation_dx = 0.0
        self.total_translation_dy = 0.0
        self.current_rotation_rx = 0.0      # 当前旋转阶段累计 (用于rotation_safe)
        self.current_rotation_ry = 0.0      # 当前旋转阶段累计
        self.current_translation_dx = 0.0   # 当前平移阶段累计 (用于 converter 内部限幅)
        self.current_translation_dy = 0.0   # 当前平移阶段累计

        self.data_lock = threading.Lock()
        self.latest_frame = None            # 存储摄像头捕获的最新帧
        self.display_frame = None           # 存储处理完毕、可以显示的帧
        self.processing_thread_running = False # 控制处理线程的运行

        self.auto_run = False
        self.finished = False
        self.system_terminated = False  # 系统终止标志
        self.current_state = SystemState.IDLE

        self.task_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # 打开摄像头
        self.cap = cv2.VideoCapture(self.video_idx)
        if not self.cap.isOpened():
            self.get_logger().error(f"摄像头 {self.video_idx} 打开失败")
            self.system_terminated = True
        
        cv2.namedWindow("Inside Navigation", cv2.WINDOW_NORMAL)
        self.get_logger().info(">>> 按 [空格] 开始/暂停，按 [R] 重置深度，按 [Q/ESC] 退出 <<<")

        # 打印初始化参数，便于调试
        self.get_logger().info("=" * 60)
        self.get_logger().info(">>> 体内导航系统 ROS 节点初始化 <<<")
        self.get_logger().info(f"  [旋转增益] {self.converter.min_rotation_gain:.4f} ~ {self.converter.max_rotation_gain:.4f} | "
                              f"单次上限={self.converter.max_rotation_deg}°")
        self.get_logger().info(f"  [平移步长] {self.converter.min_translate_step_mm:.3f} ~ {self.converter.max_translate_step_mm:.3f}mm | "
                              f"每阶段上限={self.converter.max_translate_per_phase_mm}mm")
        self.get_logger().info(f"  [切换] 偏离>{self.ALIGN_DIST_START_PX}px开始校准 | 偏离<{self.ALIGN_DIST_STOP_PX}px允许直行")
        self.get_logger().info(f"  [防抖] 旋转最少{self.MIN_ROTATE_FRAMES}帧 | 前进最少{self.MIN_ADVANCE_FRAMES}帧 | 过渡{self.TRANSITION_FRAMES}帧")
        self.get_logger().info(f"  [安全] 单阶段旋转上限={self.MAX_TOTAL_ROTATION_DEG}° | 最大深度={self.MAX_DEPTH_MM}mm")
        self.get_logger().info("=" * 60)

    def _processing_loop(self):
        while self.processing_thread_running and rclpy.ok():
            try:
                with self.data_lock:
                    frame_to_process = self.latest_frame
                
                if frame_to_process is None:
                    time.sleep(0.01) # 等待第一帧
                    continue

                # ==================================================
                # 2. 视觉处理 & 卡尔曼滤波目标点
                # ==================================================
                vis_frame, raw_force, raw_goal = self.vision.process_frame(frame_to_process)


                goal = None
                if raw_goal is not None:
                    filtered_pos = self.goal_kalman_filter.update(raw_goal)
                    goal = (int(filtered_pos[0]), int(filtered_pos[1]))
                else:
                    self.goal_kalman_filter.reset()

                # ==================================================
                # 3. 力场与距离计算（基于滤波后目标）
                # ==================================================
                pixel_dist = 0.0
                if goal is not None:
                    # 用滤波后目标点重建吸引力，叠加原始排斥力
                    att_filtered = self.vision._calculate_attractive_force(goal)
                    if raw_goal is not None:
                        att_raw = self.vision._calculate_attractive_force(raw_goal)
                        rep_force = raw_force - att_raw # 减去原始吸引力，得到纯排斥力
                    else:
                        rep_force = np.array([0.0, 0.0])
                    combined_force = att_filtered + rep_force

                    fx = float(self.force_x_filter.update(combined_force[0]))
                    fy = float(self.force_y_filter.update(combined_force[1]))
                    filtered_force = np.array([fx, fy])
                    
                    raw_dist  = math.hypot(
                        goal[0] - self.vision.center[0],
                        goal[1] - self.vision.center[1]
                    )
                    pixel_dist = float(self.dist_filter.update(raw_dist))
                    self.blocked_start_time = None
                else:
                    filtered_force = np.array([0.0, 0.0])
                    self.force_x_filter.reset()
                    self.force_y_filter.reset()

                force_mag = np.linalg.norm(filtered_force)
                
                # ==================================================
                # 4. 状态机评估
                # ==================================================
                new_state   = self.current_state
                action_type = None
                rx_cmd = ry_cmd = dx_cmd = dy_cmd = dz_cmd = 0.0
                status_msg  = ""

                if not self.auto_run:
                    new_state  = SystemState.IDLE
                    status_msg = "IDLE (已暂停)"

                elif not self.blind_entry_completed:
                    # ===== 盲走阶段 =====
                    new_state = SystemState.BLIND_ENTRY

                    if goal is not None:
                        self.blind_goal_consecutive += 1
                    else:
                        self.blind_goal_consecutive = 0

                    if self.blind_goal_consecutive >= self.BLIND_GOAL_CONFIRM_FRAMES:
                        self.blind_entry_completed = True
                        self.get_logger().info(f"[盲走完成] 连续{self.BLIND_GOAL_CONFIRM_FRAMES}帧检测到目标，切换正常导航。")
                        self.force_x_filter.reset()
                        self.force_y_filter.reset()
                        new_state = SystemState.TRANSITION_TO_ROTATE # 过渡到旋转状态
                        self.current_rotation_rx = self.current_rotation_ry = 0.0
                        self.current_translation_dx = self.current_translation_dy = 0.0
                        status_msg = "Blind->Transition to Rotate"

                    elif self.blind_entry_distance >= self.BLIND_ENTRY_MAX_MM:
                        self.blind_entry_completed = True
                        self.get_logger().info(f"[盲走上限] 已盲走{self.blind_entry_distance:.1f}mm，强制切换。")
                        self.force_x_filter.reset()
                        self.force_y_filter.reset()
                        new_state = SystemState.TRANSITION_TO_ROTATE # 过渡到旋转状态
                        self.current_rotation_rx = self.current_rotation_ry = 0.0
                        self.current_translation_dx = self.current_translation_dy = 0.0
                        status_msg = "Blind Max->Transition to Rotate"
                    else:
                        now = time.time()
                        if now - self.blind_last_step_time >= self.BLIND_ENTRY_INTERVAL_SEC:
                            action_type = 'blind_z'
                            dz_cmd = self.BLIND_ENTRY_STEP_MM
                            self.blind_last_step_time = now
                        status_msg = f"BLIND_ENTRY {self.blind_entry_distance:.1f}/{self.BLIND_ENTRY_MAX_MM}mm"

                else:
                    # ===== 正常导航阶段 =====

                    # rotation_safe 仅在旋转相关状态下限制
                    current_rot = math.sqrt(self.current_rotation_rx**2 + self.current_rotation_ry**2)
                    if self.current_state in [SystemState.ROTATE_ALIGN, SystemState.TRANSITION_TO_ROTATE]:
                        rotation_safe = current_rot < self.MAX_TOTAL_ROTATION_DEG
                    else:
                        rotation_safe = True  # 非旋转状态始终允许重新启动旋转

                    # --- 最高优先级判断 ---

                    # (a) 达到最大深度
                    if self.current_depth >= self.MAX_DEPTH_MM:
                        new_state  = SystemState.MAX_DEPTH_REACHED
                        status_msg = f"MAX DEPTH {self.current_depth:.1f}mm (按R重置)"

                    # (b) 目标丢失 → BLOCKED / RETREAT
                    elif goal is None:
                        if self.blocked_start_time is None:
                            self.blocked_start_time = time.time()
                        elapsed = time.time() - self.blocked_start_time
                        if elapsed > self.BLOCKED_TIMEOUT_SEC:
                            new_state   = SystemState.RETREAT
                            action_type = 'retreat'
                            dz_cmd      = self.RETREAT_STEP_MM
                            status_msg  = f"RETREAT elapsed={elapsed:.1f}s"
                        else:
                            new_state  = SystemState.BLOCKED
                            status_msg = f"BLOCKED {elapsed:.1f}/{self.BLOCKED_TIMEOUT_SEC}s"

                    # (c) 正常滞回切换
                    else:
                        if self.current_state in [SystemState.ADVANCE_Z, SystemState.IDLE]:
                            if (pixel_dist > self.ALIGN_DIST_START_PX
                                    and rotation_safe
                                    and self.state_frame_counter >= self.MIN_ADVANCE_FRAMES):
                                new_state = SystemState.TRANSITION_TO_ROTATE
                            else:
                                new_state   = SystemState.ADVANCE_Z
                                action_type = 'advance_z'
                                dz_cmd      = self.Z_ADVANCE_STEP_MM
                                status_msg  = f"ADVANCE D={pixel_dist:.1f} F={force_mag:.1f}"

                        elif self.current_state == SystemState.ROTATE_ALIGN:
                            if ((pixel_dist < self.ALIGN_DIST_STOP_PX or not rotation_safe)
                                    and self.state_frame_counter >= self.MIN_ROTATE_FRAMES):
                                new_state = SystemState.TRANSITION_TO_ADVANCE
                            else:
                                new_state   = SystemState.ROTATE_ALIGN
                                action_type = 'rotate_translate'
                                rx_cmd, ry_cmd, dx_cmd, dy_cmd = self.converter.convert(
                                    filtered_force,
                                    self.current_translation_dx,
                                    self.current_translation_dy
                                )
                                status_msg = (f"ROTATE D={pixel_dist:.1f} "
                                            f"Rx={rx_cmd:.3f}° Ry={ry_cmd:.3f}° "
                                            f"dx={dx_cmd:.3f} dy={dy_cmd:.3f}")

                        elif self.current_state in [SystemState.TRANSITION_TO_ROTATE, SystemState.TRANSITION_TO_ADVANCE]:
                            pass # 由下方过渡逻辑处理
                        else:
                            # 从 BLOCKED/RETREAT/其他状态恢复  
                            new_state = SystemState.TRANSITION_TO_ROTATE

                # ==================================================
                # 5. 状态切换：重置计数器与累积量
                # ==================================================
                if new_state != self.current_state:
                    self.get_logger().info(f"状态切换: {self.current_state.name} -> {new_state.name}")
                    self.current_state       = new_state
                    self.state_frame_counter = 0
                    self.last_move_time = 0.0 # 强制允许立即执行下一个状态的动作

                    # 进入旋转相关状态时重置当前阶段累计
                    if self.current_state in [SystemState.TRANSITION_TO_ROTATE, SystemState.ROTATE_ALIGN]:
                        self.current_rotation_rx    = 0.0
                        self.current_rotation_ry    = 0.0
                        self.current_translation_dx = 0.0
                        self.current_translation_dy = 0.0

                    # 正式进入前进状态时清空（过渡期保留）
                    if self.current_state == SystemState.ADVANCE_Z:
                        self.current_rotation_rx    = 0.0
                        self.current_rotation_ry    = 0.0
                        self.current_translation_dx = 0.0
                        self.current_translation_dy = 0.0
                else:
                    self.state_frame_counter += 1

                # ==================================================
                # 6. 过渡状态的运动指令计算
                # ==================================================
                if self.current_state == SystemState.TRANSITION_TO_ROTATE:
                    ratio = min(self.state_frame_counter / max(self.TRANSITION_FRAMES, 1), 1.0)
                    t_rx, t_ry, t_dx, t_dy = self.converter.convert(
                        filtered_force,
                        self.current_translation_dx,
                        self.current_translation_dy
                    )
                    rx_cmd = t_rx * ratio
                    ry_cmd = t_ry * ratio
                    dx_cmd = t_dx * ratio
                    dy_cmd = t_dy * ratio
                    action_type = 'rotate_translate'
                    status_msg  = f"Transition→ROTATE ({self.state_frame_counter}/{self.TRANSITION_FRAMES})"
                    if self.state_frame_counter >= self.TRANSITION_FRAMES:
                        self.get_logger().info(f"过渡完成: {self.current_state.name} -> {SystemState.ROTATE_ALIGN.name}")
                        self.current_state       = SystemState.ROTATE_ALIGN
                        self.state_frame_counter = 0

                elif self.current_state == SystemState.TRANSITION_TO_ADVANCE:
                    ratio = min(self.state_frame_counter / max(self.TRANSITION_FRAMES, 1), 1.0)
                    t_rx, t_ry, t_dx, t_dy = self.converter.convert(
                        filtered_force,
                        self.current_translation_dx,
                        self.current_translation_dy
                    )
                    # 旋转/平移线性淡出，Z轴线性淡入
                    rx_cmd = t_rx * (1.0 - ratio)
                    ry_cmd = t_ry * (1.0 - ratio)
                    dx_cmd = t_dx * (1.0 - ratio)
                    dy_cmd = t_dy * (1.0 - ratio)
                    dz_cmd = self.Z_ADVANCE_STEP_MM * ratio
                    action_type = 'all'
                    status_msg  = f"Transition→ADVANCE ({self.state_frame_counter}/{self.TRANSITION_FRAMES})"
                    if self.state_frame_counter >= self.TRANSITION_FRAMES:
                        self.get_logger().info(f"过渡完成: {self.current_state.name} -> {SystemState.ADVANCE_Z.name}")
                        self.current_state       = SystemState.ADVANCE_Z
                        self.state_frame_counter = 0
                        # 过渡完成正式进入前进，清空累积量
                        self.current_rotation_rx    = 0.0
                        self.current_rotation_ry    = 0.0
                        self.current_translation_dx = 0.0
                        self.current_translation_dy = 0.0

                # ==================================================
                # 7. 执行动作 (频率限制)
                # ==================================================
                now = time.time()
                # 确保动作执行频率不高于 MIN_MOVE_INTERVAL
                if self.auto_run and action_type is not None:
                    # 检查运动是否正在进行，防止指令堆积
                    with self.moving_lock:
                        if self.motion_in_progress:
                            # self.get_logger().debug("运动进行中，跳过本帧指令")
                            pass # 静默处理，或者取消注释以进行调试
                        else:
                            # 标记运动开始
                            self.motion_in_progress = True

                            # 定义通用的回调函数
                            def general_callback(specific_handler, *args):
                                specific_handler(*args)
                                with self.moving_lock:
                                    self.motion_in_progress = False
                                    self.last_move_time = time.time()

                            # 根据动作类型分发异步任务
                            if action_type == 'blind_z':
                                callback = lambda success, dz: general_callback(self._on_move_z_complete, success, dz)
                                self._async_move_z(dz_cmd, callback)

                            elif action_type == 'rotate_translate':
                                callback = lambda success, rx, ry, dx, dy: general_callback(self._on_rotate_translate_complete, success, rx, ry, dx, dy)
                                self._async_rotate_translate(rx_cmd, ry_cmd, dx_cmd, dy_cmd, callback)

                            elif action_type == 'advance_z':
                                callback = lambda success, dz: general_callback(self._on_move_z_complete, success, dz)
                                self._async_move_z(dz_cmd, callback)

                            elif action_type == 'retreat':
                                callback = lambda success, dz: general_callback(self._on_move_z_complete, success, dz)
                                self._async_retreat(dz_cmd, callback)

                            elif action_type == 'all': # 过渡状态
                                callback = lambda success, rx, ry, dx, dy, dz: general_callback(self._on_composite_move_complete, success, rx, ry, dx, dy, dz)
                                self._async_composite_move(rx_cmd, ry_cmd, dx_cmd, dy_cmd, dz_cmd, callback)
                            
                            else:
                                # 如果没有匹配的动作，立即释放锁
                                self.motion_in_progress = False


                # ==================================================
                # 8. 可视化叠加
                # ==================================================
                if vis_frame is not None:
                    # 确保 vision._visualize_result 接受 filtered_goal 参数
                    vis_frame = self.vision._visualize_result(
                        vis_frame, filtered_force,
                        raw_goal, self.vision._find_regions(self.vision._preprocess_image(vis_frame))[0], 
                        self.vision._find_regions(self.vision._preprocess_image(vis_frame))[1],
                        filtered_goal=goal
                    )
                    self.draw_ui(vis_frame, force_mag, pixel_dist, status_msg)
                    
                    with self.data_lock:
                        self.display_frame = vis_frame.copy()

            except Exception as e:
                    # <<< 关键：捕获任何异常并打印，这样我们就知道线程出了什么问题 >>>
                self.get_logger().error(f"处理线程 'processing_loop' 发生致命错误: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                time.sleep(1) # 避免错误信息刷屏

        self.get_logger().info("处理线程已停止。")
    def _async_move_z(self, dz_cmd, callback=None):
        """异步执行 Z 轴移动，完成后调用 callback(success, dz)"""
        def move_and_callback():
            success = self.robot.move_z_only(dz_cmd)
            if callback:
                callback(success, dz_cmd)
        future = self.task_executor.submit(move_and_callback)
        return future

    def _async_rotate_translate(self, rx, ry, dx, dy, callback=None):
        """异步执行旋转+平移，完成后调用 callback(success, rx, ry, dx, dy)"""
        def move_and_callback():
            # 注意：原始逻辑中 rotate_translate 是分别调用 rotate_tool_frame 和 move_xy
            # 为了保持一致性，这里也分别调用（也可以合并成一次复合运动）
            success_rot = True
            success_trans = True
            if abs(rx) > 1e-6 or abs(ry) > 1e-6:
                success_rot = self.robot.rotate_tool_frame(rx, ry, 0.0)
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                success_trans = self.robot.move_xy(dx, dy)
            success = success_rot and success_trans
            if callback:
                callback(success, rx, ry, dx, dy)
        self.task_executor.submit(move_and_callback)

    def _async_retreat(self, dz_cmd, callback=None):
        """异步后退，与 move_z 类似"""
        self._async_move_z(dz_cmd, callback)

    def _on_move_z_complete(self, success, dz):
        if not success:
            self.get_logger().warn(f"Z轴移动失败: dz={dz:.3f}")
            return
        with self.moving_lock:
            self.current_depth += abs(dz)
            # 如果是后退（负值），确保深度不小于0
            if dz < 0: # retreat_step_mm 是负值
                self.current_depth = max(0.0, self.current_depth)
                self.blocked_start_time = None # 后退成功，重置阻塞计时器
            
            # 盲走距离更新（仅在盲走阶段）
            if self.current_state == SystemState.BLIND_ENTRY:
                self.blind_entry_distance += abs(dz)

        self.get_logger().debug(f"Z轴移动完成，当前深度={self.current_depth:.2f}mm")

    def _on_rotate_translate_complete(self, success, rx, ry, dx, dy):
        if not success:
            self.get_logger().warn(f"旋转平移失败: rx={rx:.3f} ry={ry:.3f} dx={dx:.3f} dy={dy:.3f}")
            return
        with self.moving_lock:
            self.current_rotation_rx += abs(rx)
            self.current_rotation_ry += abs(ry)
            self.total_rx_deg += abs(rx)
            self.total_ry_deg += abs(ry)
            self.current_translation_dx += dx
            self.current_translation_dy += dy
            self.total_translation_dx += dx
            self.total_translation_dy += dy
        self.get_logger().debug(f"旋转平移完成: rx={rx:.3f} ry={ry:.3f} dx={dx:.3f} dy={dy:.3f}")
        
    def _async_composite_move(self, rx, ry, dx, dy, dz, callback=None):
        """异步执行旋转+平移+Z轴移动的复合动作"""
        def move_and_callback():
            success_rot = True
            success_trans = True
            success_z = True
            # 分步执行，确保每个动作都完成
            if abs(rx) > 1e-6 or abs(ry) > 1e-6:
                success_rot = self.robot.rotate_tool_frame(rx, ry, 0.0)
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                success_trans = self.robot.move_xy(dx, dy)
            if abs(dz) > 1e-6:
                success_z = self.robot.move_z_only(dz)

            success = success_rot and success_trans and success_z
            if callback:
                callback(success, rx, ry, dx, dy, dz)
        self.task_executor.submit(move_and_callback)

    def _on_composite_move_complete(self, success, rx, ry, dx, dy, dz):
        """复合动作完成后的回调"""
        if not success:
            self.get_logger().warn(f"复合动作失败: rx={rx:.3f} ry={ry:.3f} dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}")
            return
        
        with self.moving_lock:
            # 更新旋转和平移的累积量
            if abs(rx) > 1e-6 or abs(ry) > 1e-6:
                self.current_rotation_rx += abs(rx)
                self.current_rotation_ry += abs(ry)
                self.total_rx_deg += abs(rx)
                self.total_ry_deg += abs(ry)
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                self.current_translation_dx += dx
                self.current_translation_dy += dy
                self.total_translation_dx += dx
                self.total_translation_dy += dy
            # 更新深度
            if abs(dz) > 1e-6:
                self.current_depth += abs(dz)
        self.get_logger().debug("复合动作完成")

    def process_frame(self):
        if not self.cap.isOpened() or self.system_terminated:
            return

        # 1. 从摄像头读取最新帧并存入共享变量
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("无法读取视频帧")
            return
            
        with self.data_lock:
            self.latest_frame = frame
            frame_to_show = self.display_frame

        # 如果处理线程还没生成第一帧，就显示原始帧
        if frame_to_show is None:
            frame_to_show = frame

        # 2. 显示处理好的帧 (非阻塞)
        cv2.imshow("Inside Navigation", frame_to_show)

        # 3. 键盘处理 (保持在主线程，以确保UI响应)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.system_terminated = True
            self.processing_thread_running = False # 通知处理线程退出
            self.get_logger().info("用户请求退出。")
            return
        elif key == ord(' '):  # 空格：启动 / 暂停
            self.auto_run = not self.auto_run
            if self.auto_run:
                # 启动时重置所有累积量与状态
                self.current_state          = SystemState.IDLE
                self.state_frame_counter    = 0
                self.current_depth          = 0.0
                self.blind_entry_completed  = False
                self.blind_entry_distance   = 0.0
                self.blind_goal_consecutive = 0
                self.blind_last_step_time   = 0.0
                self.total_rx_deg           = 0.0
                self.total_ry_deg           = 0.0
                self.total_translation_dx   = 0.0
                self.total_translation_dy   = 0.0
                self.current_rotation_rx    = 0.0
                self.current_rotation_ry    = 0.0
                self.current_translation_dx = 0.0
                self.current_translation_dy = 0.0
                self.blocked_start_time     = None
                self.goal_kalman_filter.reset()
                self.force_x_filter.reset()
                self.force_y_filter.reset()
                self.dist_filter.reset()
                self.get_logger().info(">>> 自动导航已启动 <<<")
            else:
                self.get_logger().info(">>> 自动导航已暂停 <<<")
        elif key == ord('r') or key == ord('R'):
            if self.current_state == SystemState.MAX_DEPTH_REACHED:
                self.current_depth  = 0.0
                self.current_state  = SystemState.IDLE
                self.state_frame_counter = 0
                self.get_logger().info(">>> 深度上限已手动重置，系统回到 IDLE <<<")
            else:
                self.current_depth = 0.0
                self.get_logger().info(f">>> 深度计数已重置为 0 <<<")

             
    def draw_ui(self, frame, force_mag, pixel_dist, status_msg):
        h_img, w_img = frame.shape[:2]

        # 状态与深度
        cv2.putText(frame,
                    f"State: {self.current_state.name}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame,
                    f"Depth: {self.current_depth:.1f}/{self.MAX_DEPTH_MM}mm",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame,
                    f"Dist: {pixel_dist:.1f}px  F: {force_mag:.1f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame,
                    status_msg,
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        # 旋转累积信息
        cv2.putText(frame,
                    f"Rot: Rx={self.current_rotation_rx:.3f}° Ry={self.current_rotation_ry:.3f}° "
                    f"(max={self.MAX_TOTAL_ROTATION_DEG}°)",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        # 平移累积信息
        cv2.putText(frame,
                    f"Trans: dx={self.current_translation_dx:.3f}mm "
                    f"dy={self.current_translation_dy:.3f}mm",
                    (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        # 总累积信息
        cv2.putText(frame,
                    f"Total: Rx={self.total_rx_deg:.2f}° Ry={self.total_ry_deg:.2f}° "
                    f"dx={self.total_translation_dx:.2f}mm dy={self.total_translation_dy:.2f}mm",
                    (10, 171), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)

        # 运行状态指示
        run_color = (0, 255, 0) if self.auto_run else (0, 0, 255)
        run_text  = "AUTO" if self.auto_run else "PAUSED"
        cv2.putText(frame,
                    run_text,
                    (w_img - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, run_color, 2)

        # 深度进度条
        bar_x, bar_y, bar_w, bar_h = 10, h_img - 20, w_img - 20, 12
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * min(self.current_depth / max(self.MAX_DEPTH_MM, 1e-6), 1.0))
        bar_color = (0, 200, 255) if self.current_depth < self.MAX_DEPTH_MM * 0.8 else (0, 80, 255)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

        # MAX_DEPTH_REACHED 警告
        if self.current_state == SystemState.MAX_DEPTH_REACHED:
            cv2.putText(frame,
                        "MAX DEPTH REACHED  Press R to reset",
                        (w_img // 2 - 200, h_img // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def destroy_node(self):
        # 确保处理线程先停止
        self.processing_thread_running = False
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        # 1. 先执行需要ROS通信的清理
        if hasattr(self, 'robot'):
            self.robot.disconnect()

        # 2. 然后再关闭摄像头和窗口
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        # 3. 最后关闭两个执行器
        # <<< 新增：关闭我们自己的任务执行器 >>>
        if hasattr(self, 'task_executor'):
            self.task_executor.shutdown(wait=True) # wait=True 确保所有挂起的移动任务完成或被取消

        # 关闭ROS的执行器
        if hasattr(self, 'executor'):
            self.executor.shutdown()

        super().destroy_node()
def main(args=None):
    rclpy.init(args=args)
    node = MoveInsideNode()
    
    # 将 executor 命名为 ros_executor 以示区分
    ros_executor = rclpy.executors.MultiThreadedExecutor()
    ros_executor.add_node(node)
    
    # 把它存入node实例，以便在 destroy_node 中访问
    node.ros_executor = ros_executor
    
    spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()

    # 新增：启动后台处理线程
    node.processing_thread_running = True
    node.processing_thread = threading.Thread(target=node._processing_loop, daemon=True)
    node.processing_thread.start()

    try:
        while rclpy.ok() and not node.system_terminated:
            # 主线程现在只做显示和键盘输入，非常快
            node.process_frame()
            # 可以加一个小的延时，避免CPU空转
            # time.sleep(0.01) # 大约100fps
    except KeyboardInterrupt:
        pass
    finally:
        node.processing_thread_running = False # 确保线程退出
        node.destroy_node() # destroy_node 现在会等待处理线程
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
        
if __name__ == '__main__':
    main()
