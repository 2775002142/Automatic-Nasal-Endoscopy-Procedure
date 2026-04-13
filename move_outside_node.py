import sys
import rclpy
import threading
from rclpy.node import Node
import cv2
import math
import time
import numpy as np
import concurrent.futures # <<< 新增：导入并发库
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# 导入自定义消息和服务
from fairino_msgs.srv import RemoteCmdInterface
from fairino_msgs.msg import RobotNonrtState
# 导入你的模块
from fr5_vision_control.vision_system import VisionSystem
from fr5_vision_control.robot_controller import RobotController
from fr5_vision_control.utils import SystemState, EMAFilter, clamp

class MoveOutsideNode(Node):
    def __init__(self):
        super().__init__('move_outside_node')

        # === 1. 声明和获取参数 ===
        self.declare_parameter('video_idx', '0')
        self.declare_parameter('robot_ip', '192.168.58.2')
        self.declare_parameter('simulate', True)

        self.declare_parameter('align_tolerance_enter', 20.0)
        self.declare_parameter('align_tolerance_exit', 10.0)
        self.declare_parameter('min_align_frames', 3)
        self.declare_parameter('min_approach_frames', 3)
        self.declare_parameter('transition_frames', 5)

        raw_idx = self.get_parameter('video_idx').get_parameter_value().string_value
        self.video_idx = int(raw_idx) if raw_idx.isdigit() else raw_idx
        self.simulate = self.get_parameter('simulate').get_parameter_value().bool_value

        self.get_logger().info(f'启动参数 -> video_idx: {self.video_idx}, simulate: {self.simulate}')

        # === 2. 系统控制参数 ===
        # <<< 新增：异步执行基础设施，与 inside_node 对齐 >>>
        self.task_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.moving_lock = threading.Lock()
        self.motion_in_progress = False

        self.NOSTRIL_DISTANCE_MM = 12.0
        self.XY_TOLERANCE = 10
        self.TARGET_WIDTH_THRESHOLD = 300
        
        self.Z_APPROACH_STEP = 2.0
        self.XY_MAX_STEP_MM = 2.0
        self.Z_MAX_STEP_MM = 2.0
        
        self.MAX_Z_TOTAL_MM = 150.0
        self.RETREAT_STEP_MM = 20.0
        self.MAX_JUMP_PX = 90
        self.MAX_WIDTH_JUMP_RATIO = 0.40
        
        self.LOST_TIMEOUT_SEC = 3.0
        self.FILTER_RESET_TIMEOUT = 2.0
        self.TARGET_SELECTION = 'left'

        self.XY_DAMPING = 0.6
        self.MIN_MOVE_INTERVAL = 0.1

        self.MIN_EFFECTIVE_MOVE_MM = 0.1      # 最小有效移动距离（mm）
        self.consecutive_move_fails = 0
        self.MAX_CONSECUTIVE_FAILS = 3

        # 加载迟滞与过渡参数
        self.align_tolerance_enter = self.get_parameter('align_tolerance_enter').value
        self.align_tolerance_exit = self.get_parameter('align_tolerance_exit').value
        self.min_align_frames = self.get_parameter('min_align_frames').value
        self.min_approach_frames = self.get_parameter('min_approach_frames').value
        self.transition_frames = self.get_parameter('transition_frames').value

        # === 3. 初始化子系统 ===
        self.vision = VisionSystem()
        self.robot = RobotController(self, simulate=self.simulate)

        self.cap = cv2.VideoCapture(self.video_idx)
        if not self.cap.isOpened():
            self.get_logger().error(f"摄像头 {self.video_idx} 打开失败！")
            self.system_terminated = True # <<< 修改：初始化失败时设置终止标志
            return

        cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)

        # === 4. 状态机与全局变量 ===
        self.auto_run = False
        self.finished = False
        self.system_terminated = False
        self.current_state = SystemState.IDLE

        # <<< 新增：用于在线程间传递图像数据的变量和锁 >>>
        self.data_lock = threading.Lock()
        self.latest_frame = None
        self.display_frame = None
        self.processing_thread_running = False

        self.pos_filter = EMAFilter(alpha=0.35)
        self.w_filter = EMAFilter(alpha=0.25)
        self.nostril_filter = EMAFilter(alpha=0.25)

        self.last_raw_pos = None
        self.last_raw_w = None
        self.lost_start_time = None
        self.had_target_before = False
        self.filter_reset_done = False

        self.z_total_moved = 0.0
        self.retreat_attempted = False
        self.last_move_time = 0.0

        self.state_frame_counter = 0

        self.get_logger().info(">>> 系统初始化完成 <<<")
        self.get_logger().info("请在弹出的图像窗口中按 [空格] 启动自动运动，按 [Q] 退出程序")

    # <<< 新增：异步移动函数 >>>
    def _async_move_offset(self, dx, dy, dz, callback=None):
        """异步执行工具坐标系下的偏移移动"""
        def move_and_callback():
            success = self.robot.move_offset_tool_frame(dx, dy, dz)
            if callback:
                callback(success, dx, dy, dz)
        self.task_executor.submit(move_and_callback)

    # <<< 新增：移动完成的回调函数 >>>
    def _on_move_complete(self, success, dx, dy, dz):
        """移动完成后的回调，用于更新状态"""
        if not success:
            self.get_logger().error(f"机械臂移动失败: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
            self.consecutive_move_fails += 1
            if self.consecutive_move_fails >= self.MAX_CONSECUTIVE_FAILS:
                self.get_logger().error("移动连续失败次数达到上限，自动停止运行！")
                self.auto_run = False
                self.consecutive_move_fails = 0
            return
        else:
            self.consecutive_move_fails = 0
            self.get_logger().info(f"移动成功: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")

        with self.moving_lock:
            if dz > 0:
                self.z_total_moved += dz
            elif dz < 0:
                self.z_total_moved += dz
                self.z_total_moved = max(0.0, self.z_total_moved)
        self.get_logger().debug(f"移动完成, Z累计: {self.z_total_moved:.2f}mm")

    # <<< 新增：核心逻辑处理循环，将在后台线程运行 >>>
    def _processing_loop(self):
        while self.processing_thread_running and rclpy.ok():
            try:
                with self.data_lock:
                    frame_to_process = self.latest_frame
                
                if frame_to_process is None:
                    time.sleep(0.01)
                    continue

                eff_frame, _ = self.vision.crop_effective_area(frame_to_process)
                if eff_frame is None: eff_frame = frame_to_process.copy()
                h, w = eff_frame.shape[:2]
                cam_center_x, cam_center_y = w // 2, h // 2

                # 1. 目标检测
                nose_pos, nose_width, nostril_px = self.vision.detect_nose_target(eff_frame, target_side=self.TARGET_SELECTION)

                # 2. 异常帧过滤
                valid_obs = False
                if nose_pos is not None and nose_width > 1.0 and nostril_px > 1.0:
                    jump = 0.0
                    w_jump_ratio = 0.0
                    if self.last_raw_pos is not None:
                        jump = math.hypot(nose_pos[0] - self.last_raw_pos[0], nose_pos[1] - self.last_raw_pos[1])
                    if self.last_raw_w is not None and self.last_raw_w > 1.0:
                        w_jump_ratio = abs(nose_width - self.last_raw_w) / self.last_raw_w
                    if jump <= self.MAX_JUMP_PX and w_jump_ratio <= self.MAX_WIDTH_JUMP_RATIO:
                        valid_obs = True

                filtered_pos, filtered_w, filtered_nostril, dynamic_pixel_to_mm = None, None, None, 0.05
                if valid_obs:
                    self.last_raw_pos, self.last_raw_w = nose_pos, nose_width
                    filtered_pos = self.pos_filter.update([nose_pos[0], nose_pos[1]])
                    filtered_w = float(self.w_filter.update([nose_width])[0])
                    filtered_nostril = float(self.nostril_filter.update([nostril_px])[0])
                    if filtered_nostril > 5.0:
                        dynamic_pixel_to_mm = self.NOSTRIL_DISTANCE_MM / filtered_nostril
                    self.lost_start_time, self.filter_reset_done, self.had_target_before = None, False, True
                else:
                    nose_pos, nose_width, nostril_px = None, 0.0, 0.0

                err_x = int(filtered_pos[0]) - cam_center_x if filtered_pos is not None else 0
                err_y = int(filtered_pos[1]) - cam_center_y if filtered_pos is not None else 0


                # 3. 状态机更新（带过渡与帧计数防抖）
                if self.finished:
                    new_state = SystemState.TARGET_REACHED
                elif not self.auto_run:
                    new_state = SystemState.IDLE
                elif filtered_pos is None:
                    new_state = SystemState.TARGET_LOST
                else:
                    # 计算当前误差（像素）
                    dist_err = math.hypot(err_x, err_y)

                    # --- 稳态切换逻辑（带迟滞阈值与帧计数保护）---
                    if self.current_state == SystemState.APPROACH_Z:
                        # 进近状态下，需要误差大于 enter 阈值才退回对齐，且必须满足最小进近帧数
                        if dist_err > self.align_tolerance_enter and self.state_frame_counter >= self.min_approach_frames:
                            new_state = SystemState.TRANSITION_TO_ALIGN
                        else:
                            new_state = SystemState.APPROACH_Z

                    elif self.current_state == SystemState.ALIGN_XY:
                        # 对齐状态下，误差小于 exit 阈值且帧数达标，才允许进入进近
                        if dist_err <= self.align_tolerance_exit and self.state_frame_counter >= self.min_align_frames:
                            if filtered_w < self.TARGET_WIDTH_THRESHOLD:
                                new_state = SystemState.TRANSITION_TO_APPROACH
                            else:
                                new_state = SystemState.TARGET_REACHED
                        else:
                            new_state = SystemState.ALIGN_XY

                    elif self.current_state in [SystemState.IDLE, SystemState.TARGET_LOST]:
                        # 从空闲或丢失恢复时，直接根据误差判断进入哪个稳态
                        if dist_err > self.align_tolerance_exit:
                            new_state = SystemState.ALIGN_XY
                        else:
                            new_state = SystemState.APPROACH_Z

                    # --- 过渡状态内部处理 ---
                    elif self.current_state == SystemState.TRANSITION_TO_APPROACH:
                        if self.state_frame_counter >= self.transition_frames:
                            new_state = SystemState.APPROACH_Z
                        else:
                            new_state = SystemState.TRANSITION_TO_APPROACH

                    elif self.current_state == SystemState.TRANSITION_TO_ALIGN:
                        if self.state_frame_counter >= self.transition_frames:
                            new_state = SystemState.ALIGN_XY
                        else:
                            new_state = SystemState.TRANSITION_TO_ALIGN

                    else:
                        # 安全回退：若状态未定义，根据误差默认进入对齐
                        new_state = SystemState.ALIGN_XY if dist_err > self.align_tolerance_exit else SystemState.APPROACH_Z

                # 状态更新与帧计数器管理
                if new_state != self.current_state:
                    self.get_logger().info(f"状态切换: {self.current_state.name} -> {new_state.name}")
                    self.current_state = new_state
                    self.state_frame_counter = 0
                else:
                    self.state_frame_counter += 1

                # 重置运动指令变量
                move_x, move_y, move_z = 0.0, 0.0, 0.0
                action_type = None

                # 4. 根据状态计算运动指令
                if self.current_state == SystemState.TRANSITION_TO_APPROACH:
                    # 计算目标XY移动量（复用 ALIGN_XY 的计算逻辑）
                    dist_err = math.hypot(err_x, err_y)
                    adaptive_damping = self.XY_DAMPING * 0.7 if dist_err < 50 else self.XY_DAMPING
                    target_move_x = err_x * dynamic_pixel_to_mm * adaptive_damping
                    target_move_y = err_y * dynamic_pixel_to_mm * adaptive_damping
                    target_move_x = clamp(target_move_x, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    target_move_y = clamp(target_move_y, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)

                    # 过渡：XY逐渐淡出，Z逐渐淡入
                    ratio = min(self.state_frame_counter / self.transition_frames, 1.0)
                    move_x = target_move_x * (1.0 - ratio)
                    move_y = target_move_y * (1.0 - ratio)
                    move_z = self.Z_APPROACH_STEP * ratio
                    action_type = 'move'

                elif self.current_state == SystemState.TRANSITION_TO_ALIGN:
                    # 计算目标XY移动量（同上）
                    dist_err = math.hypot(err_x, err_y)
                    adaptive_damping = self.XY_DAMPING * 0.7 if dist_err < 50 else self.XY_DAMPING
                    target_move_x = err_x * dynamic_pixel_to_mm * adaptive_damping
                    target_move_y = err_y * dynamic_pixel_to_mm * adaptive_damping
                    target_move_x = clamp(target_move_x, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    target_move_y = clamp(target_move_y, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)

                    # 过渡：XY逐渐淡入，Z归零
                    ratio = min(self.state_frame_counter / self.transition_frames, 1.0)
                    move_x = target_move_x * ratio
                    move_y = target_move_y * ratio
                    move_z = 0.0
                    action_type = 'move'

                elif self.current_state == SystemState.ALIGN_XY:
                    dist_err = math.hypot(err_x, err_y)
                    adaptive_damping = self.XY_DAMPING * 0.7 if dist_err < 50 else self.XY_DAMPING
                    target_move_x = err_x * dynamic_pixel_to_mm * adaptive_damping
                    target_move_y = err_y * dynamic_pixel_to_mm * adaptive_damping
                    move_x = clamp(target_move_x, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    move_y = clamp(target_move_y, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    if abs(move_x) > self.MIN_EFFECTIVE_MOVE_MM or abs(move_y) > self.MIN_EFFECTIVE_MOVE_MM:
                        action_type = 'move'

                elif self.current_state == SystemState.APPROACH_Z:
                    width_diff = self.TARGET_WIDTH_THRESHOLD - filtered_w
                    if width_diff > 60: actual_step = self.Z_APPROACH_STEP
                    elif width_diff > 25: actual_step = self.Z_APPROACH_STEP * 0.5
                    else: actual_step = 1.0
                    
                    if self.z_total_moved + actual_step >= self.MAX_Z_TOTAL_MM:
                        self.finished = True
                        self.get_logger().warn(f"深度安全限制触发 ({self.z_total_moved}mm), 已停止")
                    else:
                        move_z = clamp(actual_step, 0.0, self.Z_MAX_STEP_MM)
                        if abs(move_z) > self.MIN_EFFECTIVE_MOVE_MM:
                            action_type = 'move'

                elif self.current_state == SystemState.TARGET_LOST:
                    if self.auto_run and self.had_target_before:
                        if self.lost_start_time is None:
                            self.lost_start_time = time.time()
                        lost_dt = time.time() - self.lost_start_time
                        if lost_dt >= self.LOST_TIMEOUT_SEC and not self.retreat_attempted:
                            move_z = -self.RETREAT_STEP_MM
                            self.retreat_attempted = True
                            action_type = 'retreat'

                elif self.current_state == SystemState.TARGET_REACHED:
                    # 已到达目标，不产生任何运动指令
                    action_type = None

                # 5. 异步执行移动
                now = time.time()
                if self.auto_run and action_type and (now - self.last_move_time >= self.MIN_MOVE_INTERVAL):
                    with self.moving_lock:
                        if not self.motion_in_progress:
                            self.motion_in_progress = True

                            def general_callback(success, dx, dy, dz):
                                self._on_move_complete(success, dx, dy, dz)
                                with self.moving_lock:
                                    self.motion_in_progress = False
                                    self.last_move_time = time.time()
                            
                            self._async_move_offset(move_x, move_y, move_z, general_callback)

                # 6. 可视化
                cv2.line(eff_frame, (cam_center_x - 10, cam_center_y), (cam_center_x + 10, cam_center_y), (0, 255, 0), 2)
                cv2.line(eff_frame, (cam_center_x, cam_center_y - 10), (cam_center_x, cam_center_y + 10), (0, 255, 0), 2)
                if filtered_pos is not None:
                    cv2.circle(eff_frame, (int(filtered_pos[0]), int(filtered_pos[1])), 5, (0, 0, 255), -1)

                cv2.putText(eff_frame, f"State: {self.current_state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,100,0), 2)
                cv2.putText(eff_frame, f"Auto: {self.auto_run} Finished: {self.finished}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                cv2.putText(eff_frame, f"Z Moved: {self.z_total_moved:.1f}mm / {self.MAX_Z_TOTAL_MM}mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,100), 1)
                if filtered_pos is not None:
                    cv2.putText(eff_frame, f"Err(px): X={err_x}, Y={err_y}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
                    cv2.putText(eff_frame, f"Width: {int(filtered_w)}px", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
                    cv2.putText(eff_frame, f"Pixel->MM: {dynamic_pixel_to_mm:.4f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,255), 1)
                
                with self.data_lock:
                    self.display_frame = eff_frame.copy()

            except Exception as e:
                self.get_logger().error(f"处理线程 'processing_loop' 发生致命错误: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                time.sleep(1)

        self.get_logger().info("处理线程已停止。")
    
    # <<< 修改：`process_frame` 现在只负责UI和摄像头 >>>
    def process_frame(self):
        if not self.cap.isOpened() or self.system_terminated:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("无法读取视频帧")
            return
            
        with self.data_lock:
            self.latest_frame = frame
            frame_to_show = self.display_frame

        if frame_to_show is None:
            frame_to_show = frame

        cv2.imshow("Robot View", frame_to_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.system_terminated = True
            self.processing_thread_running = False
            self.get_logger().info("用户请求退出。")
            return
        elif key == ord(' '):
            if not self.finished:
                self.auto_run = not self.auto_run
                self.get_logger().info(f"自动运行状态切换 -> {self.auto_run}")
                if self.auto_run:
                    self.lost_start_time = None
                    self.retreat_attempted = False
                    # 重置状态以重新开始
                    self.current_state = SystemState.IDLE
                    # 可以考虑在这里重置更多状态变量
            else:
                self.get_logger().info("运动已完成或系统终止，按Q退出")


    # <<< 修改：destroy_node 需要关闭线程池和处理线程 >>>
    def destroy_node(self):
        self.get_logger().info("释放资源中...")
        self.processing_thread_running = False
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        if hasattr(self, 'task_executor'):
            self.task_executor.shutdown(wait=False)

        super().destroy_node()

# <<< 修改：main 函数结构，与 inside_node 对齐 >>>
def main(args=None):
    rclpy.init(args=args)
    node = MoveOutsideNode()
    
    if node.system_terminated: # 如果摄像头初始化失败，直接退出
        node.get_logger().error("节点初始化失败，即将退出。")
        node.destroy_node()
        rclpy.shutdown()
        return

    ros_executor = rclpy.executors.MultiThreadedExecutor()
    ros_executor.add_node(node)
    
    spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()

    # 启动后台处理线程
    node.processing_thread_running = True
    node.processing_thread = threading.Thread(target=node._processing_loop, daemon=True)
    node.processing_thread.start()

    try:
        while rclpy.ok() and not node.system_terminated:
            node.process_frame()
            time.sleep(0.01) # 避免主线程CPU空转
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("开始关闭节点...")
        node.system_terminated = True
        node.processing_thread_running = False
        node.destroy_node()
        
        ros_executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        
        if spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
        node.get_logger().info('完全退出')

if __name__ == '__main__':
    main()