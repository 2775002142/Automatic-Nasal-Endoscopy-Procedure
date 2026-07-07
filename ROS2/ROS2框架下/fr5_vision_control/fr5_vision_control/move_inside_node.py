"""Inside-navigation ROS 2 node (体内导航).

Refactored: algorithm logic lives in ``nasal_endoscopy_algorithms``;
this node *only* handles ROS communication, camera I/O, and motion dispatch.
``RobotController`` is called exactly as before — its four-step sequence
is preserved verbatim.
"""

import sys
import threading
import time
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node, SetParametersResult
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

# ── ROS image transport ──
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ── Custom messages ──
from nasal_endoscopy_msgs.msg import APFForceField, NavigationState, MotionCommand as MotionCmdMsg, MotionResult

# ── Diagnostics ──
from diagnostic_msgs.msg import DiagnosticStatus

# ── Action Server ──
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from nasal_endoscopy_msgs.action import NavigateInside

# ── Algorithm package (zero-ROS) ──
from nasal_endoscopy_algorithms.vision.inside_vision import (
    APFVisionSystem, APFResult,
)
from nasal_endoscopy_algorithms.control.force_to_motion import (
    ForceToMotionConverter, MotionCommand,
)
from nasal_endoscopy_algorithms.filters.ema_filter import EMAFilter
from nasal_endoscopy_algorithms.filters.kalman_filter_2d import KalmanFilter2D
from nasal_endoscopy_algorithms.utils.state_enums import SystemState
from nasal_endoscopy_algorithms.state_machine.inside_sm import (
    InsideStateMachine, InsideSMConfig,
)

# ── Robot communication done via ROS topic → motion_executor ──


class _MjpegStreamReader:
    """Read MJPEG frames from a Flask / multipart HTTP stream.

    Drop-in replacement for ``cv2.VideoCapture(url)`` — provides
    ``.read()``, ``.isOpened()``, and ``.release()``.
    """

    def __init__(self, url, timeout=3.0, logger=None):
        import urllib.request
        self._url = url
        self._timeout = timeout
        self._log = logger
        self._stream = None
        self._buffer = b""
        self._opened = False

        try:
            self._stream = urllib.request.urlopen(url, timeout=timeout)
            content_type = self._stream.headers.get('Content-Type', '')
            if 'multipart' in content_type or 'mixed-replace' in content_type:
                self._boundary = None
                for part in content_type.split(';'):
                    part = part.strip()
                    if part.startswith('boundary='):
                        self._boundary = part.split('=', 1)[1].encode()
                        break
                if self._boundary is None:
                    self._boundary = b'frame'
                self._opened = True
            else:
                self._boundary = None
                self._opened = True
        except Exception as e:
            if self._log:
                self._log.error(f"MJPEG 流打开失败: {e}")

    def isOpened(self):
        return self._opened

    def read(self):
        import numpy as np
        if not self._opened or self._stream is None:
            return False, None
        try:
            if self._boundary:
                return self._read_multipart()
            else:
                return self._read_raw()
        except Exception:
            return False, None

    def _read_multipart(self):
        import numpy as np
        max_attempts = 20
        for _ in range(max_attempts):
            chunk = self._stream.read(4096)
            if not chunk:
                return False, None
            self._buffer += chunk
            while True:
                b_start = self._buffer.find(b'--' + self._boundary + b'\r\n')
                if b_start == -1:
                    b_start = self._buffer.find(b'--' + self._boundary + b'\n')
                if b_start == -1:
                    break
                after_start = self._buffer[b_start:]
                header_end = after_start.find(b'\r\n\r\n')
                if header_end == -1:
                    break
                body_start = b_start + header_end + 4
                b_next = self._buffer.find(b'--' + self._boundary, body_start)
                if b_next == -1:
                    break
                jpeg_data = self._buffer[body_start:b_next]
                while jpeg_data.endswith(b'\r') or jpeg_data.endswith(b'\n'):
                    jpeg_data = jpeg_data[:-1]
                self._buffer = self._buffer[b_next:]
                if len(jpeg_data) < 100:
                    continue
                frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
        return False, None

    def _read_raw(self):
        import numpy as np
        max_attempts = 20
        for _ in range(max_attempts):
            chunk = self._stream.read(4096)
            if not chunk:
                return False, None
            self._buffer += chunk
            soi = self._buffer.find(b'\xff\xd8')
            eoi = self._buffer.find(b'\xff\xd9', soi + 2)
            if soi >= 0 and eoi > soi:
                jpeg_data = self._buffer[soi:eoi + 2]
                self._buffer = self._buffer[eoi + 2:]
                frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
        return False, None

    def release(self):
        self._opened = False
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = None


class MoveInsideNode(Node):
    """ROS 2 node for inside (lumen-following) navigation."""

    def __init__(self):
        super().__init__('move_inside_node')

        # ── 1. Parameters ───────────────────────────────────
        self.declare_parameter('video_idx', '0')
        self.declare_parameter('simulate', True)
        self.declare_parameter('max_depth_mm', 50.0)
        self.declare_parameter('align_dist_start_px', 180.0)
        self.declare_parameter('align_dist_stop_px', 130.0)

        self.declare_parameter('min_rotate_frames', 2)
        self.declare_parameter('min_advance_frames', 5)
        self.declare_parameter('transition_frames', 10)

        self.declare_parameter('z_advance_step_mm', 0.5)
        self.declare_parameter('retreat_step_mm', -0.5)

        self.declare_parameter('blocked_timeout_sec', 1.5)
        self.declare_parameter('ema_alpha', 0.25)
        self.declare_parameter('kalman_process_noise', 0.05)
        self.declare_parameter('kalman_measure_noise', 4.0)

        self.declare_parameter('kalman_predict_max_frames', 5)
        self.declare_parameter('kalman_predict_force_scale', 0.3)

        self.declare_parameter('blind_entry_step_mm', 2.0)
        self.declare_parameter('blind_entry_max_mm', 20.0)
        self.declare_parameter('blind_entry_interval_sec', 0.15)
        self.declare_parameter('blind_goal_confirm_frames', 20)

        self.declare_parameter('max_total_rotation_deg', 2.0)

        self.declare_parameter('converter_force_deadzone', 10.0)
        self.declare_parameter('converter_max_force_for_scale', 1150.0)
        self.declare_parameter('converter_min_rotation_gain', 0.001)
        self.declare_parameter('converter_max_rotation_gain', 0.012)
        self.declare_parameter('converter_rotation_gain_curve_factor', 0.6)
        self.declare_parameter('converter_max_rotation_deg', 0.2)
        self.declare_parameter('converter_min_translate_step_mm', 0.02)
        self.declare_parameter('converter_max_translate_step_mm', 0.15)
        self.declare_parameter('converter_translate_step_curve_factor', 0.7)
        self.declare_parameter('converter_max_translate_per_phase_mm', 1.0)

        # ── 2. Resolve params ───────────────────────────────
        video_param = self.get_parameter('video_idx').get_parameter_value().string_value
        self.video_idx = int(video_param) if video_param.isdigit() else video_param
        self.simulate = self.get_parameter('simulate').get_parameter_value().bool_value

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
        self.KALMAN_PREDICT_MAX_FRAMES = self.get_parameter('kalman_predict_max_frames').get_parameter_value().integer_value
        self.KALMAN_PREDICT_FORCE_SCALE = self.get_parameter('kalman_predict_force_scale').get_parameter_value().double_value

        self.get_logger().info(f'启动内部导航 -> 视频源: {self.video_idx}, 模拟: {self.simulate}')

        # ── 3. Algorithm subsystems ─────────────────────────
        self.vision = APFVisionSystem(debug=False)

        # ── Subscribe to motion results ─────────────────────
        self._motion_result_sub = self.create_subscription(
            MotionResult, '/robot/motion_result', self._motion_result_cb, 10,
        )
        self._last_motion_sent_time = 0.0

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
            max_translate_per_phase_mm=self.get_parameter('converter_max_translate_per_phase_mm').get_parameter_value().double_value,
        )

        sm_cfg = InsideSMConfig(
            align_dist_start_px=self.ALIGN_DIST_START_PX,
            align_dist_stop_px=self.ALIGN_DIST_STOP_PX,
            min_rotate_frames=self.MIN_ROTATE_FRAMES,
            min_advance_frames=self.MIN_ADVANCE_FRAMES,
            transition_frames=self.TRANSITION_FRAMES,
            blocked_timeout_sec=self.BLOCKED_TIMEOUT_SEC,
            blind_entry_max_mm=self.BLIND_ENTRY_MAX_MM,
            blind_goal_confirm_frames=self.BLIND_GOAL_CONFIRM_FRAMES,
        )
        self.state_machine = InsideStateMachine(sm_cfg)

        # ── 4. State variables ──────────────────────────────
        self.current_state = SystemState.IDLE  # safe default before processing starts
        self.auto_run = False
        self.system_terminated = False
        self.current_depth = 0.0
        self.moving_lock = threading.Lock()
        self.motion_in_progress = False

        self.goal_kalman_filter = KalmanFilter2D(
            process_noise_cov=self.KALMAN_PROCESS_NOISE,
            measurement_noise_cov=self.KALMAN_MEASURE_NOISE,
        )
        self.force_x_filter = EMAFilter(alpha=self.EMA_ALPHA)
        self.force_y_filter = EMAFilter(alpha=self.EMA_ALPHA)
        self.dist_filter = EMAFilter(alpha=self.EMA_ALPHA)

        self.goal_is_predicted = False
        self.consecutive_predicted_frames = 0

        self.blocked_start_time = None
        self.last_move_time = 0.0

        # Blind-entry tracking (distance is tracked here, not in SM)
        self.blind_entry_distance = 0.0
        self.blind_last_step_time = 0.0

        # Accumulators
        self.total_rx_deg = 0.0
        self.total_ry_deg = 0.0
        self.total_translation_dx = 0.0
        self.total_translation_dy = 0.0
        self.current_rotation_rx = 0.0
        self.current_rotation_ry = 0.0
        self.current_translation_dx = 0.0
        self.current_translation_dy = 0.0

        self.data_lock = threading.Lock()
        self.latest_frame = None
        self.display_frame = None
        self.processing_thread_running = False

        self.finished = False

        # ── FPS counter ──
        self._fps_frame_count = 0
        self._fps_last_print = 0.0

        # ── 5. Camera (with auto-detect Windows Flask server) ──
        self._mjpeg_reader = None
        self.cap = None

        def _try_open(source):
            src_str = str(source)
            if src_str.startswith('http://') or src_str.startswith('https://'):
                reader = _MjpegStreamReader(src_str, logger=self.get_logger())
                if reader.isOpened():
                    return reader, src_str
                return None, src_str
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap, src_str
            cap.release()
            return None, src_str

        cap, used_src = _try_open(self.video_idx)
        if cap is None:
            auto_url = self._detect_windows_camera()
            if auto_url:
                self.get_logger().info(f"本地摄像头失败，自动探测到 Windows Flask 视频流: {auto_url}")
                cap, used_src = _try_open(auto_url)
        if cap is None:
            self.get_logger().error(f"摄像头 {self.video_idx} 打开失败！请确认: "
                                    "1) Windows 端 Flask 已启动 http://localhost:5000/video_feed "
                                    "2) WSL 防火墙允许访问 Windows 宿主")
            self._publish_diag(DiagnosticStatus.ERROR,
                               f'Camera open failed: {self.video_idx}')
            self.system_terminated = True
            return

        self.cap = cap
        if isinstance(cap, _MjpegStreamReader):
            self._mjpeg_reader = cap
        self.get_logger().info(f"视频源已连接: {used_src}")

        cv2.namedWindow("Inside Navigation", cv2.WINDOW_NORMAL)

        # ── Action Server ──
        self._action_cb_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            NavigateInside,
            '/navigate_inside',
            execute_callback=self._execute_navigate_cb,
            cancel_callback=self._cancel_navigate_cb,
            callback_group=self._action_cb_group,
        )

        # ── ROS 2 image transport ──
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._image_pub = self.create_publisher(
            Image, '/camera/inside/image_raw', qos_profile=image_qos,
        )
        self._cv_bridge = CvBridge()

        # ── Vision / state / control publishers ──
        vision_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        control_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._apf_pub = self.create_publisher(
            APFForceField, '/vision/inside/apf', qos_profile=vision_qos,
        )
        self._nav_state_pub = self.create_publisher(
            NavigationState, '/navigation/inside/state', qos_profile=state_qos,
        )
        self._motion_cmd_pub = self.create_publisher(
            MotionCmdMsg, '/control/inside/command', qos_profile=control_qos,
        )
        self._diag_pub = self.create_publisher(
            DiagnosticStatus, '/diagnostics/update', 10,
        )
        self.get_logger().info(">>> 按 [空格] 开始/暂停，按 [R] 重置深度，按 [Q/ESC] 退出 <<<")

        # Print init summary
        self.get_logger().info("=" * 60)
        self.get_logger().info(">>> 体内导航系统 ROS 节点初始化 <<<")
        self._publish_diag(DiagnosticStatus.OK, 'Initialised')

        # ── Parameter hot-reload ──
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"  [旋转增益] {self.converter.min_rotation_gain:.4f} ~ {self.converter.max_rotation_gain:.4f} | "
            f"单次上限={self.converter.max_rotation_deg}°"
        )
        self.get_logger().info(
            f"  [平移步长] {self.converter.min_translate_step_mm:.3f} ~ {self.converter.max_translate_step_mm:.3f}mm | "
            f"每阶段上限={self.converter.max_translate_per_phase_mm}mm"
        )
        self.get_logger().info(
            f"  [切换] 偏离>{self.ALIGN_DIST_START_PX}px开始校准 | 偏离<{self.ALIGN_DIST_STOP_PX}px允许直行"
        )
        self.get_logger().info(
            f"  [防抖] 旋转最少{self.MIN_ROTATE_FRAMES}帧 | 前进最少{self.MIN_ADVANCE_FRAMES}帧 | 过渡{self.TRANSITION_FRAMES}帧"
        )
        self.get_logger().info(
            f"  [安全] 单阶段旋转上限={self.MAX_TOTAL_ROTATION_DEG}° | 最大深度={self.MAX_DEPTH_MM}mm"
        )
        self.get_logger().info("=" * 60)

    # ═══════════════════════════════════════════════════════════
    #  Diagnostics helper
    # ═══════════════════════════════════════════════════════════

    def _publish_diag(self, level: int, message: str):
        st = DiagnosticStatus()
        st.name = 'move_inside_node'
        st.level = level
        st.message = message
        st.hardware_id = 'inside_nav'
        self._diag_pub.publish(st)

    # ── Parameter hot-reload ─────────────────────────────────

    def _on_param_change(self, params):
        from rclpy.parameter import Parameter
        for p in params:
            try:
                if p.name == 'max_depth_mm':
                    self.MAX_DEPTH_MM = p.value
                elif p.name == 'align_dist_start_px':
                    self.ALIGN_DIST_START_PX = p.value
                    self.state_machine.cfg.align_dist_start_px = p.value
                elif p.name == 'align_dist_stop_px':
                    self.ALIGN_DIST_STOP_PX = p.value
                    self.state_machine.cfg.align_dist_stop_px = p.value
                elif p.name == 'min_rotate_frames':
                    self.MIN_ROTATE_FRAMES = p.value
                    self.state_machine.cfg.min_rotate_frames = p.value
                elif p.name == 'min_advance_frames':
                    self.MIN_ADVANCE_FRAMES = p.value
                    self.state_machine.cfg.min_advance_frames = p.value
                elif p.name == 'transition_frames':
                    self.TRANSITION_FRAMES = p.value
                    self.state_machine.cfg.transition_frames = p.value
                elif p.name == 'z_advance_step_mm':
                    self.Z_ADVANCE_STEP_MM = p.value
                elif p.name == 'retreat_step_mm':
                    self.RETREAT_STEP_MM = p.value
                elif p.name == 'blocked_timeout_sec':
                    self.BLOCKED_TIMEOUT_SEC = p.value
                    self.state_machine.cfg.blocked_timeout_sec = p.value
                elif p.name == 'ema_alpha':
                    self.EMA_ALPHA = p.value
                    self.force_x_filter.alpha = p.value
                    self.force_y_filter.alpha = p.value
                    self.dist_filter.alpha = p.value
                elif p.name == 'kalman_process_noise':
                    self.KALMAN_PROCESS_NOISE = p.value
                elif p.name == 'kalman_measure_noise':
                    self.KALMAN_MEASURE_NOISE = p.value
                elif p.name == 'kalman_predict_max_frames':
                    self.KALMAN_PREDICT_MAX_FRAMES = p.value
                elif p.name == 'kalman_predict_force_scale':
                    self.KALMAN_PREDICT_FORCE_SCALE = p.value
                elif p.name == 'blind_entry_step_mm':
                    self.BLIND_ENTRY_STEP_MM = p.value
                elif p.name == 'blind_entry_max_mm':
                    self.BLIND_ENTRY_MAX_MM = p.value
                    self.state_machine.cfg.blind_entry_max_mm = p.value
                elif p.name == 'blind_entry_interval_sec':
                    self.BLIND_ENTRY_INTERVAL_SEC = p.value
                elif p.name == 'blind_goal_confirm_frames':
                    self.BLIND_GOAL_CONFIRM_FRAMES = p.value
                    self.state_machine.cfg.blind_goal_confirm_frames = p.value
                elif p.name == 'max_total_rotation_deg':
                    self.MAX_TOTAL_ROTATION_DEG = p.value
                elif p.name == 'converter_force_deadzone':
                    self.converter.force_deadzone = p.value
                elif p.name == 'converter_max_force_for_scale':
                    self.converter.max_force_for_scale = p.value
                elif p.name == 'converter_min_rotation_gain':
                    self.converter.min_rotation_gain = p.value
                elif p.name == 'converter_max_rotation_gain':
                    self.converter.max_rotation_gain = p.value
                elif p.name == 'converter_rotation_gain_curve_factor':
                    self.converter.rotation_gain_curve_factor = p.value
                elif p.name == 'converter_max_rotation_deg':
                    self.converter.max_rotation_deg = p.value
                elif p.name == 'converter_min_translate_step_mm':
                    self.converter.min_translate_step_mm = p.value
                elif p.name == 'converter_max_translate_step_mm':
                    self.converter.max_translate_step_mm = p.value
                elif p.name == 'converter_translate_step_curve_factor':
                    self.converter.translate_step_curve_factor = p.value
                elif p.name == 'converter_max_translate_per_phase_mm':
                    self.converter.max_translate_per_phase_mm = p.value
            except Exception as e:
                self.get_logger().warn(f'Failed to apply param {p.name}={p.value}: {e}')
        return SetParametersResult(successful=True)

    # ═══════════════════════════════════════════════════════════
    #  Action Server callbacks
    # ═══════════════════════════════════════════════════════════

    def _cancel_navigate_cb(self, goal_handle):
        """Accept cancellation of an in-progress navigation goal."""
        self.get_logger().info('[Action] NavigateInside cancellation requested')
        self.auto_run = False
        return CancelResponse.ACCEPT

    def _execute_navigate_cb(self, goal_handle):
        """Execute a NavigateInside action goal."""
        max_depth = goal_handle.request.max_depth_mm
        self.get_logger().info(
            f'[Action] NavigateInside goal accepted: max_depth={max_depth}mm'
        )

        # Apply goal parameters
        if max_depth > 0.0:
            self.MAX_DEPTH_MM = max_depth

        # Start navigation
        self.auto_run = True
        self.current_depth = 0.0
        self.blind_entry_distance = 0.0
        self.blind_last_step_time = 0.0
        self.total_rx_deg = 0.0
        self.total_ry_deg = 0.0
        self.total_translation_dx = 0.0
        self.total_translation_dy = 0.0
        self.current_rotation_rx = 0.0
        self.current_rotation_ry = 0.0
        self.current_translation_dx = 0.0
        self.current_translation_dy = 0.0
        self.blocked_start_time = None
        self.goal_is_predicted = False
        self.consecutive_predicted_frames = 0
        self.goal_kalman_filter.reset()
        self.force_x_filter.reset()
        self.force_y_filter.reset()
        self.dist_filter.reset()
        self.state_machine.reset()

        feedback = NavigateInside.Feedback()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.auto_run = False
                self.get_logger().info('[Action] NavigateInside cancelled')
                result = NavigateInside.Result()
                result.depth_reached = False
                result.final_depth_mm = float(self.current_depth)
                result.final_state = self.current_state.name
                result.total_translation_dx = self.total_translation_dx
                result.total_translation_dy = self.total_translation_dy
                return result

            # Publish feedback at ~10 Hz
            feedback.current_state = self.current_state.name
            feedback.current_depth_mm = float(self.current_depth)
            feedback.target_dist_px = 0.0  # will be updated
            feedback.force_magnitude = 0.0
            feedback.current_rotation_rx = self.current_rotation_rx
            feedback.current_rotation_ry = self.current_rotation_ry
            goal_handle.publish_feedback(feedback)

            if self.current_state == SystemState.MAX_DEPTH_REACHED:
                break

            if not self.auto_run:  # keyboard pause
                self.get_logger().info('[Action] Navigation paused by keyboard')
                break

            time.sleep(0.1)

        depth_reached = self.current_state == SystemState.MAX_DEPTH_REACHED
        result = NavigateInside.Result()
        result.depth_reached = depth_reached
        result.final_depth_mm = float(self.current_depth)
        result.final_state = self.current_state.name
        result.total_translation_dx = self.total_translation_dx
        result.total_translation_dy = self.total_translation_dy

        if depth_reached:
            goal_handle.succeed()
            self.get_logger().info(f'[Action] NavigateInside succeeded at depth={self.current_depth:.1f}mm')
        else:
            goal_handle.abort()
            self.get_logger().info('[Action] NavigateInside aborted')

        self.auto_run = False
        return result

    # ═══════════════════════════════════════════════════════════
    #  Auto-detect Windows Flask camera server from WSL
    # ═══════════════════════════════════════════════════════════

    def _detect_windows_camera(self):
        """Try to find the Windows host running Flask camera server."""
        import subprocess
        import urllib.request

        candidates = []
        # 1) /etc/resolv.conf nameserver (usually Windows host in WSL2)
        try:
            with open('/etc/resolv.conf') as f:
                for line in f:
                    if line.startswith('nameserver'):
                        ip = line.split()[-1].strip()
                        if ip and not ip.startswith('127.'):
                            candidates.append(ip)
        except Exception:
            pass
        # 2) default route gateway
        try:
            result = subprocess.run(
                ['ip', 'route', 'show', 'default'],
                capture_output=True, text=True, timeout=2)
            for line in result.stdout.splitlines():
                parts = line.split()
                if 'via' in parts:
                    idx = parts.index('via')
                    if idx + 1 < len(parts):
                        candidates.append(parts[idx + 1])
        except Exception:
            pass
        # 3) common WSL2 gateway fallbacks
        candidates.extend([
            '172.30.224.1', '172.25.80.1', '172.28.176.1',
            '192.168.0.1', '10.0.0.1',
        ])

        for ip in candidates:
            url = f'http://{ip}:5000/video_feed'
            try:
                req = urllib.request.Request(url, method='HEAD')
                urllib.request.urlopen(req, timeout=0.8)
                return url
            except Exception:
                continue
        return None

    # ═══════════════════════════════════════════════════════════
    #  Processing loop
    # ═══════════════════════════════════════════════════════════

    def _processing_loop(self):
        while self.processing_thread_running and rclpy.ok():
            try:
                with self.data_lock:
                    frame_to_process = self.latest_frame

                if frame_to_process is None:
                    time.sleep(0.01)
                    continue

                # ── 1. Vision ──
                result: APFResult = self.vision.process_frame(frame_to_process)
                vis_frame, raw_force, raw_goal = result.vis_image, result.force_vector, result.goal

                # ── 2. Kalman goal tracking ──
                goal = None
                self.goal_is_predicted = False

                if raw_goal is not None:
                    filtered_pos = self.goal_kalman_filter.update(raw_goal)
                    goal = (int(filtered_pos[0]), int(filtered_pos[1]))
                    self.consecutive_predicted_frames = 0
                else:
                    predicted_pos = self.goal_kalman_filter.predict_only()
                    if predicted_pos is not None and self.consecutive_predicted_frames < self.KALMAN_PREDICT_MAX_FRAMES:
                        goal = (int(predicted_pos[0]), int(predicted_pos[1]))
                        self.goal_is_predicted = True
                        self.consecutive_predicted_frames += 1
                        self.get_logger().debug(
                            f"[卡尔曼预测] 第 {self.consecutive_predicted_frames}/{self.KALMAN_PREDICT_MAX_FRAMES} 帧"
                        )
                    else:
                        self.goal_kalman_filter.reset()
                        self.consecutive_predicted_frames = 0

                # ── 3. Force & distance ──
                pixel_dist = 0.0
                if goal is not None:
                    att_filtered = self.vision._calculate_attractive_force(goal)
                    if raw_goal is not None:
                        att_raw = self.vision._calculate_attractive_force(raw_goal)
                        rep_force = raw_force - att_raw
                    else:
                        rep_force = np.array([0.0, 0.0])
                    combined_force = att_filtered + rep_force

                    fx = float(self.force_x_filter.update(combined_force[0]))
                    fy = float(self.force_y_filter.update(combined_force[1]))
                    filtered_force = np.array([fx, fy])

                    if self.goal_is_predicted:
                        filtered_force = filtered_force * self.KALMAN_PREDICT_FORCE_SCALE

                    raw_dist = math.hypot(
                        goal[0] - self.vision.center[0],
                        goal[1] - self.vision.center[1],
                    )
                    pixel_dist = float(self.dist_filter.update(raw_dist))
                    self.blocked_start_time = None
                else:
                    filtered_force = np.array([0.0, 0.0])
                    self.force_x_filter.reset()
                    self.force_y_filter.reset()

                force_mag = float(np.linalg.norm(filtered_force))

                # Publish APF force field
                apf_msg = APFForceField()
                apf_msg.header.stamp = self.get_clock().now().to_msg()
                apf_msg.force_x = float(filtered_force[0])
                apf_msg.force_y = float(filtered_force[1])
                apf_msg.force_magnitude = force_mag
                apf_msg.goal_x = goal[0] if goal is not None else -1
                apf_msg.goal_y = goal[1] if goal is not None else -1
                apf_msg.goal_valid = goal is not None
                apf_msg.center_x = self.vision.center[0]
                apf_msg.center_y = self.vision.center[1]
                self._apf_pub.publish(apf_msg)

                # ── 4. Rotation-safety check ──
                current_rot = math.sqrt(self.current_rotation_rx ** 2 + self.current_rotation_ry ** 2)
                in_rotate = self.state_machine.current_state in (
                    SystemState.ROTATE_ALIGN, SystemState.TRANSITION_TO_ROTATE,
                )
                rotation_safe = (not in_rotate) or (current_rot < self.MAX_TOTAL_ROTATION_DEG)

                # ── 5. State machine ──
                self.current_state = self.state_machine.evaluate(
                    goal_exists=(goal is not None),
                    pixel_dist=pixel_dist,
                    is_auto_run=self.auto_run,
                    current_depth_mm=self.current_depth,
                    max_depth_mm=self.MAX_DEPTH_MM,
                    rotation_safe=rotation_safe,
                )

                # Publish navigation state
                nav_msg = NavigationState()
                nav_msg.header.stamp = self.get_clock().now().to_msg()
                nav_msg.state = self.current_state.name
                nav_msg.current_depth_mm = float(self.current_depth)
                nav_msg.max_depth_mm = float(self.MAX_DEPTH_MM)
                nav_msg.target_dist_px = float(pixel_dist)
                nav_msg.force_magnitude = force_mag
                nav_msg.auto_run = self.auto_run
                nav_msg.finished = (self.current_state == SystemState.MAX_DEPTH_REACHED)
                self._nav_state_pub.publish(nav_msg)

                # ── 6. Motion command computation ──
                action_type = None
                rx_cmd = ry_cmd = dx_cmd = dy_cmd = dz_cmd = 0.0
                status_msg = ""
                state = self.current_state
                fc = self.state_machine.frame_counter
                tf = self.state_machine.cfg.transition_frames
                sm = self.state_machine

                if not self.auto_run:
                    status_msg = "IDLE (已暂停)"

                elif not sm.blind_entry_completed:
                    # ── Blind entry ──
                    if sm.blind_goal_consecutive >= self.BLIND_GOAL_CONFIRM_FRAMES:
                        sm.blind_entry_completed = True
                        self.get_logger().info(
                            f"[盲走完成] 连续{self.BLIND_GOAL_CONFIRM_FRAMES}帧检测到目标，切换正常导航。"
                        )
                        self.force_x_filter.reset()
                        self.force_y_filter.reset()
                        self.current_state = SystemState.TRANSITION_TO_ROTATE
                        self.current_rotation_rx = self.current_rotation_ry = 0.0
                        self.current_translation_dx = self.current_translation_dy = 0.0
                        status_msg = "Blind->Transition to Rotate"

                    elif self.blind_entry_distance >= self.BLIND_ENTRY_MAX_MM:
                        sm.blind_entry_completed = True
                        self.get_logger().info(
                            f"[盲走上限] 已盲走{self.blind_entry_distance:.1f}mm，强制切换。"
                        )
                        self.force_x_filter.reset()
                        self.force_y_filter.reset()
                        self.current_state = SystemState.TRANSITION_TO_ROTATE
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
                    # ── Normal navigation ──
                    if state == SystemState.MAX_DEPTH_REACHED:
                        status_msg = f"MAX DEPTH {self.current_depth:.1f}mm (按R重置)"

                    elif state == SystemState.RETREAT:
                        action_type = 'retreat'
                        dz_cmd = self.RETREAT_STEP_MM
                        status_msg = "RETREAT"

                    elif state == SystemState.BLOCKED:
                        elapsed = time.time() - (sm.blocked_start_time or time.time())
                        status_msg = f"BLOCKED {elapsed:.1f}/{self.BLOCKED_TIMEOUT_SEC}s"

                    elif state == SystemState.ADVANCE_Z:
                        action_type = 'advance_z'
                        dz_cmd = self.Z_ADVANCE_STEP_MM
                        status_msg = f"ADVANCE D={pixel_dist:.1f} F={force_mag:.1f}"

                    elif state == SystemState.ROTATE_ALIGN:
                        action_type = 'rotate_translate'
                        cmd: MotionCommand = self.converter.convert(
                            filtered_force,
                            self.current_translation_dx,
                            self.current_translation_dy,
                        )
                        rx_cmd, ry_cmd, dx_cmd, dy_cmd = cmd.rx_deg, cmd.ry_deg, cmd.dx_mm, cmd.dy_mm
                        status_msg = (
                            f"ROTATE D={pixel_dist:.1f} "
                            f"Rx={rx_cmd:.3f}° Ry={ry_cmd:.3f}° "
                            f"dx={dx_cmd:.3f} dy={dy_cmd:.3f}"
                        )

                    elif state == SystemState.TRANSITION_TO_ROTATE:
                        ratio = min(fc / max(tf, 1), 1.0)
                        cmd = self.converter.convert(
                            filtered_force,
                            self.current_translation_dx,
                            self.current_translation_dy,
                        )
                        rx_cmd = cmd.rx_deg * ratio
                        ry_cmd = cmd.ry_deg * ratio
                        dx_cmd = cmd.dx_mm * ratio
                        dy_cmd = cmd.dy_mm * ratio
                        action_type = 'rotate_translate'
                        status_msg = f"Transition→ROTATE ({fc}/{tf})"
                        if fc >= tf:
                            self.get_logger().info(
                                f"过渡完成: {self.current_state.name} -> {SystemState.ROTATE_ALIGN.name}"
                            )
                            self.current_state = SystemState.ROTATE_ALIGN
                            self.state_machine.current_state = SystemState.ROTATE_ALIGN
                            self.state_machine.frame_counter = 0

                    elif state == SystemState.TRANSITION_TO_ADVANCE:
                        ratio = min(fc / max(tf, 1), 1.0)
                        cmd = self.converter.convert(
                            filtered_force,
                            self.current_translation_dx,
                            self.current_translation_dy,
                        )
                        rx_cmd = cmd.rx_deg * (1.0 - ratio)
                        ry_cmd = cmd.ry_deg * (1.0 - ratio)
                        dx_cmd = cmd.dx_mm * (1.0 - ratio)
                        dy_cmd = cmd.dy_mm * (1.0 - ratio)
                        dz_cmd = self.Z_ADVANCE_STEP_MM * ratio
                        action_type = 'all'
                        status_msg = f"Transition→ADVANCE ({fc}/{tf})"
                        if fc >= tf:
                            self.get_logger().info(
                                f"过渡完成: {self.current_state.name} -> {SystemState.ADVANCE_Z.name}"
                            )
                            self.current_state = SystemState.ADVANCE_Z
                            self.state_machine.current_state = SystemState.ADVANCE_Z
                            self.state_machine.frame_counter = 0
                            self.current_rotation_rx = 0.0
                            self.current_rotation_ry = 0.0
                            self.current_translation_dx = 0.0
                            self.current_translation_dy = 0.0

                # ── 7. State-change: reset accumulators ──
                # (Handled by the transition completion blocks above,
                # plus per-cycle tracking in move-complete callbacks.)

                # ── 8. Build MotionCommand ──
                if self.auto_run and action_type is not None:
                    motion_cmd = MotionCmdMsg()
                    motion_cmd.header.stamp = self.get_clock().now().to_msg()
                    motion_cmd.dx_mm = float(dx_cmd)
                    motion_cmd.dy_mm = float(dy_cmd)
                    motion_cmd.dz_mm = float(dz_cmd)
                    motion_cmd.rx_deg = float(rx_cmd)
                    motion_cmd.ry_deg = float(ry_cmd)
                    motion_cmd.rz_deg = 0.0
                    motion_cmd.source = 'inside'

                    # ── 9. Publish + dispatch (guarded by motion_in_progress) ──
                    now = time.time()
                    with self.moving_lock:
                        # Timeout protection: reset if stuck > 5s
                        if self.motion_in_progress and (now - self._last_motion_sent_time > 5.0):
                            self.get_logger().error('Motion timeout — resetting')
                            self.motion_in_progress = False

                        if not self.motion_in_progress:
                            self.motion_in_progress = True
                            self._last_motion_sent_time = now
                            # Publish to ROS channel — motion_executor picks it up
                            self._motion_cmd_pub.publish(motion_cmd)
                            self.get_logger().debug(
                                f'Motion cmd: dx={dx_cmd:.2f} dy={dy_cmd:.2f} dz={dz_cmd:.2f} '
                                f'rx={rx_cmd:.2f} ry={ry_cmd:.2f}'
                            )

                # ── 10. Prediction marker ──
                if self.goal_is_predicted:
                    status_msg += " [PRED]"

                # ── 10. Visualization ──
                if vis_frame is not None:
                    vis_frame = self.vision._visualize_result(
                        vis_frame, filtered_force,
                        raw_goal,
                        self.vision._find_regions(self.vision._preprocess_image(vis_frame))[0],
                        self.vision._find_regions(self.vision._preprocess_image(vis_frame))[1],
                        filtered_goal=goal,
                    )
                    self.draw_ui(vis_frame, force_mag, pixel_dist, status_msg)

                    with self.data_lock:
                        self.display_frame = vis_frame.copy()

            except Exception as e:
                self.get_logger().error(f"处理线程 'processing_loop' 发生致命错误: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                time.sleep(1)

        self.get_logger().info("处理线程已停止。")

    # ═══════════════════════════════════════════════════════════
    #  Motion result callback (ROS-path mode)
    # ═══════════════════════════════════════════════════════════

    def _motion_result_cb(self, msg: MotionResult):
        """Receive motion execution result from motion_executor."""
        with self.moving_lock:
            if msg.success:
                dz = msg.executed_dz_mm
                if dz > 0:
                    self.current_depth += dz
                    if self.current_state == SystemState.BLIND_ENTRY:
                        self.blind_entry_distance += dz
                elif dz < 0:
                    self.current_depth = max(0.0, self.current_depth + dz)
                    self.blocked_start_time = None
                    self.state_machine.clear_blocked()
                self.current_rotation_rx += abs(msg.executed_rx_deg)
                self.current_rotation_ry += abs(msg.executed_ry_deg)
                self.total_rx_deg += abs(msg.executed_rx_deg)
                self.total_ry_deg += abs(msg.executed_ry_deg)
                self.current_translation_dx += msg.executed_dx_mm
                self.current_translation_dy += msg.executed_dy_mm
                self.total_translation_dx += msg.executed_dx_mm
                self.total_translation_dy += msg.executed_dy_mm
            else:
                self.get_logger().warn(f'Motion failed: {msg.error_msg}')
            self.motion_in_progress = False
            self.last_move_time = time.time()

    # ═══════════════════════════════════════════════════════════
    #  UI
    # ═══════════════════════════════════════════════════════════

    def process_frame(self):
        if self.cap is None or not self.cap.isOpened() or self.system_terminated:
            return

        # ── FPS counter ──
        self._fps_frame_count += 1
        now_fps = time.time()
        if self._fps_last_print == 0.0:
            self._fps_last_print = now_fps
        elif now_fps - self._fps_last_print >= 1.0:
            fps = self._fps_frame_count / (now_fps - self._fps_last_print)
            self.get_logger().info(f"FPS: {fps:.1f}")
            self._fps_frame_count = 0
            self._fps_last_print = now_fps

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("无法读取视频帧")
            return

        with self.data_lock:
            self.latest_frame = frame
            frame_to_show = self.display_frame

        # ── Publish camera frame to ROS topic ──
        try:
            ros_image = self._cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'endoscope_camera'
            self._image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"图像发布失败: {e}", throttle_duration_sec=5.0)

        if frame_to_show is None:
            frame_to_show = frame

        cv2.imshow("Inside Navigation", frame_to_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.system_terminated = True
            self.processing_thread_running = False
            self.get_logger().info("用户请求退出。")
            return
        elif key == ord(' '):
            self.auto_run = not self.auto_run
            if self.auto_run:
                self.current_depth = 0.0
                self.blind_entry_distance = 0.0
                self.blind_last_step_time = 0.0
                self.total_rx_deg = 0.0
                self.total_ry_deg = 0.0
                self.total_translation_dx = 0.0
                self.total_translation_dy = 0.0
                self.current_rotation_rx = 0.0
                self.current_rotation_ry = 0.0
                self.current_translation_dx = 0.0
                self.current_translation_dy = 0.0
                self.blocked_start_time = None
                self.goal_is_predicted = False
                self.consecutive_predicted_frames = 0
                self.goal_kalman_filter.reset()
                self.force_x_filter.reset()
                self.force_y_filter.reset()
                self.dist_filter.reset()
                self.state_machine.reset()
                self.get_logger().info(">>> 自动导航已启动 <<<")
            else:
                self.get_logger().info(">>> 自动导航已暂停 <<<")
        elif key == ord('r') or key == ord('R'):
            if self.current_state == SystemState.MAX_DEPTH_REACHED:
                self.current_depth = 0.0
                self.state_machine.current_state = SystemState.IDLE
                self.state_machine.frame_counter = 0
                self.get_logger().info(">>> 深度上限已手动重置，系统回到 IDLE <<<")
            else:
                self.current_depth = 0.0
                self.get_logger().info(f">>> 深度计数已重置为 0 <<<")

    def draw_ui(self, frame, force_mag, pixel_dist, status_msg):
        h_img, w_img = frame.shape[:2]
        state = self.current_state

        cv2.putText(frame, f"State: {state.name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Depth: {self.current_depth:.1f}/{self.MAX_DEPTH_MM}mm",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Dist: {pixel_dist:.1f}px  F: {force_mag:.1f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, status_msg, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(frame,
                    f"Rot: Rx={self.current_rotation_rx:.3f}° Ry={self.current_rotation_ry:.3f}° "
                    f"(max={self.MAX_TOTAL_ROTATION_DEG}°)",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(frame,
                    f"Trans: dx={self.current_translation_dx:.3f}mm dy={self.current_translation_dy:.3f}mm",
                    (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(frame,
                    f"Total: Rx={self.total_rx_deg:.2f}° Ry={self.total_ry_deg:.2f}° "
                    f"dx={self.total_translation_dx:.2f}mm dy={self.total_translation_dy:.2f}mm",
                    (10, 171), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)
        run_color = (0, 255, 0) if self.auto_run else (0, 0, 255)
        run_text = "AUTO" if self.auto_run else "PAUSED"
        cv2.putText(frame, run_text, (w_img - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, run_color, 2)

        # Depth progress bar
        bar_x, bar_y, bar_w, bar_h = 10, h_img - 20, w_img - 20, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * min(self.current_depth / max(self.MAX_DEPTH_MM, 1e-6), 1.0))
        bar_color = (0, 200, 255) if self.current_depth < self.MAX_DEPTH_MM * 0.8 else (0, 80, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

        if state == SystemState.MAX_DEPTH_REACHED:
            cv2.putText(frame, "MAX DEPTH REACHED  Press R to reset",
                        (w_img // 2 - 200, h_img // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ═══════════════════════════════════════════════════════════
    #  Cleanup
    # ═══════════════════════════════════════════════════════════

    def destroy_node(self):
        self.processing_thread_running = False
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        if hasattr(self, 'robot'):
            self.robot.disconnect()

        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

        if hasattr(self, 'ros_executor'):
            self.ros_executor.shutdown()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MoveInsideNode()

    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(node)
    node.ros_executor = ros_executor

    spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()

    node.processing_thread_running = True
    node.processing_thread = threading.Thread(target=node._processing_loop, daemon=True)
    node.processing_thread.start()

    try:
        while rclpy.ok() and not node.system_terminated:
            node.process_frame()
    except KeyboardInterrupt:
        pass
    finally:
        node.processing_thread_running = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread.is_alive():
            spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
