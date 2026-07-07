"""Outside-navigation ROS 2 node (体外导航).

Refactored: algorithm logic lives in ``nasal_endoscopy_algorithms``;
this node *only* handles ROS communication, camera I/O, and motion dispatch.
``RobotController`` is called exactly as before — its four-step sequence
is preserved verbatim.
"""

import sys
import rclpy
import threading
import time
import math
import cv2
import numpy as np
from rclpy.node import Node, SetParametersResult
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

# ── ROS image transport ──
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ── Custom messages ──
from nasal_endoscopy_msgs.msg import NostrilDetection, NavigationState, MotionCommand as MotionCmdMsg, MotionResult

# ── Diagnostics ──
from diagnostic_msgs.msg import DiagnosticStatus

# ── Action Server ──
from rclpy.action import ActionServer, CancelResponse
from nasal_endoscopy_msgs.action import NavigateOutside

# ── Algorithm package (zero-ROS) ──
from nasal_endoscopy_algorithms.vision.outside_vision import (
    VisionSystem, NostrilDetectionResult,
)
from nasal_endoscopy_algorithms.filters.ema_filter import EMAFilter
from nasal_endoscopy_algorithms.utils.geometry import clamp
from nasal_endoscopy_algorithms.utils.state_enums import SystemState
from nasal_endoscopy_algorithms.state_machine.outside_sm import (
    OutsideStateMachine, OutsideSMConfig,
)

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
                # Flask multipart/x-mixed-replace — parse boundary
                self._boundary = None
                for part in content_type.split(';'):
                    part = part.strip()
                    if part.startswith('boundary='):
                        self._boundary = part.split('=', 1)[1].encode()
                        break
                if self._boundary is None:
                    self._boundary = b'frame'  # Flask default
                self._opened = True
            else:
                # Raw binary stream (single JPEG or other)
                self._boundary = None
                self._opened = True
        except Exception as e:
            if self._log:
                self._log.error(f"MJPEG 流打开失败: {e}")

    def isOpened(self):
        return self._opened

    def read(self):
        """Return (success: bool, frame: np.ndarray | None)."""
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
        # Read until we have a complete JPEG between boundaries
        max_attempts = 20
        for _ in range(max_attempts):
            chunk = self._stream.read(4096)
            if not chunk:
                return False, None
            self._buffer += chunk

            # Look for boundary-bounded JPEG
            while True:
                # Find start boundary
                b_start = self._buffer.find(b'--' + self._boundary + b'\r\n')
                if b_start == -1:
                    b_start = self._buffer.find(b'--' + self._boundary + b'\n')
                if b_start == -1:
                    break  # need more data

                # Find content-length or next boundary
                after_start = self._buffer[b_start:]
                header_end = after_start.find(b'\r\n\r\n')
                if header_end == -1:
                    break

                body_start = b_start + header_end + 4
                b_next = self._buffer.find(b'--' + self._boundary, body_start)
                if b_next == -1:
                    break  # need more data

                # Extract JPEG body
                jpeg_data = self._buffer[body_start:b_next]
                # Trim trailing \r\n before next boundary
                while jpeg_data.endswith(b'\r') or jpeg_data.endswith(b'\n'):
                    jpeg_data = jpeg_data[:-1]
                self._buffer = self._buffer[b_next:]

                if len(jpeg_data) < 100:
                    continue  # too small, skip

                frame = cv2.imdecode(
                    np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
            # loop: read more data
        return False, None  # exhausted attempts

    def _read_raw(self):
        import numpy as np
        # Raw JPEG stream — read until we have a complete JPEG (ends with FF D9)
        max_attempts = 20
        for _ in range(max_attempts):
            chunk = self._stream.read(4096)
            if not chunk:
                return False, None
            self._buffer += chunk
            # Find JPEG end marker
            soi = self._buffer.find(b'\xff\xd8')
            eoi = self._buffer.find(b'\xff\xd9', soi + 2)
            if soi >= 0 and eoi > soi:
                jpeg_data = self._buffer[soi:eoi + 2]
                self._buffer = self._buffer[eoi + 2:]
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
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


class MoveOutsideNode(Node):
    """ROS 2 node for outside (nostril-approach) navigation."""

    def __init__(self):
        super().__init__('move_outside_node')

        # ── 1. Parameters (all read from YAML / launch / CLI) ──
        self.declare_parameter('video_idx', '0')
        self.declare_parameter('robot_ip', '192.168.58.2')
        self.declare_parameter('simulate', True)

        self.declare_parameter('align_tolerance_enter', 20.0)
        self.declare_parameter('align_tolerance_exit', 10.0)
        self.declare_parameter('min_align_frames', 3)
        self.declare_parameter('min_approach_frames', 3)
        self.declare_parameter('transition_frames', 5)

        self.declare_parameter('xy_max_step_mm', 2.0)
        self.declare_parameter('z_max_step_mm', 2.0)
        self.declare_parameter('z_approach_step', 2.0)
        self.declare_parameter('xy_damping', 0.6)
        self.declare_parameter('max_z_total_mm', 150.0)
        self.declare_parameter('retreat_step_mm', 20.0)

        self.declare_parameter('max_jump_px', 90)
        self.declare_parameter('max_width_jump_ratio', 0.40)

        self.declare_parameter('lost_timeout_sec', 3.0)
        self.declare_parameter('min_move_interval', 0.1)

        self.declare_parameter('target_selection', 'left')
        self.declare_parameter('nostril_distance_mm', 12.0)
        self.declare_parameter('target_width_threshold', 300)

        self.declare_parameter('pos_ema_alpha', 0.35)
        self.declare_parameter('width_ema_alpha', 0.25)
        self.declare_parameter('nostril_ema_alpha', 0.25)

        self.declare_parameter('max_consecutive_fails', 3)
        self.declare_parameter('min_effective_move_mm', 0.1)

        # ── Resolve parameters ────────────────────────────────
        raw_idx = self.get_parameter('video_idx').get_parameter_value().string_value
        self.video_idx = int(raw_idx) if raw_idx.isdigit() else raw_idx
        self.simulate = self.get_parameter('simulate').get_parameter_value().bool_value
        self.get_logger().info(
            f'启动参数 -> video_idx: {self.video_idx}, simulate: {self.simulate}'
        )

        # ── 2. Async infrastructure ─────────────────────────
        self.moving_lock = threading.Lock()
        self.motion_in_progress = False

        # ── 3. Tuning constants (from ROS params) ───────────
        self.NOSTRIL_DISTANCE_MM = self.get_parameter('nostril_distance_mm').get_parameter_value().double_value
        self.TARGET_WIDTH_THRESHOLD = self.get_parameter('target_width_threshold').get_parameter_value().integer_value

        self.Z_APPROACH_STEP = self.get_parameter('z_approach_step').get_parameter_value().double_value
        self.XY_MAX_STEP_MM = self.get_parameter('xy_max_step_mm').get_parameter_value().double_value
        self.Z_MAX_STEP_MM = self.get_parameter('z_max_step_mm').get_parameter_value().double_value

        self.MAX_Z_TOTAL_MM = self.get_parameter('max_z_total_mm').get_parameter_value().double_value
        self.RETREAT_STEP_MM = self.get_parameter('retreat_step_mm').get_parameter_value().double_value
        self.MAX_JUMP_PX = self.get_parameter('max_jump_px').get_parameter_value().integer_value
        self.MAX_WIDTH_JUMP_RATIO = self.get_parameter('max_width_jump_ratio').get_parameter_value().double_value

        self.LOST_TIMEOUT_SEC = self.get_parameter('lost_timeout_sec').get_parameter_value().double_value
        self.TARGET_SELECTION = self.get_parameter('target_selection').get_parameter_value().string_value

        self.XY_DAMPING = self.get_parameter('xy_damping').get_parameter_value().double_value
        self.MIN_MOVE_INTERVAL = self.get_parameter('min_move_interval').get_parameter_value().double_value
        self.MIN_EFFECTIVE_MOVE_MM = self.get_parameter('min_effective_move_mm').get_parameter_value().double_value
        self.consecutive_move_fails = 0
        self.MAX_CONSECUTIVE_FAILS = self.get_parameter('max_consecutive_fails').get_parameter_value().integer_value

        # ── 4. Algorithm subsystems ─────────────────────────
        self.vision = VisionSystem()

        # ── Subscribe to motion results (ROS-path mode) ─────
        self._motion_result_sub = self.create_subscription(
            MotionResult, '/robot/motion_result', self._motion_result_cb, 10,
        )
        self._last_motion_sent_time = 0.0

        # Build state-machine config from ROS params
        sm_cfg = OutsideSMConfig(
            align_tolerance_enter=self.get_parameter('align_tolerance_enter').value,
            align_tolerance_exit=self.get_parameter('align_tolerance_exit').value,
            min_align_frames=self.get_parameter('min_align_frames').value,
            min_approach_frames=self.get_parameter('min_approach_frames').value,
            transition_frames=self.get_parameter('transition_frames').value,
        )
        self.state_machine = OutsideStateMachine(sm_cfg)

        # ── 5. Camera (with auto-detect Windows Flask server) ──
        self._mjpeg_reader = None  # set when URL source is used
        self.cap = None

        def _try_open(source):
            """Try to open source (int index, str URL, or str device)."""
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

        cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)

        # ── ROS 2 image transport ──
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._image_pub = self.create_publisher(
            Image, '/camera/outside/image_raw', qos_profile=image_qos,
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
        self._detection_pub = self.create_publisher(
            NostrilDetection, '/vision/outside/detection', qos_profile=vision_qos,
        )
        self._nav_state_pub = self.create_publisher(
            NavigationState, '/navigation/outside/state', qos_profile=state_qos,
        )
        self._motion_cmd_pub = self.create_publisher(
            MotionCmdMsg, '/control/outside/command', qos_profile=control_qos,
        )
        self._diag_pub = self.create_publisher(
            DiagnosticStatus, '/diagnostics/update', 10,
        )

        # ── 6. State variables ──────────────────────────────
        self.current_state = SystemState.IDLE  # safe default before processing starts
        self.auto_run = False
        self.finished = False
        self.system_terminated = False

        self.data_lock = threading.Lock()
        self.latest_frame = None
        self.display_frame = None
        self.processing_thread_running = False

        # ── FPS counter ──
        self._fps_frame_count = 0
        self._fps_last_print = 0.0

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

        self._action_cb_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            NavigateOutside,
            '/navigate_outside',
            execute_callback=self._execute_navigate_cb,
            cancel_callback=self._cancel_navigate_cb,
            callback_group=self._action_cb_group,
        )

        self.get_logger().info(">>> 系统初始化完成 <<<")
        self.get_logger().info("请在弹出的图像窗口中按 [空格] 启动自动运动，按 [Q] 退出程序")
        self._publish_diag(DiagnosticStatus.OK, 'Initialised')

        # ── Parameter hot-reload ──
        self.add_on_set_parameters_callback(self._on_param_change)

    # ═══════════════════════════════════════════════════════════
    #  Diagnostics helper
    # ═══════════════════════════════════════════════════════════

    def _publish_diag(self, level: int, message: str):
        st = DiagnosticStatus()
        st.name = 'move_outside_node'
        st.level = level
        st.message = message
        st.hardware_id = 'outside_nav'
        self._diag_pub.publish(st)

    # ── Parameter hot-reload ─────────────────────────────────

    def _on_param_change(self, params):
        from rclpy.parameter import Parameter
        for p in params:
            try:
                if p.name == 'align_tolerance_enter':
                    self.state_machine.cfg.align_tolerance_enter = p.value
                elif p.name == 'align_tolerance_exit':
                    self.state_machine.cfg.align_tolerance_exit = p.value
                elif p.name == 'min_align_frames':
                    self.state_machine.cfg.min_align_frames = p.value
                elif p.name == 'min_approach_frames':
                    self.state_machine.cfg.min_approach_frames = p.value
                elif p.name == 'transition_frames':
                    self.state_machine.cfg.transition_frames = p.value
                elif p.name == 'xy_max_step_mm':
                    self.XY_MAX_STEP_MM = p.value
                elif p.name == 'z_max_step_mm':
                    self.Z_MAX_STEP_MM = p.value
                elif p.name == 'z_approach_step':
                    self.Z_APPROACH_STEP = p.value
                elif p.name == 'xy_damping':
                    self.XY_DAMPING = p.value
                elif p.name == 'max_z_total_mm':
                    self.MAX_Z_TOTAL_MM = p.value
                elif p.name == 'retreat_step_mm':
                    self.RETREAT_STEP_MM = p.value
                elif p.name == 'max_jump_px':
                    self.MAX_JUMP_PX = p.value
                elif p.name == 'max_width_jump_ratio':
                    self.MAX_WIDTH_JUMP_RATIO = p.value
                elif p.name == 'lost_timeout_sec':
                    self.LOST_TIMEOUT_SEC = p.value
                elif p.name == 'min_move_interval':
                    self.MIN_MOVE_INTERVAL = p.value
                elif p.name == 'target_selection':
                    self.TARGET_SELECTION = p.value
                elif p.name == 'nostril_distance_mm':
                    self.NOSTRIL_DISTANCE_MM = p.value
                elif p.name == 'target_width_threshold':
                    self.TARGET_WIDTH_THRESHOLD = p.value
                elif p.name == 'pos_ema_alpha':
                    self.pos_filter.alpha = p.value
                elif p.name == 'width_ema_alpha':
                    self.w_filter.alpha = p.value
                elif p.name == 'nostril_ema_alpha':
                    self.nostril_filter.alpha = p.value
                elif p.name == 'max_consecutive_fails':
                    self.MAX_CONSECUTIVE_FAILS = p.value
                elif p.name == 'min_effective_move_mm':
                    self.MIN_EFFECTIVE_MOVE_MM = p.value
                # video_idx, robot_ip, simulate — not hot-reloadable
            except Exception as e:
                self.get_logger().warn(f'Failed to apply param {p.name}={p.value}: {e}')
        return SetParametersResult(successful=True)

    # ═══════════════════════════════════════════════════════════
    #  Action Server callbacks
    # ═══════════════════════════════════════════════════════════

    def _cancel_navigate_cb(self, goal_handle):
        """Accept cancellation of an in-progress navigation goal."""
        self.get_logger().info('[Action] NavigateOutside cancellation requested')
        self.auto_run = False
        return CancelResponse.ACCEPT

    def _execute_navigate_cb(self, goal_handle):
        """Execute a NavigateOutside action goal."""
        target = goal_handle.request.target_side
        max_depth = goal_handle.request.max_depth_mm
        self.get_logger().info(
            f'[Action] NavigateOutside goal accepted: target={target}, max_depth={max_depth}mm'
        )

        # Apply goal parameters
        self.TARGET_SELECTION = target if target else self.TARGET_SELECTION
        if max_depth > 0.0:
            self.MAX_Z_TOTAL_MM = max_depth

        # Start navigation (shares auto_run with keyboard)
        self.auto_run = True
        self.z_total_moved = 0.0
        self.finished = False
        self.lost_start_time = None
        self.retreat_attempted = False
        self.state_machine.reset()

        feedback = NavigateOutside.Feedback()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.auto_run = False
                self.get_logger().info('[Action] NavigateOutside cancelled')
                result = NavigateOutside.Result()
                result.reached = False
                result.final_depth_mm = float(self.z_total_moved)
                result.final_state = self.current_state.name
                return result

            # Publish feedback at ~10 Hz
            feedback.current_state = self.current_state.name
            feedback.current_depth_mm = float(self.z_total_moved)
            feedback.target_dist_px = float(getattr(self, '_last_dist_err', 0.0))
            feedback.feature_width_px = float(getattr(self, '_last_filtered_w', 0.0))
            goal_handle.publish_feedback(feedback)

            if self.finished:
                break

            if not self.auto_run:  # keyboard pause
                self.get_logger().info('[Action] Navigation paused by keyboard')
                break

            time.sleep(0.1)

        result = NavigateOutside.Result()
        result.reached = self.finished
        result.final_depth_mm = float(self.z_total_moved)
        result.final_state = self.current_state.name

        if self.finished:
            goal_handle.succeed()
            self.get_logger().info(f'[Action] NavigateOutside succeeded at depth={self.z_total_moved:.1f}mm')
        else:
            goal_handle.abort()
            self.get_logger().info('[Action] NavigateOutside aborted')

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
    #  Motion result callback (ROS-path mode)
    # ═══════════════════════════════════════════════════════════

    def _motion_result_cb(self, msg: MotionResult):
        """Receive motion execution result from motion_executor."""
        with self.moving_lock:
            if msg.success:
                if msg.executed_dz_mm > 0:
                    self.z_total_moved += msg.executed_dz_mm
                elif msg.executed_dz_mm < 0:
                    self.z_total_moved += msg.executed_dz_mm
                    self.z_total_moved = max(0.0, self.z_total_moved)
                self.consecutive_move_fails = 0
            else:
                self.consecutive_move_fails += 1
                self.get_logger().error(
                    f'Motion failed: {msg.error_msg} '
                    f'(fails={self.consecutive_move_fails}/{self.MAX_CONSECUTIVE_FAILS})'
                )
                if self.consecutive_move_fails >= self.MAX_CONSECUTIVE_FAILS:
                    self.get_logger().error('连续失败次数上限，自动停止')
                    self._publish_diag(DiagnosticStatus.WARN,
                                       f'Max consecutive fails ({self.MAX_CONSECUTIVE_FAILS})')
                    self.auto_run = False
                    self.consecutive_move_fails = 0
            self.motion_in_progress = False
            self.last_move_time = time.time()

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

                eff_frame, _ = self.vision.crop_effective_area(frame_to_process)
                if eff_frame is None:
                    eff_frame = frame_to_process.copy()
                h, w = eff_frame.shape[:2]
                cam_center_x, cam_center_y = w // 2, h // 2

                # ── 1. Detection ──
                detection: NostrilDetectionResult = self.vision.detect_nose_target(
                    eff_frame, target_side=self.TARGET_SELECTION,
                )

                # Publish detection result
                det_msg = NostrilDetection()
                det_msg.header.stamp = self.get_clock().now().to_msg()
                det_msg.header.frame_id = 'endoscope_camera'
                has_target = detection.nose_pos is not None
                det_msg.nose_x = detection.nose_pos[0] if has_target else -1
                det_msg.nose_y = detection.nose_pos[1] if has_target else -1
                det_msg.feature_width = detection.feature_width
                det_msg.nostril_distance_px = detection.nostril_distance_px
                det_msg.valid = has_target
                self._detection_pub.publish(det_msg)

                # ── 2. Outlier filtering ──
                valid_obs = False
                nose_pos = detection.nose_pos
                nose_width = detection.feature_width
                nostril_px = detection.nostril_distance_px

                if nose_pos is not None and nose_width > 1.0 and nostril_px > 1.0:
                    jump = 0.0
                    w_jump_ratio = 0.0
                    if self.last_raw_pos is not None:
                        jump = math.hypot(nose_pos[0] - self.last_raw_pos[0],
                                          nose_pos[1] - self.last_raw_pos[1])
                    if self.last_raw_w is not None and self.last_raw_w > 1.0:
                        w_jump_ratio = abs(nose_width - self.last_raw_w) / self.last_raw_w
                    if jump <= self.MAX_JUMP_PX and w_jump_ratio <= self.MAX_WIDTH_JUMP_RATIO:
                        valid_obs = True

                filtered_pos, filtered_w, filtered_nostril, dynamic_pixel_to_mm = \
                    None, None, None, 0.05

                if valid_obs:
                    self.last_raw_pos, self.last_raw_w = nose_pos, nose_width
                    filtered_pos = self.pos_filter.update([nose_pos[0], nose_pos[1]])
                    filtered_w = float(self.w_filter.update([nose_width])[0])
                    filtered_nostril = float(self.nostril_filter.update([nostril_px])[0])
                    if filtered_nostril > 5.0:
                        dynamic_pixel_to_mm = self.NOSTRIL_DISTANCE_MM / filtered_nostril
                    self.lost_start_time, self.filter_reset_done, self.had_target_before = \
                        None, False, True
                else:
                    nose_pos, nose_width, nostril_px = None, 0.0, 0.0

                err_x = int(filtered_pos[0]) - cam_center_x if filtered_pos is not None else 0
                err_y = int(filtered_pos[1]) - cam_center_y if filtered_pos is not None else 0
                dist_err = math.hypot(err_x, err_y) if filtered_pos is not None else 0.0
                self._last_dist_err = dist_err
                self._last_filtered_w = filtered_w if filtered_w is not None else 0.0

                # ── 3. State machine ──
                self.current_state = self.state_machine.evaluate(
                    has_target=(filtered_pos is not None),
                    dist_err_px=dist_err,
                    filtered_w=filtered_w if filtered_w is not None else 0.0,
                    is_auto_run=self.auto_run,
                    is_finished=self.finished,
                    target_width_threshold=self.TARGET_WIDTH_THRESHOLD,
                )

                # Publish navigation state
                nav_msg = NavigationState()
                nav_msg.header.stamp = self.get_clock().now().to_msg()
                nav_msg.state = self.current_state.name
                nav_msg.current_depth_mm = float(self.z_total_moved)
                nav_msg.max_depth_mm = self.MAX_Z_TOTAL_MM
                nav_msg.target_dist_px = float(dist_err)
                nav_msg.force_magnitude = 0.0  # outside nav doesn't use APF
                nav_msg.auto_run = self.auto_run
                nav_msg.finished = self.finished
                self._nav_state_pub.publish(nav_msg)

                # ── 4. Motion command computation ──
                move_x, move_y, move_z = 0.0, 0.0, 0.0
                action_type = None
                state = self.current_state
                fc = self.state_machine.frame_counter
                tf = self.state_machine.cfg.transition_frames

                if state == SystemState.TRANSITION_TO_APPROACH:
                    dist_err_now = math.hypot(err_x, err_y)
                    adaptive_damping = self.XY_DAMPING * 0.7 if dist_err_now < 50 else self.XY_DAMPING
                    target_move_x = err_x * dynamic_pixel_to_mm * adaptive_damping
                    target_move_y = err_y * dynamic_pixel_to_mm * adaptive_damping
                    target_move_x = clamp(target_move_x, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    target_move_y = clamp(target_move_y, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    ratio = min(fc / max(tf, 1), 1.0)
                    move_x = target_move_x * (1.0 - ratio)
                    move_y = target_move_y * (1.0 - ratio)
                    move_z = self.Z_APPROACH_STEP * ratio
                    action_type = 'move'

                elif state == SystemState.TRANSITION_TO_ALIGN:
                    dist_err_now = math.hypot(err_x, err_y)
                    adaptive_damping = self.XY_DAMPING * 0.7 if dist_err_now < 50 else self.XY_DAMPING
                    target_move_x = err_x * dynamic_pixel_to_mm * adaptive_damping
                    target_move_y = err_y * dynamic_pixel_to_mm * adaptive_damping
                    target_move_x = clamp(target_move_x, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    target_move_y = clamp(target_move_y, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    ratio = min(fc / max(tf, 1), 1.0)
                    move_x = target_move_x * ratio
                    move_y = target_move_y * ratio
                    move_z = 0.0
                    action_type = 'move'

                elif state == SystemState.ALIGN_XY:
                    dist_err_now = math.hypot(err_x, err_y)
                    adaptive_damping = self.XY_DAMPING * 0.7 if dist_err_now < 50 else self.XY_DAMPING
                    target_move_x = err_x * dynamic_pixel_to_mm * adaptive_damping
                    target_move_y = err_y * dynamic_pixel_to_mm * adaptive_damping
                    move_x = clamp(target_move_x, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    move_y = clamp(target_move_y, -self.XY_MAX_STEP_MM, self.XY_MAX_STEP_MM)
                    if abs(move_x) > self.MIN_EFFECTIVE_MOVE_MM or abs(move_y) > self.MIN_EFFECTIVE_MOVE_MM:
                        action_type = 'move'

                elif state == SystemState.APPROACH_Z:
                    width_diff = self.TARGET_WIDTH_THRESHOLD - (filtered_w or 0)
                    if width_diff > 60:
                        actual_step = self.Z_APPROACH_STEP
                    elif width_diff > 25:
                        actual_step = self.Z_APPROACH_STEP * 0.5
                    else:
                        actual_step = 1.0

                    if self.z_total_moved + actual_step >= self.MAX_Z_TOTAL_MM:
                        self.finished = True
                        self.get_logger().warn(f"深度安全限制触发 ({self.z_total_moved}mm), 已停止")
                    else:
                        move_z = clamp(actual_step, 0.0, self.Z_MAX_STEP_MM)
                        if abs(move_z) > self.MIN_EFFECTIVE_MOVE_MM:
                            action_type = 'move'

                elif state == SystemState.TARGET_LOST:
                    if self.auto_run and self.had_target_before:
                        if self.lost_start_time is None:
                            self.lost_start_time = time.time()
                        lost_dt = time.time() - self.lost_start_time
                        if lost_dt >= self.LOST_TIMEOUT_SEC and not self.retreat_attempted:
                            move_z = -self.RETREAT_STEP_MM
                            self.retreat_attempted = True
                            action_type = 'retreat'

                elif state == SystemState.TARGET_REACHED:
                    action_type = None

                # ── 5. Build MotionCommand ──
                if action_type is not None and self.auto_run:
                    motion_cmd = MotionCmdMsg()
                    motion_cmd.header.stamp = self.get_clock().now().to_msg()
                    motion_cmd.dx_mm = float(move_x)
                    motion_cmd.dy_mm = float(move_y)
                    motion_cmd.dz_mm = float(move_z)
                    motion_cmd.rx_deg = 0.0
                    motion_cmd.ry_deg = 0.0
                    motion_cmd.rz_deg = 0.0
                    motion_cmd.source = 'outside'

                    # ── 6. Publish + dispatch (guarded by motion_in_progress) ──
                    now = time.time()
                    if now - self.last_move_time >= self.MIN_MOVE_INTERVAL:
                        with self.moving_lock:
                            # Timeout protection: reset if stuck > 5s
                            if self.motion_in_progress and (now - self._last_motion_sent_time > 5.0):
                                self.get_logger().error('Motion timeout — resetting motion_in_progress')
                                self.motion_in_progress = False

                            if not self.motion_in_progress:
                                self.motion_in_progress = True
                                self._last_motion_sent_time = now
                                # Publish to ROS channel — motion_executor picks it up
                                self._motion_cmd_pub.publish(motion_cmd)
                                self.get_logger().debug(
                                    f'Motion cmd: dx={move_x:.2f} dy={move_y:.2f} dz={move_z:.2f}'
                                )

                # ── 7. Visualization ──
                cv2.line(eff_frame, (cam_center_x - 10, cam_center_y),
                         (cam_center_x + 10, cam_center_y), (0, 255, 0), 2)
                cv2.line(eff_frame, (cam_center_x, cam_center_y - 10),
                         (cam_center_x, cam_center_y + 10), (0, 255, 0), 2)
                if filtered_pos is not None:
                    cv2.circle(eff_frame, (int(filtered_pos[0]), int(filtered_pos[1])),
                               5, (0, 0, 255), -1)

                cv2.putText(eff_frame, f"State: {state.name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
                cv2.putText(eff_frame, f"Auto: {self.auto_run} Finished: {self.finished}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(eff_frame, f"Z Moved: {self.z_total_moved:.1f}mm / {self.MAX_Z_TOTAL_MM}mm",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
                if filtered_pos is not None:
                    cv2.putText(eff_frame, f"Err(px): X={err_x}, Y={err_y}",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.putText(eff_frame, f"Width: {int(filtered_w)}px",
                                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.putText(eff_frame, f"Pixel->MM: {dynamic_pixel_to_mm:.4f}",
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)

                with self.data_lock:
                    self.display_frame = eff_frame.copy()

            except Exception as e:
                self.get_logger().error(f"处理线程 'processing_loop' 发生致命错误: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                time.sleep(1)

        self.get_logger().info("处理线程已停止。")

    # ═══════════════════════════════════════════════════════════
    #  UI / main loop
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
                    self.state_machine.reset()
            else:
                self.get_logger().info("运动已完成或系统终止，按Q退出")

    def destroy_node(self):
        self.get_logger().info("释放资源中...")
        self.processing_thread_running = False
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MoveOutsideNode()

    if node.system_terminated:
        node.get_logger().error("节点初始化失败，即将退出。")
        node.destroy_node()
        rclpy.shutdown()
        return

    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(node)

    spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    spin_thread.start()

    node.processing_thread_running = True
    node.processing_thread = threading.Thread(target=node._processing_loop, daemon=True)
    node.processing_thread.start()

    try:
        while rclpy.ok() and not node.system_terminated:
            node.process_frame()
            time.sleep(0.01)
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
