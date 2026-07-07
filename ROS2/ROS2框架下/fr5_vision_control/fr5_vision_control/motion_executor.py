"""Motion executor — sole owner of RobotController.

Subscribes to ``/robot/motion_command`` (post-arbiter).  Executes each
command via ``RobotController`` in a background thread pool, then publishes
the result to ``/robot/motion_result``.

In ``direct_motion=false`` mode this is the *only* node that talks to the
robot — navigation nodes publish commands and subscribe to results.
"""

import math
import time
import threading
import concurrent.futures
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nasal_endoscopy_msgs.msg import MotionCommand, MotionResult
from diagnostic_msgs.msg import DiagnosticStatus
from fr5_vision_control.robot_controller import RobotController


class MotionExecutor(Node):
    """Hold the sole RobotController.  Execute /robot/motion_command."""

    def __init__(self):
        super().__init__('motion_executor')

        self.declare_parameter('simulate', True)
        self.declare_parameter('max_velocity_mm_s', 30.0)
        self.simulate = self.get_parameter('simulate').value

        # ── Robot Controller (sole instance) ────────────────
        self.robot = RobotController(self, simulate=self.simulate)

        # ── Subscriptions ───────────────────────────────────
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.cmd_sub = self.create_subscription(
            MotionCommand, '/robot/motion_command', self._cmd_cb, cmd_qos,
        )

        # ── Result publisher ────────────────────────────────
        self.result_pub = self.create_publisher(
            MotionResult, '/robot/motion_result', 10,
        )

        # ── Diagnostics ─────────────────────────────────────
        self._diag_pub = self.create_publisher(DiagnosticStatus, '/diagnostics/update', 10)

        # ── Async executor ──────────────────────────────────
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._exec_lock = threading.Lock()
        self._motion_in_progress = False
        self._executed_count = 0
        self._fail_count = 0

        self._publish_diag(DiagnosticStatus.OK, 'MotionExecutor ready')
        self.get_logger().info(
            f'[MotionExec] Ready — simulate={self.simulate}, '
            f'listening on /robot/motion_command'
        )

    # ── diagnostics ─────────────────────────────────────────

    def _publish_diag(self, level: int, message: str):
        st = DiagnosticStatus()
        st.name = 'motion_executor'
        st.level = level
        st.message = message
        st.hardware_id = 'motion_exec'
        self._diag_pub.publish(st)

    # ── command callback ────────────────────────────────────

    def _cmd_cb(self, msg: MotionCommand):
        # Measure latency from navigation publish to receipt here
        now = self.get_clock().now()
        rtt_ms = 0.0
        if msg.header.stamp.sec > 0:
            sent = rclpy.time.Time.from_msg(msg.header.stamp)
            rtt_ms = (now - sent).nanoseconds / 1e6

        with self._exec_lock:
            if self._motion_in_progress:
                self.get_logger().warn(
                    '[MotionExec] Command dropped — previous motion still in progress',
                    throttle_duration_sec=2.0,
                )
                return
            self._motion_in_progress = True

        self.get_logger().info(
            f'[MotionExec] ← cmd: dx={msg.dx_mm:.2f} dy={msg.dy_mm:.2f} '
            f'dz={msg.dz_mm:.2f} rx={msg.rx_deg:.2f} ry={msg.ry_deg:.2f} '
            f'RTT={rtt_ms:.1f}ms'
        )
        self._executor.submit(self._execute, msg, rtt_ms)

    def _execute(self, cmd: MotionCommand, rtt_ms: float):
        """Run in thread pool — call RobotController and publish result."""
        try:
            # Determine motion type and call appropriate method
            has_rotation = abs(cmd.rx_deg) > 1e-6 or abs(cmd.ry_deg) > 1e-6 or abs(cmd.rz_deg) > 1e-6
            has_translation = abs(cmd.dx_mm) > 1e-6 or abs(cmd.dy_mm) > 1e-6
            has_z = abs(cmd.dz_mm) > 1e-6

            if has_rotation:
                # move_rotate_and_translate handles all DOFs in one call
                success = self.robot.move_rotate_and_translate(
                    cmd.rx_deg, cmd.ry_deg, cmd.dx_mm, cmd.dy_mm, cmd.dz_mm, cmd.rz_deg,
                )
            elif has_z and not has_translation:
                # Pure Z move
                success = self.robot.move_z_only(cmd.dz_mm)
            else:
                # Pure XY or offset move (may include Z)
                success = self.robot.move_offset_tool_frame(cmd.dx_mm, cmd.dy_mm, cmd.dz_mm)

            # Publish result
            result = MotionResult()
            result.header.stamp = self.get_clock().now().to_msg()
            result.success = success
            result.executed_dx_mm = cmd.dx_mm
            result.executed_dy_mm = cmd.dy_mm
            result.executed_dz_mm = cmd.dz_mm
            result.executed_rx_deg = cmd.rx_deg
            result.executed_ry_deg = cmd.ry_deg
            result.executed_rz_deg = cmd.rz_deg
            result.error_msg = '' if success else 'RobotController returned failure'
            self.result_pub.publish(result)

            with self._exec_lock:
                self._executed_count += 1
                if not success:
                    self._fail_count += 1

            self.get_logger().info(
                f'[MotionExec] → done: success={success} '
                f'(total ok={self._executed_count - self._fail_count} fail={self._fail_count})'
            )

        except Exception as e:
            self.get_logger().error(f'[MotionExec] Execution error: {e}')
            result = MotionResult()
            result.header.stamp = self.get_clock().now().to_msg()
            result.success = False
            result.executed_dx_mm = cmd.dx_mm
            result.executed_dy_mm = cmd.dy_mm
            result.executed_dz_mm = cmd.dz_mm
            result.error_msg = str(e)[:200]
            self.result_pub.publish(result)
            self._publish_diag(DiagnosticStatus.ERROR, f'Execution error: {e}')

            with self._exec_lock:
                self._fail_count += 1

        finally:
            with self._exec_lock:
                self._motion_in_progress = False

    # ── cleanup ─────────────────────────────────────────────

    def destroy_node(self):
        if hasattr(self, 'robot'):
            self.robot.disconnect()
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotionExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
