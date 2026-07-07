"""Simulation bridge — mocks the official ``ros2_cmd_server``.

Provides the **exact same** service and topic interfaces as the real
fairino hardware driver so that ``RobotController`` can be tested
without a physical robot.

Service: ``/fairino_remote_command_service`` (RemoteCmdInterface)
Topic:   ``/nonrt_state_data`` (RobotNonrtState, 50 Hz)

The simulated robot starts at a configurable home pose and applies
Cartesian deltas with a simple motion-delay model.
"""

import math
import time
import threading
import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticStatus
from fairino_msgs.srv import RemoteCmdInterface
from fairino_msgs.msg import RobotNonrtState


# ── helpers ─────────────────────────────────────────────────

def _parse_floats(s: str, n: int):
    """Extract the first *n* float values from a command string."""
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return [float(x) for x in nums[:n]]


class RobotSimBridge(Node):
    """Pure-Python simulation node that mimics the FR5 ROS 2 driver."""

    def __init__(self):
        super().__init__('robot_sim_bridge')

        # ── 1. Parameters ───────────────────────────────────
        self.declare_parameter('home_pose', [0.3, 0.0, 0.3, 180.0, 0.0, 0.0])
        self.declare_parameter('state_hz', 50.0)
        self.declare_parameter('motion_speed_mm_s', 50.0)
        self.declare_parameter('motion_speed_deg_s', 30.0)

        home = self.get_parameter('home_pose').value
        self.pose = list(map(float, home))  # [x, y, z, rx, ry, rz]  mm / deg
        self.state_hz = self.get_parameter('state_hz').value
        self.speed_mm = self.get_parameter('motion_speed_mm_s').value
        self.speed_deg = self.get_parameter('motion_speed_deg_s').value

        # Offset-enabled state
        self._offset_enabled = False
        self._offset = [0.0] * 6  # dx, dy, dz, drx, dry, drz

        # ── 2. Service ──────────────────────────────────────
        self.srv = self.create_service(
            RemoteCmdInterface,
            '/fairino_remote_command_service',
            self._handle_command,
        )
        self.get_logger().info('[SimBridge] /fairino_remote_command_service ready')

        # ── 3. State publisher ──────────────────────────────
        self.state_pub = self.create_publisher(
            RobotNonrtState, '/nonrt_state_data', 10,
        )
        self._pub_timer = self.create_timer(1.0 / self.state_hz, self._publish_state)

        # ── 4. Diagnostics publisher ────────────────────────
        self._diag_pub = self.create_publisher(DiagnosticStatus, '/diagnostics/update', 10)
        self._publish_diag(DiagnosticStatus.OK, 'SimBridge initialised')

        self.get_logger().info(
            f'[SimBridge] Initialised — home={self.pose[:3]} mm, '
            f'publishing @ {self.state_hz} Hz'
        )

    def _publish_diag(self, level: int, message: str):
        st = DiagnosticStatus()
        st.name = 'robot_sim_bridge'
        st.level = level
        st.message = message
        st.hardware_id = 'sim_bridge'
        self._diag_pub.publish(st)

    # ── service handler ─────────────────────────────────────

    def _handle_command(self, request, response):
        cmd: str = request.cmd_str
        self.get_logger().debug(f'[SimBridge] ← {cmd}')

        try:
            if cmd.startswith('CARTPoint'):
                nums = _parse_floats(cmd, 7)
                if len(nums) >= 7:
                    self.pose = nums[1:7]
                response.cmd_res = "0"

            elif cmd.startswith('PointsOffsetEnable'):
                nums = _parse_floats(cmd, 7)
                if len(nums) >= 7:
                    self._offset_enabled = True
                    self._offset = nums[1:7]
                response.cmd_res = "0"

            elif cmd.startswith('PointsOffsetDisable'):
                self._offset_enabled = False
                self._offset = [0.0] * 6
                response.cmd_res = "0"

            elif cmd.startswith('MoveL'):
                response.cmd_res = self._simulate_movel(cmd)

            elif cmd.startswith('ResetAllError'):
                self.get_logger().info('[SimBridge] ResetAllError')
                response.cmd_res = "0"

            else:
                self.get_logger().warn(f'[SimBridge] Unknown command: {cmd}')
                response.cmd_res = "-1"

        except Exception as e:
            self.get_logger().error(f'[SimBridge] Error handling command: {e}')
            self._publish_diag(DiagnosticStatus.ERROR, f'Command error: {e}')
            response.cmd_res = "-1"

        return response

    def _simulate_movel(self, cmd: str) -> str:
        """Apply the offset and simulate a short motion delay."""
        if not self._offset_enabled:
            return "0"

        dx, dy, dz, drx, dry, drz = self._offset
        dist_mm = math.sqrt(dx * dx + dy * dy + dz * dz)
        dist_deg = math.sqrt(drx * drx + dry * dry + drz * drz)

        # Simulate travel time
        t_mm = dist_mm / max(self.speed_mm, 1e-6)
        t_deg = dist_deg / max(self.speed_deg, 1e-6)
        delay = max(t_mm, t_deg, 0.02)

        # Apply to pose
        self.pose[0] += dx
        self.pose[1] += dy
        self.pose[2] += dz
        self.pose[3] += drx
        self.pose[4] += dry
        self.pose[5] += drz

        # Simulated motion applied instantly — no blocking sleep
        # (previous time.sleep blocked the ROS executor and starved state publishing)

        self._offset_enabled = False
        self._offset = [0.0] * 6

        self.get_logger().debug(
            f'[SimBridge] MoveL → pose=[{self.pose[0]:.1f}, {self.pose[1]:.1f}, '
            f'{self.pose[2]:.1f}] delay={delay:.3f}s'
        )
        return "0"

    # ── state publisher ─────────────────────────────────────

    def _publish_state(self):
        msg = RobotNonrtState()
        msg.cart_x_cur_pos = self.pose[0]
        msg.cart_y_cur_pos = self.pose[1]
        msg.cart_z_cur_pos = self.pose[2]
        msg.cart_a_cur_pos = self.pose[3]
        msg.cart_b_cur_pos = self.pose[4]
        msg.cart_c_cur_pos = self.pose[5]
        # Leave remaining fields at their default (0)
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotSimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
