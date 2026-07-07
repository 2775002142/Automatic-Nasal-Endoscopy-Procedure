"""Command arbiter — transparent forwarder for motion commands.

Subscribes to outside and inside command topics and unconditionally
forwards every command to ``/robot/motion_command``.  The two navigation
nodes are guaranteed to never run simultaneously, so no source selection
or arbitration logic is needed.

Uses TRANSIENT_LOCAL QoS so late-joining subscribers (e.g. motion_executor)
immediately receive the latest command.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nasal_endoscopy_msgs.msg import MotionCommand
from diagnostic_msgs.msg import DiagnosticStatus


class CommandArbiter(Node):
    """Transparent relay: /control/{outside,inside}/command → /robot/motion_command.

    Publishes
    ---------
    /robot/motion_command : MotionCommand
        TRANSIENT_LOCAL so late joiners always see the latest command.

    Subscribes
    ----------
    /control/outside/command : MotionCommand
    /control/inside/command  : MotionCommand
    """

    def __init__(self):
        super().__init__('command_arbiter')

        # ── QoS: TRANSIENT_LOCAL for motion commands ────────
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_ALL,
        )

        self.cmd_pub = self.create_publisher(
            MotionCommand, '/robot/motion_command', cmd_qos,
        )

        # ── Subscriptions (both forward unconditionally) ───
        self.outside_sub = self.create_subscription(
            MotionCommand, '/control/outside/command', self._forward, 10,
        )
        self.inside_sub = self.create_subscription(
            MotionCommand, '/control/inside/command', self._forward, 10,
        )

        # ── Diagnostics ─────────────────────────────────────
        self._diag_pub = self.create_publisher(DiagnosticStatus, '/diagnostics/update', 10)

        self._publish_diag(DiagnosticStatus.OK, 'Arbiter ready')
        self.get_logger().info('[Arbiter] Ready — forwarding all commands')

    def _publish_diag(self, level: int, message: str):
        st = DiagnosticStatus()
        st.name = 'command_arbiter'
        st.level = level
        st.message = message
        st.hardware_id = 'arbiter'
        self._diag_pub.publish(st)

    def _forward(self, msg: MotionCommand):
        """Forward the command unchanged."""
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CommandArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
