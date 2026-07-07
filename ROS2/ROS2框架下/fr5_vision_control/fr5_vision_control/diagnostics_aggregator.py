"""Diagnostics aggregator — system-wide health using standard ``diagnostic_msgs``.

Subscribes to ``/diagnostics/update`` from all subsystems and publishes a
consolidated ``diagnostic_msgs/DiagnosticArray`` at a regular interval.

Compatible with ``rqt_runtime_monitor`` out of the box.
"""

import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus


class DiagnosticsAggregator(Node):
    """Collects subsystem health via subscription and publishes DiagnosticArray."""

    def __init__(self):
        super().__init__('diagnostics_aggregator')

        self.declare_parameter('publish_hz', 2.0)

        # ── Aggregated output ──
        self._pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self._timer = self.create_timer(
            1.0 / self.get_parameter('publish_hz').value,
            self._publish,
        )

        # ── Input: listen for status updates from all nodes ──
        self._statuses: dict[str, DiagnosticStatus] = {}
        self._update_sub = self.create_subscription(
            DiagnosticStatus,
            '/diagnostics/update',
            self._on_status_update,
            10,
        )

        # Self-report
        self._publish_self_status()
        self.get_logger().info('[Diagnostics] Aggregator started — listening on /diagnostics/update')

    # ── subscription callback ──────────────────────────────

    def _on_status_update(self, msg: DiagnosticStatus):
        """Receive a status update from one subsystem."""
        self._statuses[msg.name] = msg

    # ── self-monitor ───────────────────────────────────────

    def _publish_self_status(self):
        st = DiagnosticStatus()
        st.name = 'diagnostics_aggregator'
        st.level = DiagnosticStatus.OK
        st.message = 'Aggregator running'
        st.hardware_id = 'none'
        self._statuses[st.name] = st

    # ── publisher ──────────────────────────────────────────

    def _publish(self):
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = list(self._statuses.values())
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DiagnosticsAggregator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
