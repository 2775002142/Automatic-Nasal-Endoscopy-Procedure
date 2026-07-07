"""TF2 broadcaster — publishes the endoscope camera frame for RViz.

Reads ``/nonrt_state_data`` and broadcasts the dynamic transform::

    base_link → endoscope_camera

This is **Scene A** (visualisation only).  The control pipeline still
uses ``PointsOffsetEnable(flag=2)`` (tool-frame) — this TF tree does
NOT feed back into the control loop.

When a proper hand-eye calibration is completed, the TF tree can be
extended to::

    base_link → tool0 → endoscope_camera

without modifying any control code.
"""

import math
import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticStatus
from fairino_msgs.msg import RobotNonrtState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def cart_to_quaternion(rx_deg, ry_deg, rz_deg):
    """Convert Fairino Cartesian ABC angles (degrees) to a quaternion.

    The Fairino convention (as used in the SDK) is:
    A = Rx, B = Ry, C = Rz in degrees, applied in ZYX intrinsic order.
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    # ZYX intrinsic = Rz(rz) * Ry(ry) * Rx(rx)
    cy = math.cos(rz * 0.5)
    sy = math.sin(rz * 0.5)
    cp = math.cos(ry * 0.5)
    sp = math.sin(ry * 0.5)
    cr = math.cos(rx * 0.5)
    sr = math.sin(rx * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return qx, qy, qz, qw


class TFBroadcaster(Node):
    """Publishes ``base_link → endoscope_camera`` at the robot-state rate."""

    def __init__(self):
        super().__init__('tf_broadcaster')

        self.declare_parameter('parent_frame', 'base_link')
        self.declare_parameter('child_frame', 'endoscope_camera')

        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value

        self._tf_broadcaster = TransformBroadcaster(self)

        self._sub = self.create_subscription(
            RobotNonrtState,
            '/nonrt_state_data',
            self._state_cb,
            10,
        )

        # ── Diagnostics ─────────────────────────────────────
        self._diag_pub = self.create_publisher(DiagnosticStatus, '/diagnostics/update', 10)
        st = DiagnosticStatus()
        st.name = 'tf_broadcaster'
        st.level = DiagnosticStatus.OK
        st.message = f'Broadcasting {self.parent_frame}→{self.child_frame}'
        st.hardware_id = 'tf'
        self._diag_pub.publish(st)

        self.get_logger().info(
            f'[TF] Broadcasting: {self.parent_frame} → {self.child_frame}'
        )

    def _state_cb(self, msg: RobotNonrtState):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        # Translation: mm → m
        t.transform.translation.x = msg.cart_x_cur_pos / 1000.0
        t.transform.translation.y = msg.cart_y_cur_pos / 1000.0
        t.transform.translation.z = msg.cart_z_cur_pos / 1000.0

        # Rotation: Fairino ABC (deg) → quaternion
        qx, qy, qz, qw = cart_to_quaternion(
            msg.cart_a_cur_pos, msg.cart_b_cur_pos, msg.cart_c_cur_pos,
        )
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self._tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = TFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
