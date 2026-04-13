import time
import math
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from fairino_msgs.srv import RemoteCmdInterface
from fairino_msgs.msg import RobotNonrtState
import threading

class RobotController:
    def __init__(self, node: Node, simulate=False):
        self.node = node
        self.simulate = simulate 
        self.connected = False

        self.tool_id = 1
        self.user_id = 0
        self.temp_point_id = 1 # 用于存储基准点的ID
        self.current_pose = None # 缓存的当前位姿

        # 创建独立的回调组
        self.client_cb_group = MutuallyExclusiveCallbackGroup()
        self.sub_cb_group = MutuallyExclusiveCallbackGroup()

        if not self.simulate:
            # 将服务客户端与订阅器分配到单独的回调组
            self.cli = node.create_client(
                RemoteCmdInterface, 
                '/fairino_remote_command_service',
                callback_group=self.client_cb_group
            )
            while not self.cli.wait_for_service(timeout_sec=1.0):
                node.get_logger().info('等待 /fairino_remote_command_service 服务...')
            node.get_logger().info('服务客户端已创建')

            self.sub = node.create_subscription(
                RobotNonrtState,
                '/nonrt_state_data',
                self.state_callback,
                10,
                callback_group=self.sub_cb_group
            )
            node.get_logger().info('已订阅 /nonrt_state_data')
        else:
            node.get_logger().info('模拟模式：不创建实际服务客户端')

        self.connected = True

    def state_callback(self, msg):
        """更新当前位姿缓存，从 RobotNonrtState 中提取笛卡尔位姿"""
        # 注意：ROS消息中的角度通常是弧度，如果 Fairino SDK 期望角度，可能需要转换
        # 这里假设 Fairino SDK 的 a,b,c 就是角度制
        self.current_pose = [
            msg.cart_x_cur_pos,
            msg.cart_y_cur_pos,
            msg.cart_z_cur_pos,
            msg.cart_a_cur_pos, # Rx (A)
            msg.cart_b_cur_pos, # Ry (B)
            msg.cart_c_cur_pos  # Rz (C)
        ]

    def _call_service(self, command_str, timeout=3.0):
        """
        发送指令字符串到服务，返回 (success, ret_code_str)
        """
        if self.simulate:
            self.node.get_logger().info(f"[模拟] 指令: {command_str}")
            return True, "0"

        req = RemoteCmdInterface.Request()
        req.cmd_str = command_str
        future = self.cli.call_async(req)

        event = threading.Event()
        future.add_done_callback(lambda f: event.set())

        if event.wait(timeout):
            try:
                resp = future.result()
                success = (resp.cmd_res == "0")
                return success, resp.cmd_res
            except Exception as e:
                self.node.get_logger().error(f"服务调用异常: {e}，指令：{command_str}")
                return False, "-1"
        else:
            future.cancel()
            self.node.get_logger().error(f"服务调用超时: {command_str}")
            return False, "-1"

    def get_current_pose(self, timeout=1.0):
        """等待获取最新位姿，若缓存为空则等待"""
        start = time.time()
        while self.current_pose is None and not self.simulate:
            if time.time() - start > timeout:
                self.node.get_logger().error("获取当前位姿超时")
                return None
            time.sleep(0.01) # 短暂等待
        return self.current_pose

    def clear_errors(self):
        """调用 ResetAllError 清除机械臂错误"""
        if self.simulate:
            self.node.get_logger().info("[模拟] 清除错误")
            return True
        success, ret = self._call_service("ResetAllError()", timeout=1.0)
        if not success or ret != "0":
            self.node.get_logger().warn(f"清除错误失败: {ret}")
            return False
        return True

    def move_offset_tool_frame(self, dx, dy, dz, max_retries=3):
        """
        【修改】工具坐标系下相对移动（ROS2 版），现在支持XYZ任意方向。
        用于 move_xy 和 move_z_only 的底层实现。
        """
        if not self.connected:
            self.node.get_logger().error("未连接到机械臂服务")
            return False
        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dz) < 1e-6:
            return True # 无需移动

        if self.simulate:
            self.node.get_logger().info(f"[模拟] 移动向量 -> X:{dx:.2f}, Y:{dy:.2f}, Z:{dz:.2f}")
            return True

        for attempt in range(max_retries):
            if attempt > 0:
                self.clear_errors()
                time.sleep(0.2)

            current = self.get_current_pose(timeout=1.0)
            if current is None:
                self.node.get_logger().warn(f"获取当前位姿失败，准备重试 {attempt + 1}/{max_retries}")
                continue

            cart_cmd = f"CARTPoint({self.temp_point_id},{current[0]:.3f},{current[1]:.3f},{current[2]:.3f},{current[3]:.3f},{current[4]:.3f},{current[5]:.3f})"
            success, ret = self._call_service(cart_cmd, timeout=2.0)
            if not success or ret != "0":
                self.node.get_logger().warn(f"CARTPoint 失败，返回码 {ret}，重试 {attempt + 1}/{max_retries}")
                continue

            offset_cmd = f"PointsOffsetEnable(2,{dx:.3f},{dy:.3f},{dz:.3f},0,0,0)" # 0,0,0 for rotation
            success, ret = self._call_service(offset_cmd, timeout=2.0)
            if not success or ret != "0":
                self.node.get_logger().warn(f"PointsOffsetEnable 失败，返回码 {ret}，重试 {attempt + 1}/{max_retries}")
                continue

            move_cmd = f"MoveL(CART{self.temp_point_id},5.0,{self.tool_id},{self.user_id},0,0,0,0)"
            success, ret = self._call_service(move_cmd, timeout=5.0)

            self._call_service("PointsOffsetDisable()", timeout=1.0) # 无论成功失败，关闭偏移

            if success and ret == "0":
                # self.node.get_logger().info(f"移动成功: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}")
                return True
            else:
                self.node.get_logger().error(f"MoveL执行失败: 返回码 {ret}, 尝试 {attempt + 1}/{max_retries}")
                time.sleep(0.5)

        self.node.get_logger().error(f"move_offset_tool_frame 失败，已重试 {max_retries} 次")
        return False

    def move_rotate_and_translate(self, rx, ry, dx, dy, dz=0.0, rz=0.0, max_retries=3):
        """
        【修改】复合运动：旋转 + 平移（符合法奥 ROS 服务语法的标准实现）
        现在可以同时指定 Rx, Ry, Rz 和 Dx, Dy, Dz。
        """
        if not self.connected:
            self.node.get_logger().error("未连接到机械臂服务")
            return False
        
        # 仅当所有参数都接近零时才直接返回 True
        if (abs(rx) < 1e-6 and abs(ry) < 1e-6 and abs(rz) < 1e-6 and
            abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dz) < 1e-6):
            return True

        if self.simulate:
            self.node.get_logger().info(f"[Sim] 旋转 Rx:{rx:.3f} Ry:{ry:.3f} Rz:{rz:.3f} | 平移 dx:{dx:.3f} dy:{dy:.3f} dz:{dz:.3f}")
            return True

        for attempt in range(max_retries):
            if attempt > 0:
                self.clear_errors()
                time.sleep(0.2)

            current = self.get_current_pose(timeout=1.0)
            if current is None:
                self.node.get_logger().warn(f"获取位姿失败，重试 {attempt + 1}/{max_retries}")
                continue

            cart_cmd = f"CARTPoint({self.temp_point_id},{current[0]:.3f},{current[1]:.3f},{current[2]:.3f},{current[3]:.3f},{current[4]:.3f},{current[5]:.3f})"
            success, ret = self._call_service(cart_cmd, timeout=2.0)
            if not success or ret != "0":
                self.node.get_logger().warn(f"CARTPoint 失败，返回码 {ret}，重试 {attempt + 1}/{max_retries}")
                continue

            # PointsOffsetEnable 的参数是 dx, dy, dz, dRx, dRy, dRz
            offset_cmd = f"PointsOffsetEnable(2,{dx:.3f},{dy:.3f},{dz:.3f},{rx:.3f},{ry:.3f},{rz:.3f})"
            success, ret = self._call_service(offset_cmd, timeout=2.0)
            if not success or ret != "0":
                self.node.get_logger().warn(f"PointsOffsetEnable 失败，返回码 {ret}，重试 {attempt + 1}/{max_retries}")
                continue

            move_cmd = f"MoveL(CART{self.temp_point_id},5.0,{self.tool_id},{self.user_id},0,0,0,0)"
            success, ret = self._call_service(move_cmd, timeout=5.0)

            self._call_service("PointsOffsetDisable()", timeout=1.0)

            if success and ret == "0":
                # self.node.get_logger().info("旋转+平移 执行成功！")
                return True
            else:
                self.node.get_logger().error(f"MoveL执行失败: 返回码 {ret}, 尝试 {attempt + 1}/{max_retries}")
                time.sleep(0.5)

        self.node.get_logger().error(f"move_rotate_and_translate 失败，已重试 {max_retries} 次")
        return False
        
    def rotate_tool_frame(self, rx, ry, rz=0.0, max_retries=3):
        """【新增】纯姿态旋转 (Rx, Ry, Rz) - 方便解耦调用"""
        return self.move_rotate_and_translate(rx, ry, 0.0, 0.0, 0.0, rz, max_retries)

    def move_xy(self, dx, dy, max_retries=3):
        """【新增】纯XY平移 - 方便解耦调用"""
        return self.move_offset_tool_frame(dx, dy, 0.0, max_retries)

    def move_z_only(self, dz, max_retries=3):
        """【新增】纯Z轴平移 - 方便解耦调用"""
        return self.move_offset_tool_frame(0.0, 0.0, dz, max_retries)

    def reset_all_error(self):
        """清除所有的错误状态（保留兼容性）"""
        return self.clear_errors()

    def disconnect(self):
        """断开操作时的清理"""
        if self.simulate:
            self.node.get_logger().info("[模拟] 断开")
        else:
            self.reset_all_error()
            self.node.get_logger().info("机械臂控制器：错误归零已下发")