"""Full-system launch — infrastructure nodes (no navigation).

Staggered startup ensures dependencies are ready before subscribers.
Navigation nodes are launched separately via outside_nav.launch.py /
inside_nav.launch.py.

Usage::

    ros2 launch fr5_vision_control system.launch.py
    ros2 launch fr5_vision_control system.launch.py simulate:=true
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    pkg_share = FindPackageShare('fr5_vision_control')

    simulate_arg = DeclareLaunchArgument(
        'simulate', default_value='false',
        description='Run in simulation mode',
    )

    simulate = LaunchConfiguration('simulate')

    # ── Group 0 (t+0s): Infrastructure — no dependencies ───────
    tf_node = Node(
        package='fr5_vision_control',
        executable='tf_broadcaster',
        name='tf_broadcaster',
        output='screen',
    )
    diag_node = Node(
        package='fr5_vision_control',
        executable='diagnostics_aggregator',
        name='diagnostics_aggregator',
        output='screen',
    )

    # ── Group 1 (t+1s): Sim bridge ──────────────────────────────
    sim_bridge = Node(
        package='fr5_vision_control',
        executable='robot_sim_bridge',
        name='robot_sim_bridge',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_share, 'config', 'simulation.yaml']),
        ],
        condition=IfCondition(simulate),
    )

    # ── Group 2 (t+2s): Arbiter + motion executor ───────────────
    arbiter = Node(
        package='fr5_vision_control',
        executable='command_arbiter',
        name='command_arbiter',
        output='screen',
        parameters=[],
    )
    motion_exec = Node(
        package='fr5_vision_control',
        executable='motion_executor',
        name='motion_executor',
        output='screen',
        parameters=[{'simulate': simulate}],
    )

    return LaunchDescription([
        simulate_arg,
        # Group 0: immediate
        tf_node,
        diag_node,
        # Group 1: t+1s
        TimerAction(period=1.0, actions=[sim_bridge]),
        # Group 2: t+2s
        TimerAction(period=2.0, actions=[arbiter]),
        TimerAction(period=2.0, actions=[motion_exec]),
    ])
