"""Launch outside (nostril-approach) navigation.

Usage::

    ros2 launch fr5_vision_control outside_nav.launch.py

Equivalent to (in separate terminals)::

    ros2 run fairino_hardware ros2_cmd_server
    ros2 run fr5_vision_control move_outside_node --ros-args -p simulate:=false
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('fr5_vision_control')

    # ── Arguments ───────────────────────────────────────────
    simulate_arg = DeclareLaunchArgument(
        'simulate', default_value='false',
        description='Run in simulation mode (skip hardware)',
    )
    config_arg = DeclareLaunchArgument(
        'config', default_value='outside_nav.yaml',
        description='YAML parameter file name (inside config/)',
    )

    simulate = LaunchConfiguration('simulate')
    config_file = LaunchConfiguration('config')

    # ── Nodes ───────────────────────────────────────────────
    move_outside = Node(
        package='fr5_vision_control',
        executable='move_outside_node',
        name='move_outside_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_share, 'config', config_file]),
            {'simulate': simulate},
        ],
    )

    return LaunchDescription([
        simulate_arg,
        config_arg,
        move_outside,
    ])
