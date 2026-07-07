"""Launch inside (lumen-following) navigation.

Usage::

    ros2 launch fr5_vision_control inside_nav.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('fr5_vision_control')

    simulate_arg = DeclareLaunchArgument(
        'simulate', default_value='false',
    )
    config_arg = DeclareLaunchArgument(
        'config', default_value='inside_nav.yaml',
    )

    simulate = LaunchConfiguration('simulate')
    config_file = LaunchConfiguration('config')

    move_inside = Node(
        package='fr5_vision_control',
        executable='move_inside_node',
        name='move_inside_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_share, 'config', config_file]),
        ],
    )

    return LaunchDescription([
        simulate_arg,
        config_arg,
        move_inside,
    ])
