"""Simulation — outside navigation only (single camera).

Camera index is configured in ``config/outside_nav.yaml`` → ``video_idx``.

Usage::

    ros2 launch fr5_vision_control sim_outside.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('fr5_vision_control')

    sim_bridge = Node(
        package='fr5_vision_control',
        executable='robot_sim_bridge',
        name='robot_sim_bridge',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_share, 'config', 'simulation.yaml']),
        ],
    )

    move_outside = Node(
        package='fr5_vision_control',
        executable='move_outside_node',
        name='move_outside_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([pkg_share, 'config', 'outside_nav.yaml']),
            {'simulate': True},
        ],
    )

    return LaunchDescription([
        sim_bridge,
        move_outside,
    ])
