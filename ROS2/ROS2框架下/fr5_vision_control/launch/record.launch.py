"""rosbag2 selective recording — records all key topics for post-hoc analysis.

Usage::

    ros2 launch fr5_vision_control record.launch.py
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess


TOPICS = [
    # Camera feeds
    '/camera/outside/image_raw',
    '/camera/inside/image_raw',
    # Robot state
    '/nonrt_state_data',
    '/tf',
    # Vision & navigation
    '/vision/outside/detection',
    '/vision/inside/apf',
    '/navigation/outside/state',
    '/navigation/inside/state',
    # Control pipeline (post Phase 2+7)
    '/control/outside/command',
    '/control/inside/command',
    '/robot/motion_command',
    '/robot/motion_result',
    # System health
    '/diagnostics',
    '/safety/status',
]


def generate_launch_description():
    record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', 'nasal_endoscopy_session'] + TOPICS,
        output='screen',
    )

    return LaunchDescription([record])
