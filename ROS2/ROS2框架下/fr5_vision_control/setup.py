import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'fr5_vision_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # YAML config files
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mio',
    maintainer_email='mio@todo.todo',
    description='FR5 vision-guided nasal endoscopy robot control',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # Core navigation
            'move_outside_node = fr5_vision_control.move_outside_node:main',
            'move_inside_node = fr5_vision_control.move_inside_node:main',
            # Simulation
            'robot_sim_bridge = fr5_vision_control.robot_sim_bridge:main',
            # Motion execution (sole RobotController owner for ROS-path mode)
            'motion_executor = fr5_vision_control.motion_executor:main',
            # Orchestration
            'command_arbiter = fr5_vision_control.command_arbiter:main',
            # Infrastructure
            'diagnostics_aggregator = fr5_vision_control.diagnostics_aggregator:main',
            'tf_broadcaster = fr5_vision_control.tf_broadcaster:main',
        ],
    },
)
