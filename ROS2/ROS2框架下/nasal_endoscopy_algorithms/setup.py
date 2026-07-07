from setuptools import find_packages, setup

setup(
    name='nasal_endoscopy_algorithms',
    version='0.1.0',
    description='Pure Python algorithms for nasal endoscopy robot navigation — zero ROS dependencies',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy>=1.21',
        'opencv-python>=4.5',
        'mediapipe>=0.10',
    ],
    python_requires='>=3.8',
    extras_require={
        'test': ['pytest'],
    },
)
