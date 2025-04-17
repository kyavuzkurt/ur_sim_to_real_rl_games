from setuptools import find_packages, setup
import glob
import os
package_name = 'ur_sim_to_real_rl_games'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        ('share/' + package_name + '/models', glob.glob('models/*.pth')),
        ('share/' + package_name + '/config', glob.glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kadir Yavuz Kurt',
    maintainer_email='k.yavuzkurt1@gmail.com',
    description='Sim to real application for UR3 robot trained with RL Games library.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = ur_sim_to_real_rl_games.controller:main',
            'policy_node = ur_sim_to_real_rl_games.policy_node:main',
            'target_pose_publisher = ur_sim_to_real_rl_games.target_pose_publisher:main',
            'pose_publisher = ur_sim_to_real_rl_games.pose_publisher:main',
            'pose_visualizer = ur_sim_to_real_rl_games.pose_visualizer:main',
        ],
    },
)
