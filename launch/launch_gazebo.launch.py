#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('ur_sim_to_real_rl_games')
    
    # Include the main simulation launch file
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'simulation.launch.py')
        )
    )
    
    return LaunchDescription([
        simulation_launch
    ])
