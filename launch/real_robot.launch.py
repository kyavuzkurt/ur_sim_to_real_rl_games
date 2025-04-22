#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Get the package directories
    ur_sim_to_real_pkg_dir = get_package_share_directory('ur_sim_to_real_rl_games')
    
    # Launch arguments
    ur_type = LaunchConfiguration('ur_type', default='ur10')
    robot_ip = LaunchConfiguration('robot_ip', default='192.168.56.101')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Define model path
    model_path = os.path.join(ur_sim_to_real_pkg_dir, 'models', 'reach_ur10.pth')
    
    # Define config file path - using single consolidated config file
    config_file = os.path.join(ur_sim_to_real_pkg_dir, 'config', 'real_robot.yaml')
    
    # Declare launch arguments


    # Log model path
    log_model_path = ExecuteProcess(
        cmd=['echo', f'Using model path: {model_path}'],
        output='screen'
    )
    
    # Launch the policy node with the trained model
    policy_node = Node(
        package='ur_sim_to_real_rl_games',
        executable='policy_node',
        name='policy_node',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
        parameters=[
            config_file,
            {'model_path': model_path}  # Override the placeholder in config
        ]
    )
    
    # Launch the controller node to handle transformations and robot commands
    controller_node = Node(
        package='ur_sim_to_real_rl_games',
        executable='controller',
        name='controller',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
        parameters=[config_file]
    )
    
    # Launch the target pose publisher
    target_pose_publisher_node = Node(
        package='ur_sim_to_real_rl_games',
        executable='target_pose_publisher',
        name='target_pose_publisher',
        output='screen',
        parameters=[config_file]
    )
    
    # Launch the pose visualizer for RViz
    pose_visualizer_node = Node(
        package='ur_sim_to_real_rl_games',
        executable='pose_visualizer',
        name='pose_visualizer',
        output='screen',
        parameters=[config_file]
    )
    
    return LaunchDescription([
        log_model_path,
        policy_node,
        controller_node,
        target_pose_publisher_node,
        pose_visualizer_node
    ]) 