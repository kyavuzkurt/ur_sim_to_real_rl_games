#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Get the package directories
    ur_sim_to_real_pkg_dir = get_package_share_directory('ur_sim_to_real_rl_games')
    ur_simulation_gz_pkg_dir = get_package_share_directory('ur_simulation_gz')
    
    # Parameters (can be overridden with command line arguments)
    ur_type = LaunchConfiguration('ur_type', default='ur3')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    publish_rate = LaunchConfiguration('publish_rate', default='20.0')
    
    # Declare launch arguments
    declared_arguments = [
        DeclareLaunchArgument(
            'ur_type',
            default_value='ur3',
            description='Type of UR robot: ur3, ur5, ur10, ur3e, ur5e, ur10e, etc.'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='5.0',
            description='Rate (seconds) at which to publish new target poses'
        ),
    ]
    
    # Define model path
    model_path = os.path.join(ur_sim_to_real_pkg_dir, 'models', 'reach_ur10.pth')
    
    # Define config file path - using single consolidated config file
    config_file = os.path.join(ur_sim_to_real_pkg_dir, 'config', 'simulation.yaml')
    
    # Launch the UR simulation
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_simulation_gz_pkg_dir, 'launch', 'ur_sim_control.launch.py')),
        launch_arguments={
            'ur_type': ur_type,
            'use_sim_time': use_sim_time,
        }.items()
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
    
    # Launch the controller node
    controller_node = Node(
        package='ur_sim_to_real_rl_games',
        executable='controller',
        name='controller',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
        parameters=[config_file]
    )
    
    # Log model path
    log_model_path = ExecuteProcess(
        cmd=['echo', f'Using model path: {model_path}'],
        output='screen'
    )
    
    # Launch the target pose publisher
    target_pose_publisher_node = Node(
        package='ur_sim_to_real_rl_games',
        executable='target_pose_publisher',
        name='target_pose_publisher',
        output='screen',
        parameters=[
            config_file,
            {'publish_rate': publish_rate}  # Override with launch argument
        ]
    )
    
    # Launch the pose visualizer for RViz
    pose_visualizer_node = Node(
        package='ur_sim_to_real_rl_games',
        executable='pose_visualizer',
        name='pose_visualizer',
        output='screen',
        parameters=[config_file]
    )
    
    # Launch RViz for visualization
    rviz_config_file = os.path.join(ur_simulation_gz_pkg_dir, 'rviz', 'ur.rviz')
    if not os.path.exists(rviz_config_file):
        # Fallback to default RViz config if specific one not found
        rviz_config_file = os.path.join(get_package_share_directory('rviz2'), 'default.rviz')
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    return LaunchDescription(declared_arguments + [
        log_model_path,
        simulation_launch,
        policy_node,
        controller_node,
        target_pose_publisher_node,
        pose_visualizer_node,
    ]) 