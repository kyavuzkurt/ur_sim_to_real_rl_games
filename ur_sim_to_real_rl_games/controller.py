#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32MultiArray, Bool
from builtin_interfaces.msg import Duration
import traceback

class Controller(Node):
    """Controller node for UR robots that processes policy outputs and sends commands to the robot."""
    def __init__(self):
        super().__init__('controller')
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]
        
        self.joint_name_to_idx = {
            'elbow_joint': 2,
            'shoulder_lift_joint': 1,
            'shoulder_pan_joint': 0,
            'wrist_1_joint': 3,
            'wrist_2_joint': 4,
            'wrist_3_joint': 5
        }
        
        self.pi = math.pi
        self.sim_dof_angle_limits = [
            (-360, 360, False),
            (-360, 360, False),
            (-360, 360, False),
            (-360, 360, False),
            (-360, 360, False),
            (-360, 360, False),
        ]
        
        self.servo_angle_limits = [
            (-2*self.pi, 2*self.pi),
            (-2*self.pi, 2*self.pi),
            (-2*self.pi, 2*self.pi),
            (-2*self.pi, 2*self.pi),
            (-2*self.pi, 2*self.pi),
            (-2*self.pi, 2*self.pi),
        ]
        
        # Initialization state tracking
        self.initialization_complete = False
        self.robot_ready = False
        
        # Declare parameters
        self.declare_parameter('action_scale', 1.0)
        self.declare_parameter('trajectory_duration', 0.5)
        self.declare_parameter('min_trajectory_duration', 0.5)
        self.declare_parameter('max_velocity', 0.8)
        self.declare_parameter('max_acceleration', 0.5)
        self.declare_parameter('max_jerk', 1.0)
        self.declare_parameter('moving_average', 0.5)
        self.declare_parameter('movement_deadband', 0.001)
        self.declare_parameter('zero_end_velocity', True)
        self.declare_parameter('use_linear_trajectory', True)  # New parameter for linear trajectories
        self.declare_parameter('num_waypoints', 5)  # Number of waypoints for multi-point trajectory
        
        self.declare_parameter('command_topic', '/joint_trajectory_controller/joint_trajectory')  
        self.declare_parameter('state_topic', '/joint_states')
        self.declare_parameter('use_relative_commands', True)
        self.declare_parameter('trajectory_topic', '/scaled_joint_trajectory_controller/joint_trajectory')
        self.declare_parameter('command_rate', 30.0)
        self.declare_parameter('policy_output_topic', '/policy_output')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('scale_factor', 0.5)
        
        self.declare_parameter('direct_joint_control', False)
        
        self.declare_parameter('has_gripper', False)
        self.declare_parameter('gripper_command_topic', '/gripper_command')
        
        self.declare_parameter('clip_actions', False)
        self.declare_parameter('clip_value', 100.0)
        self.declare_parameter('clip_observations', 100.0)
        
        # Initial position parameters from env.yaml
        self.declare_parameter('initialize_robot', True)
        self.declare_parameter('initialization_timeout', 10.0)  # seconds
        self.declare_parameter('initialization_tolerance', 0.05)  # radians
        self.declare_parameter('init_shoulder_pan_joint', 0.0)
        self.declare_parameter('init_shoulder_lift_joint', -1.712)
        self.declare_parameter('init_elbow_joint', 1.712)
        self.declare_parameter('init_wrist_1_joint', 0.0)
        self.declare_parameter('init_wrist_2_joint', 0.0)
        self.declare_parameter('init_wrist_3_joint', 0.0)
        
        # Get parameters
        self.action_scale = self.get_parameter('action_scale').value
        self.trajectory_duration = self.get_parameter('trajectory_duration').value
        self.min_trajectory_duration = self.get_parameter('min_trajectory_duration').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.max_jerk = self.get_parameter('max_jerk').value
        self.moving_average = self.get_parameter('moving_average').value
        self.movement_deadband = self.get_parameter('movement_deadband').value
        self.zero_end_velocity = self.get_parameter('zero_end_velocity').value
        self.use_linear_trajectory = self.get_parameter('use_linear_trajectory').value  # Get linear trajectory parameter
        self.num_waypoints = self.get_parameter('num_waypoints').value  # Get number of waypoints
        
        command_topic = self.get_parameter('command_topic').value
        state_topic = self.get_parameter('state_topic').value
        self.use_relative_commands = self.get_parameter('use_relative_commands').value
        self.trajectory_topic = self.get_parameter('trajectory_topic').value
        self.command_rate = self.get_parameter('command_rate').value
        self.policy_output_topic = self.get_parameter('policy_output_topic').value
        self.joint_topic = self.get_parameter('joint_topic').value
        self.scale_factor = self.get_parameter('scale_factor').value
        
        self.direct_joint_control = self.get_parameter('direct_joint_control').value
        
        self.has_gripper = self.get_parameter('has_gripper').value
        self.gripper_command_topic = self.get_parameter('gripper_command_topic').value
        
        self.clip_actions = self.get_parameter('clip_actions').value
        self.clip_value = self.get_parameter('clip_value').value
        self.clip_observations = self.get_parameter('clip_observations').value
        
        # Get initialization parameters
        self.initialize_robot = self.get_parameter('initialize_robot').value
        self.initialization_timeout = self.get_parameter('initialization_timeout').value
        self.initialization_tolerance = self.get_parameter('initialization_tolerance').value
        
        # Get initial joint positions
        self.initial_joint_positions = {
            'shoulder_pan_joint': self.get_parameter('init_shoulder_pan_joint').value,
            'shoulder_lift_joint': self.get_parameter('init_shoulder_lift_joint').value,
            'elbow_joint': self.get_parameter('init_elbow_joint').value,
            'wrist_1_joint': self.get_parameter('init_wrist_1_joint').value,
            'wrist_2_joint': self.get_parameter('init_wrist_2_joint').value,
            'wrist_3_joint': self.get_parameter('init_wrist_3_joint').value
        }
        
        # Create subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, 
            state_topic, 
            self.joint_states_callback,
            10)
        self.get_logger().info(f'Subscribed to {state_topic}')
        
        self.policy_output_sub = self.create_subscription(
            Float32MultiArray,
            self.policy_output_topic,
            self.policy_output_callback,
            10)
        self.get_logger().info(f'Subscribed to {self.policy_output_topic}')
        
        if self.has_gripper:
            self.gripper_sub = self.create_subscription(
                Float32MultiArray,
                self.gripper_command_topic,
                self.gripper_command_callback,
                10)
            self.get_logger().info(f'Subscribed to {self.gripper_command_topic}')
        
        # Create publishers
        self.joint_cmd_pub = self.create_publisher(
            JointTrajectory,
            command_topic,
            10)
        self.get_logger().info(f'Publishing to {command_topic}')
        
        # Publisher to communicate initialization status to policy node
        self.robot_ready_pub = self.create_publisher(
            Bool,
            '/robot_ready',
            10)
        self.get_logger().info('Publishing to /robot_ready')
        
        if self.has_gripper:
            self.gripper_pub = self.create_publisher(
                Float32MultiArray,
                '/gripper_control',
                10)
            self.get_logger().info(f'Publishing gripper commands to /gripper_control')
        
        # State variables
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_joint_dict = {}
        self.previous_target_positions = None
        self.previous_target_velocities = None
        self.last_trajectory_end_time = None  # Track when the last trajectory should end
        self.last_action_time = self.get_clock().now()
        self.expected_action_dim = 7 if self.has_gripper else 6
        self.initialization_start_time = None
        
        # Calculate a more conservative trajectory time based on command rate
        # Using 1.5x the command period to ensure enough time for execution
        self.trajectory_time_from_start = 1.5 / self.command_rate 
        self.get_logger().info(f'Trajectory time from start: {self.trajectory_time_from_start}s')
        
        self.timer = self.create_timer(1.0 / self.command_rate, self.timer_callback)
        self.get_logger().info(f'Created timer at {self.command_rate} Hz')
        
        self.log_initialization_parameters()
    
    def log_initialization_parameters(self):
        """Log all initialization parameters for better debugging."""
        self.get_logger().info('Controller node initialized')
        self.get_logger().info(f'Action scale: {self.action_scale}')
        self.get_logger().info(f'Moving average: {self.moving_average}')
        self.get_logger().info(f'Max velocity: {self.max_velocity} rad/s')
        self.get_logger().info(f'Max acceleration: {self.max_acceleration} rad/s²')
        self.get_logger().info(f'Max jerk: {self.max_jerk} rad/s³')
        self.get_logger().info(f'Movement deadband: {self.movement_deadband} rad')
        self.get_logger().info(f'Zero end velocity: {self.zero_end_velocity}')
        self.get_logger().info(f'Joint names: {self.joint_names}')
        self.get_logger().info(f'Joint name to index mapping: {self.joint_name_to_idx}')
        self.get_logger().info(f'Using relative commands: {self.use_relative_commands}')
        self.get_logger().info(f'Scale factor: {self.scale_factor}')
        self.get_logger().info(f'Direct joint control: {self.direct_joint_control}')
        self.get_logger().info(f'Has gripper: {self.has_gripper}')
        self.get_logger().info(f'Expected action dimension: {self.expected_action_dim}')
        self.get_logger().info(f'Clip actions: {self.clip_actions}')
        self.get_logger().info(f'Clip value: {self.clip_value}')
        self.get_logger().info(f'Clip observations: {self.clip_observations}')
        self.get_logger().info(f'Using linear trajectory: {self.use_linear_trajectory}')
        self.get_logger().info(f'Number of waypoints: {self.num_waypoints}')
        
        # Log initialization parameters
        self.get_logger().info(f'Initialize robot: {self.initialize_robot}')
        if self.initialize_robot:
            self.get_logger().info(f'Initial joint positions: {self.initial_joint_positions}')
            self.get_logger().info(f'Initialization timeout: {self.initialization_timeout} seconds')
            self.get_logger().info(f'Initialization tolerance: {self.initialization_tolerance} radians')
    
    def initialize_robot_position(self):
        """Move the robot to the initial position specified in the parameters with improved trajectory."""
        if self.current_joint_positions is None:
            self.get_logger().warn("Waiting for joint state feedback before initialization")
            return False
        
        # Convert from dictionary to ordered list
        initial_positions = []
        for joint_name in self.joint_names:
            if joint_name in self.initial_joint_positions:
                initial_positions.append(self.initial_joint_positions[joint_name])
            else:
                initial_positions.append(0.0)
                self.get_logger().warn(f"No initial position specified for {joint_name}, using 0.0")
        
        # Check if robot is already at the initial position
        at_initial_position = True
        for i, (current, target) in enumerate(zip(self.current_joint_positions, initial_positions)):
            if abs(current - target) > self.initialization_tolerance:
                at_initial_position = False
                self.get_logger().info(f"Joint {self.joint_names[i]} needs to move: {current:.4f} -> {target:.4f}")
                break
        
        if at_initial_position:
            self.get_logger().info("Robot is already at initial position")
            return True
        
        # Create the initialization trajectory
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = self.joint_names
        
        # Calculate an appropriate duration based on the maximum joint movement
        max_movement = 0.0
        for current, target in zip(self.current_joint_positions, initial_positions):
            movement = abs(target - current)
            max_movement = max(max_movement, movement)
        
        # Use a slow and safe speed for initialization
        # For larger movements, we need more time
        safe_velocity = self.max_velocity
        duration = max(max_movement / safe_velocity, 2.0)  # At least 3 seconds
        
        self.get_logger().info(f"Moving to initial position with duration {duration:.2f} seconds")
        
        # Generate a multi-waypoint trajectory for smooth initialization
        trajectory_points = self.generate_linear_trajectory(
            self.current_joint_positions, initial_positions, duration)
        
        # Add points to trajectory message
        for point_data in trajectory_points:
            point = JointTrajectoryPoint()
            point.positions = point_data['positions']
            point.velocities = point_data['velocities']
            point.accelerations = point_data['accelerations']
            
            # Set time from start
            time_from_start = point_data['time_from_start']
            sec = int(time_from_start)
            nanosec = int((time_from_start - sec) * 1e9)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nanosec
            
            traj_msg.points.append(point)
        
        # Publish the trajectory
        self.joint_cmd_pub.publish(traj_msg)
        self.get_logger().info(f"Published initialization trajectory with {len(trajectory_points)} waypoints")
        
        # Record expected end time
        self.last_trajectory_end_time = self.get_clock().now() + rclpy.time.Duration(seconds=duration)
        
        return False
    
    def check_initialization_status(self):
        """Check if the robot has reached the initial position."""
        if not self.initialize_robot:
            # If initialization is disabled, just return True
            return True
            
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint positions available, cannot check initialization status")
            return False
            
        # Check if initialization timeout has occurred
        if self.initialization_start_time:
            current_time = self.get_clock().now()
            elapsed = (current_time - self.initialization_start_time).nanoseconds / 1e9
            
            if elapsed > self.initialization_timeout:
                self.get_logger().warn(f"Initialization timeout after {elapsed:.2f} seconds")
                self.get_logger().warn("Proceeding anyway - CHECK ROBOT POSITION!")
                return True
        
        # Convert from dictionary to ordered list for comparison
        initial_positions = []
        for joint_name in self.joint_names:
            if joint_name in self.initial_joint_positions:
                initial_positions.append(self.initial_joint_positions[joint_name])
            else:
                initial_positions.append(0.0)
        
        # Check if robot is at the initial position
        at_initial_position = True
        for i, (current, target) in enumerate(zip(self.current_joint_positions, initial_positions)):
            if abs(current - target) > self.initialization_tolerance:
                at_initial_position = False
                self.get_logger().debug(f"Joint {self.joint_names[i]} not at target: {current:.4f} vs {target:.4f}")
                break
        
        if at_initial_position:
            self.get_logger().info("Robot has reached initial position")
            return True
        else:
            return False
            
    def publish_robot_ready_status(self, ready=True):
        """Publish the robot ready status to inform the policy node."""
        msg = Bool()
        msg.data = ready
        self.robot_ready_pub.publish(msg)
        if ready:
            self.get_logger().info("Published READY status - policy can now execute")
        else:
            self.get_logger().info("Published NOT READY status - policy execution suspended")
    
    def joint_states_callback(self, msg):
        """Process joint state messages and update current joint positions and velocities."""
        try:
            self.current_joint_dict = {}
            
            for i, name in enumerate(msg.name):
                if name in self.joint_name_to_idx:
                    self.current_joint_dict[name] = msg.position[i]
            
            if len(self.current_joint_dict) >= 6:
                self.current_joint_positions = [0.0] * 6
                for name, idx in self.joint_name_to_idx.items():
                    if name in self.current_joint_dict:
                        self.current_joint_positions[idx] = self.current_joint_dict[name]
                
                self.current_joint_velocities = [0.0] * 6
                
                if msg.velocity and len(msg.velocity) == len(msg.name):
                    vel_dict = {}
                    for i, name in enumerate(msg.name):
                        if name in self.joint_name_to_idx:
                            vel_dict[name] = msg.velocity[i]
                    
                    for name, idx in self.joint_name_to_idx.items():
                        if name in vel_dict:
                            self.current_joint_velocities[idx] = vel_dict[name]
                
                self.get_logger().debug(f'Received joint states: pos={self.current_joint_positions}, vel={self.current_joint_velocities}')
            else:
                self.get_logger().warn(f'Incomplete joint states received: {msg.name}')
        except Exception as e:
            self.get_logger().error(f'Error in joint_states_callback: {str(e)}')
            self.get_logger().error(traceback.format_exc())
    
    def policy_output_callback(self, msg):
        """Process policy output and send trajectory commands to the robot."""
        try:
            # Ignore policy outputs during initialization
            if not self.initialization_complete:
                self.get_logger().info('Received policy output but robot initialization is not complete yet - ignoring command')
                return
                
            if self.current_joint_positions is None:
                self.get_logger().warn('Received policy output but no joint positions available yet')
                return
            
            actions = np.array(msg.data)
            self.get_logger().debug(f'Received policy output: {actions}')
            
            if len(actions) != 6:
                self.get_logger().warn(f'Expected 6 arm actions, got {len(actions)}')
                if len(actions) > 6:
                    self.get_logger().warn(f'Truncating to first 6 values for arm control')
                    actions = actions[:6]
                elif len(actions) < 6:
                    self.get_logger().error(f'Not enough action values, cannot control arm')
                    return
            
            if self.clip_actions:
                pre_clip = actions.copy()
                actions = np.clip(actions, -self.clip_value, self.clip_value)
                if not np.array_equal(pre_clip, actions):
                    self.get_logger().info(f'Actions clipped from {pre_clip} to {actions}')
            
            if self.direct_joint_control:
                target_positions = actions.copy()
                self.get_logger().info(f"Using direct joint targets: {target_positions}")
                
                self.publish_commands(target_positions)
            else:
                if self.use_relative_commands:
                    # Scale the actions properly for relative commands
                    scaled_actions = actions * self.scale_factor
                    target_positions = np.array(self.current_joint_positions) + scaled_actions
                    self.get_logger().debug(f'Target positions (relative): {target_positions}')
                else:
                    # For absolute commands, scale appropriately
                    target_positions = actions * self.scale_factor
                    self.get_logger().debug(f'Target positions (absolute): {target_positions}')
                
                self.publish_trajectory(target_positions)
            
        except Exception as e:
            self.get_logger().error(f'Error in policy_output_callback: {str(e)}')
            self.get_logger().error(traceback.format_exc())
    
    def gripper_command_callback(self, msg):
        """Process gripper commands from the policy node."""
        try:
            if not self.has_gripper:
                self.get_logger().warn("Received gripper command but gripper is not enabled")
                return
                
            if len(msg.data) > 0:
                gripper_command = msg.data[0]
                self.get_logger().info(f"Received gripper command: {gripper_command}")
                
                if self.clip_actions:
                    pre_clip = gripper_command
                    gripper_command = np.clip(gripper_command, -self.clip_value, self.clip_value)
                    if pre_clip != gripper_command:
                        self.get_logger().info(f'Gripper command clipped from {pre_clip} to {gripper_command}')
                
                if hasattr(self, 'gripper_pub'):
                    gripper_msg = Float32MultiArray()
                    gripper_msg.data = [gripper_command]
                    self.gripper_pub.publish(gripper_msg)
                    self.get_logger().debug(f"Published gripper command: {gripper_command}")
                
                self.get_logger().info("Gripper command processed")
            else:
                self.get_logger().warn("Received empty gripper command message")
                
        except Exception as e:
            self.get_logger().error(f"Error in gripper_command_callback: {str(e)}")
            self.get_logger().error(traceback.format_exc())
    
    def convert_sim_to_real_angles(self, actions):
        """Convert policy actions to real robot joint target angles."""
        if self.use_relative_commands:
            target_pos = np.array(self.current_joint_positions).copy()
            
            for i, angle in enumerate(actions):
                delta = angle * 0.05 * self.action_scale
                target_pos[i] += delta
                
                self.get_logger().debug(f'Joint {i}: action={angle:.4f}, delta={delta:.4f}, current={self.current_joint_positions[i]:.4f}, target={target_pos[i]:.4f}')
                
            for i in range(len(target_pos)):
                _, _, inversed = self.sim_dof_angle_limits[i]
                A, B = self.servo_angle_limits[i]
                
                if inversed:
                    target_pos[i] = -target_pos[i]
                
                pre_clip = target_pos[i]
                
                target_pos[i] = np.clip(target_pos[i], 
                                       max(A, -self.clip_value), 
                                       min(B, self.clip_value))
                
                if pre_clip != target_pos[i]:
                    self.get_logger().warn(f'Joint {i} clipped from {pre_clip:.4f} to {target_pos[i]:.4f}')
        else:
            target_pos = np.zeros(6)
            
            scale_factor = 0.5 * self.pi * self.action_scale
            
            for i, angle in enumerate(actions):
                scaled_angle = angle * scale_factor
                
                L, U, inversed = self.sim_dof_angle_limits[i]
                A, B = self.servo_angle_limits[i]
                
                if inversed:
                    scaled_angle = -scaled_angle
                    
                pre_clip = scaled_angle
                
                target_pos[i] = np.clip(scaled_angle, 
                                      max(A, -self.clip_value), 
                                      min(B, self.clip_value))
                
                self.get_logger().debug(f'Joint {i}: action={angle:.4f}, absolute target={target_pos[i]:.4f}')
                
                if pre_clip != target_pos[i]:
                    self.get_logger().warn(f'Joint {i} clipped from {pre_clip:.4f} to {target_pos[i]:.4f}')
            
        return target_pos
    
    def remap_joint_positions_to_real_robot(self, target_pos):
        """Map joint positions from simulation order to real robot order."""
        real_positions = {}
        
        for name, sim_idx in self.joint_name_to_idx.items():
            real_positions[name] = target_pos[sim_idx]
            
        ordered_positions = [real_positions.get(name, 0.0) for name in self.joint_names]
        
        return ordered_positions
    
    def calculate_velocities(self, current_positions, target_positions, duration):
        """Calculate appropriate velocities for smooth trajectory execution.
        
        Args:
            current_positions: Current joint positions
            target_positions: Target joint positions
            duration: Duration in seconds for the trajectory
            
        Returns:
            List of joint velocities
        """
        velocities = []
        for i in range(len(current_positions)):
            # Calculate the velocity needed to reach target in given duration
            velocity = (target_positions[i] - current_positions[i]) / duration
            
            # Limit the velocity to max_velocity
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            
            # If we're zeroing end velocities, apply a tapering factor (75% of calculated velocity)
            # This helps the robot come to a smooth stop
            if self.zero_end_velocity:
                velocity *= 0.75
                
            velocities.append(velocity)
            
        # Apply jerk limits if we have current velocities
        if self.current_joint_velocities:
            velocities = self.apply_jerk_limits(velocities, self.current_joint_velocities, duration)
        
        return velocities
    
    def calculate_accelerations(self, current_velocities, target_velocities, duration):
        """Calculate appropriate accelerations for smooth trajectory execution.
        
        Args:
            current_velocities: Current joint velocities
            target_velocities: Target joint velocities
            duration: Duration in seconds for the trajectory
            
        Returns:
            List of joint accelerations
        """
        accelerations = []
        for i in range(len(current_velocities)):
            # Calculate the acceleration needed to reach target velocity in given duration
            acceleration = (target_velocities[i] - current_velocities[i]) / duration
            
            # Limit the acceleration to max_acceleration
            acceleration = np.clip(acceleration, -self.max_acceleration, self.max_acceleration)
            accelerations.append(acceleration)
            
        return accelerations
    
    def calculate_trajectory_duration(self, current_positions, target_positions):
        """Calculate an appropriate trajectory duration based on the maximum joint movement.
        
        Args:
            current_positions: Current joint positions
            target_positions: Target joint positions
            
        Returns:
            Duration in seconds for the trajectory
        """
        durations = []
        
        for i in range(len(current_positions)):
            distance = abs(target_positions[i] - current_positions[i])
            
            # Skip joints with negligible movement
            if distance < self.movement_deadband:
                continue
            
            # Calculate minimum time needed at max velocity
            velocity_duration = distance / self.max_velocity
            
            # Calculate minimum time needed at max acceleration (uses motion equation: d = 0.5 * a * t^2)
            accel_duration = math.sqrt(2 * distance / self.max_acceleration)
            
            # Use the longer of the two times to ensure we respect both limits
            joint_duration = max(velocity_duration, accel_duration)
            durations.append(joint_duration)
        
        # Use the maximum duration across all joints
        if durations:
            max_duration = max(durations)
        else:
            # If no significant movement in any joint, use minimum duration
            max_duration = self.min_trajectory_duration / 2
        
        # Ensure duration is not shorter than min_trajectory_duration
        duration = max(max_duration, self.min_trajectory_duration)
        
        # For safety with real robot, add a small buffer
        duration = duration * 1.1
        
        return duration
    
    def smooth_target_positions(self, target_positions):
        """Apply smoothing to target positions using moving average with previous targets.
        
        Args:
            target_positions: New target joint positions
            
        Returns:
            Smoothed target positions
        """
        # Initialize previous target positions if not set
        if self.previous_target_positions is None:
            self.previous_target_positions = np.array(self.current_joint_positions)
            
        smoothed_positions = []
        
        # Apply moving average between previous target and new target
        for i in range(len(target_positions)):
            # Calculate the proposed change
            current_pos = self.current_joint_positions[i]
            target_pos = target_positions[i]
            position_delta = target_pos - current_pos
            
            # Apply deadband to ignore very small movements
            if abs(position_delta) < self.movement_deadband:
                # Skip the movement if it's below the deadband
                smoothed_pos = current_pos
                self.get_logger().debug(f'Joint {i} movement ({position_delta:.6f}) below deadband, skipping')
            else:
                # Apply smoothing with moving average
                smoothed_pos = (1 - self.moving_average) * self.previous_target_positions[i] + \
                              self.moving_average * target_positions[i]
            
            smoothed_positions.append(smoothed_pos)
            
        # Update previous target positions for next time
        self.previous_target_positions = np.array(smoothed_positions)
        
        return smoothed_positions
    
    def apply_jerk_limits(self, target_velocities, current_velocities, duration):
        """Limit jerk by constraining the rate of change of acceleration.
        
        Args:
            target_velocities: Target joint velocities
            current_velocities: Current joint velocities
            duration: Duration of the trajectory segment
            
        Returns:
            Jerk-limited velocities
        """
        if self.previous_target_velocities is None:
            # Initialize with zeros on first call
            self.previous_target_velocities = np.zeros_like(target_velocities)
        
        limited_velocities = []
        
        for i in range(len(target_velocities)):
            # Calculate the change in acceleration
            prev_accel = (self.previous_target_velocities[i] - current_velocities[i]) / duration
            target_accel = (target_velocities[i] - current_velocities[i]) / duration
            accel_change = target_accel - prev_accel
            
            # Calculate jerk (change in acceleration over time)
            jerk = accel_change / duration
            
            if abs(jerk) > self.max_jerk:
                # Limit the jerk by adjusting the target velocity
                max_accel_change = self.max_jerk * duration
                limited_accel = prev_accel + np.sign(accel_change) * max_accel_change
                limited_velocity = current_velocities[i] + limited_accel * duration
                limited_velocities.append(limited_velocity)
                self.get_logger().debug(f'Joint {i} jerk limited: {jerk:.4f} -> {self.max_jerk:.4f}')
            else:
                limited_velocities.append(target_velocities[i])
        
        # Update previous velocities for next call
        self.previous_target_velocities = np.array(limited_velocities)
        
        return limited_velocities
    
    def generate_linear_trajectory(self, current_positions, target_positions, duration):
        """Generate a linear trajectory with multiple waypoints.
        
        Args:
            current_positions: Current joint positions
            target_positions: Target joint positions
            duration: Duration of the trajectory in seconds
            
        Returns:
            List of trajectory points with positions, velocities, and times
        """
        num_points = max(2, int(self.num_waypoints))  # Ensure at least 2 points
        points = []
        
        # Calculate time interval between points
        time_interval = duration / (num_points - 1)
        
        for i in range(num_points):
            # For each waypoint, calculate the position as a linear interpolation
            # between current and target positions
            alpha = i / (num_points - 1)  # Interpolation factor (0 to 1)
            
            # Linear interpolation for position
            positions = []
            for j in range(len(current_positions)):
                # Simple linear interpolation: current + alpha * (target - current)
                pos = current_positions[j] + alpha * (target_positions[j] - current_positions[j])
                positions.append(pos)
            
            # Calculate velocities
            velocities = []
            for j in range(len(current_positions)):
                # Use constant velocity for intermediate points, zero at the end if specified
                if i < num_points - 1 or not self.zero_end_velocity:
                    velocity = (target_positions[j] - current_positions[j]) / duration
                    # Limit to max velocity
                    velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
                else:
                    # Last point has zero velocity if specified
                    velocity = 0.0
                
                velocities.append(velocity)
            
            # Calculate accelerations for better control
            accelerations = []
            for j in range(len(current_positions)):
                # Use acceleration primarily for the beginning and end of trajectory
                if i == 0:  # First point - accelerate from rest
                    accel = velocities[j] / time_interval
                elif i == num_points - 1 and self.zero_end_velocity:  # Last point - decelerate to stop
                    accel = -velocities[j] / time_interval
                else:  # Middle points - maintain velocity
                    accel = 0.0
                
                # Limit acceleration
                accel = np.clip(accel, -self.max_acceleration, self.max_acceleration)
                accelerations.append(accel)
            
            # Calculate time from start for this point
            time_from_start = i * time_interval
            
            # Add the point to the trajectory
            point = {
                'positions': positions,
                'velocities': velocities,
                'accelerations': accelerations,
                'time_from_start': time_from_start
            }
            points.append(point)
        
        return points

    def publish_commands(self, actions):
        """Process and publish actions directly to the joint trajectory controller."""
        try:
            if self.direct_joint_control:
                target_pos = actions.copy()
                
                for i in range(len(target_pos)):
                    A, B = self.servo_angle_limits[i]
                    
                    pre_clip = target_pos[i]
                    
                    target_pos[i] = np.clip(target_pos[i], 
                                           max(A, -self.clip_value), 
                                           min(B, self.clip_value))
                    
                    if pre_clip != target_pos[i]:
                        self.get_logger().warn(f'Joint {i} clipped from {pre_clip:.4f} to {target_pos[i]:.4f}')
                
                self.get_logger().info(f"Using raw joint targets from policy: {target_pos}")
            else:
                target_pos = self.convert_sim_to_real_angles(actions)
            
            # Use the improved trajectory publishing method
            self.publish_trajectory(target_pos)
            
        except Exception as e:
            self.get_logger().error(f'Error in publish_commands: {str(e)}')
            self.get_logger().error(traceback.format_exc())

    def publish_trajectory(self, positions):
        """Create and publish a trajectory message with improved waypoint generation."""
        try:
            # Check if we have received joint states
            if self.current_joint_positions is None:
                self.get_logger().warn("Cannot publish trajectory: no joint state information available")
                return
            
            # Create the trajectory message
            traj_msg = JointTrajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.joint_names = self.joint_names
            
            # Get current joint positions in the right order for the robot
            current_real_positions = self.remap_joint_positions_to_real_robot(self.current_joint_positions)
            
            # Get target positions in the right order for the robot
            real_positions = self.remap_joint_positions_to_real_robot(positions)
            
            # Apply smoothing to the target positions
            smoothed_positions = self.smooth_target_positions(real_positions)
            
            # Check if target position is significantly different from current position
            total_movement = 0.0
            for i in range(len(smoothed_positions)):
                total_movement += abs(smoothed_positions[i] - current_real_positions[i])
            
            # If total movement is too small, skip sending a new trajectory
            if total_movement < self.movement_deadband * len(smoothed_positions):
                self.get_logger().info(f'Total movement ({total_movement:.6f}) below threshold, skipping trajectory')
                return
            
            # Calculate appropriate trajectory duration based on movement distance
            duration = self.calculate_trajectory_duration(current_real_positions, smoothed_positions)
            
            # Check if we should wait for previous trajectory to complete
            current_time = self.get_clock().now()
            if self.last_trajectory_end_time is not None:
                time_since_last = (current_time - self.last_trajectory_end_time).nanoseconds / 1e9
                if time_since_last < 0:  # Previous trajectory hasn't completed yet
                    self.get_logger().info(f"Waiting for previous trajectory to complete ({-time_since_last:.2f}s remaining)")
                    return
            
            # Update the expected end time of this trajectory
            self.last_trajectory_end_time = current_time + rclpy.time.Duration(seconds=duration)
            
            # Generate a multi-waypoint trajectory for smoother motion
            trajectory_points = self.generate_linear_trajectory(
                current_real_positions, smoothed_positions, duration)
            
            # Add points to the trajectory message
            for point_data in trajectory_points:
                point = JointTrajectoryPoint()
                point.positions = point_data['positions']
                point.velocities = point_data['velocities']
                point.accelerations = point_data['accelerations']
                
                # Set time from start
                time_from_start = point_data['time_from_start']
                sec = int(time_from_start)
                nanosec = int((time_from_start - sec) * 1e9)
                point.time_from_start.sec = sec
                point.time_from_start.nanosec = nanosec
                
                traj_msg.points.append(point)
            
            # Publish the trajectory
            self.joint_cmd_pub.publish(traj_msg)
            self.get_logger().info(f'Published trajectory with {len(trajectory_points)} waypoints, duration: {duration:.2f}s')
            self.get_logger().debug(f'Start positions: {trajectory_points[0]["positions"]}')
            self.get_logger().debug(f'End positions: {trajectory_points[-1]["positions"]}')
            
        except Exception as e:
            self.get_logger().error(f'Error in publish_trajectory: {str(e)}')
            self.get_logger().error(traceback.format_exc())

    def timer_callback(self):
        """Timer callback to enforce consistent command rate and handle initialization."""
        # First, check if we have received joint states
        if self.current_joint_positions is None:
            self.get_logger().debug("Waiting for joint states...")
            return
        
        # Handle initialization sequence
        if not self.initialization_complete:
            # Start initialization timer if not started
            if self.initialization_start_time is None:
                self.initialization_start_time = self.get_clock().now()
                self.get_logger().info("Starting robot initialization sequence")
            
            # Try to initialize robot position
            if self.initialize_robot and not self.robot_ready:
                # Send the robot to initial position
                init_success = self.initialize_robot_position()
                if init_success:
                    self.robot_ready = True
            
            # Check if initialization is complete
            if self.robot_ready or self.check_initialization_status():
                self.robot_ready = True
                self.initialization_complete = True
                self.publish_robot_ready_status(True)
                self.get_logger().info("Initialization complete - ready for policy execution")
            else:
                # Still initializing
                self.publish_robot_ready_status(False)
                self.get_logger().debug("Continuing initialization process...")
        
        # Debug logging for joint positions
        if self.current_joint_positions is not None:
            self.get_logger().debug(f'Current joint positions: {self.current_joint_positions}')

def main(args=None):
    """Main function to initialize and run the controller node."""
    rclpy.init(args=args)
    try:
        node = Controller()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()