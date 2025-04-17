#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Float32MultiArray
import sys
import os
from builtin_interfaces.msg import Duration
import traceback

class Controller(Node):
    """
    Controller node for UR robots that processes policy outputs and sends commands to the robot.
    
    This controller is designed to handle the following:
    - Process 6-DOF joint states from the robot
    - Receive policy actions (6 values for arm joints, optional 1 for gripper)
    - Apply appropriate scaling and control to the robot
    - Publish joint trajectories for smooth motion
    """
    def __init__(self):
        super().__init__('controller')
        
        # Define joint name mapping according to the RealWorldUR10 class
        # Note the joint ordering difference between sim and real
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]
        
        # Joint mapping between simulation index and real robot joints
        self.joint_name_to_idx = {
            'elbow_joint': 2,
            'shoulder_lift_joint': 1,
            'shoulder_pan_joint': 0,
            'wrist_1_joint': 3,
            'wrist_2_joint': 4,
            'wrist_3_joint': 5
        }
        
        # Define joint angle limits
        self.pi = math.pi
        self.sim_dof_angle_limits = [
            (-360, 360, False),
            (-360, 360, False),
            (-360, 360, False),
            (-360, 360, False),
            (-360, 360, False),
            (-360, 360, False),
        ]
        
        # Real robot joint limits in radians - using more conservative limits
        self.servo_angle_limits = [
            (-2*self.pi, 2*self.pi),  # shoulder_pan
            (-2*self.pi, 2*self.pi),  # shoulder_lift
            (-2*self.pi, 2*self.pi),  # elbow
            (-2*self.pi, 2*self.pi),  # wrist_1
            (-2*self.pi, 2*self.pi),  # wrist_2
            (-2*self.pi, 2*self.pi),  # wrist_3
        ]
        
        # Declare parameters
        # Control parameters
        self.declare_parameter('action_scale', 1.0)  # Reduced from 5.0 to 1.0 for safety
        self.declare_parameter('trajectory_duration', 0.5)  # Increased for smoother motion
        self.declare_parameter('min_trajectory_duration', 0.2)  # Minimum trajectory duration
        self.declare_parameter('max_velocity', 1.0)  # Reduced max velocity for safety
        self.declare_parameter('moving_average', 0.5)  # Increased for smoother motion
        
        # Topics and robot interface parameters
        self.declare_parameter('command_topic', '/joint_trajectory_controller/joint_trajectory')  
        self.declare_parameter('state_topic', '/joint_states')
        self.declare_parameter('use_relative_commands', True)  # Use relative motion instead of absolute
        self.declare_parameter('trajectory_topic', '/scaled_joint_trajectory_controller/joint_trajectory')
        self.declare_parameter('command_rate', 30.0)  # Updated from 5.0 to 30.0 Hz for smoother control
        self.declare_parameter('policy_output_topic', '/policy_output')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('scale_factor', 0.5)
        
        # Control mode parameters
        self.declare_parameter('direct_joint_control', False)
        
        # Gripper parameters
        self.declare_parameter('has_gripper', False)
        self.declare_parameter('gripper_command_topic', '/gripper_command')
        
        # Policy configuration parameters
        self.declare_parameter('clip_actions', False)  # Match config algo.clip_actions: False
        self.declare_parameter('clip_value', 100.0)  # Match config env.clip_actions: 100.0
        self.declare_parameter('clip_observations', 100.0)  # Match config env.clip_observations: 100.0
        
        # Get parameters
        # Control parameters
        self.action_scale = self.get_parameter('action_scale').value
        self.trajectory_duration = self.get_parameter('trajectory_duration').value
        self.min_trajectory_duration = self.get_parameter('min_trajectory_duration').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.moving_average = self.get_parameter('moving_average').value
        
        # Topics and robot interface parameters
        command_topic = self.get_parameter('command_topic').value
        state_topic = self.get_parameter('state_topic').value
        self.use_relative_commands = self.get_parameter('use_relative_commands').value
        self.trajectory_topic = self.get_parameter('trajectory_topic').value
        self.command_rate = self.get_parameter('command_rate').value
        self.policy_output_topic = self.get_parameter('policy_output_topic').value
        self.joint_topic = self.get_parameter('joint_topic').value
        self.scale_factor = self.get_parameter('scale_factor').value
        
        # Control mode parameters
        self.direct_joint_control = self.get_parameter('direct_joint_control').value
        
        # Gripper parameters
        self.has_gripper = self.get_parameter('has_gripper').value
        self.gripper_command_topic = self.get_parameter('gripper_command_topic').value
        
        # Policy configuration parameters
        self.clip_actions = self.get_parameter('clip_actions').value
        self.clip_value = self.get_parameter('clip_value').value
        self.clip_observations = self.get_parameter('clip_observations').value
        
        # Create subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, 
            state_topic, 
            self.joint_states_callback,
            10)
        self.get_logger().info(f'Subscribed to {state_topic}')
        
        # Subscribe to policy outputs from the policy node
        self.policy_output_sub = self.create_subscription(
            Float32MultiArray,
            self.policy_output_topic,
            self.policy_output_callback,
            10)
        self.get_logger().info(f'Subscribed to {self.policy_output_topic}')
        
        # Subscribe to gripper commands if we have a gripper
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
        
        # Create a publisher for gripper commands if a gripper is present (not present in this case)
        if self.has_gripper:
            # Note: The actual message type and topic would depend on your gripper hardware
            # This is just a placeholder - adjust as needed for your gripper
            self.gripper_pub = self.create_publisher(
                Float32MultiArray,  # Replace with actual gripper control message type
                '/gripper_control', # Replace with actual gripper control topic
                10)
            self.get_logger().info(f'Publishing gripper commands to /gripper_control')
        
        # State variables
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_joint_dict = {}  # Store joint states by name
        self.last_action_time = self.get_clock().now()
        self.expected_action_dim = 7 if self.has_gripper else 6  # Expected action dimension
        
        # Calculate timing
        self.trajectory_time_from_start = 1.0 / self.command_rate
        self.get_logger().info(f'Trajectory time from start: {self.trajectory_time_from_start}s')
        
        # Create a timer to enforce command rate
        self.timer = self.create_timer(1.0 / self.command_rate, self.timer_callback)
        self.get_logger().info(f'Created timer at {self.command_rate} Hz')
        
        # Log initialization parameters
        self.log_initialization_parameters()
    
    def log_initialization_parameters(self):
        """Log all initialization parameters for better debugging."""
        self.get_logger().info('Controller node initialized')
        self.get_logger().info(f'Action scale: {self.action_scale}')
        self.get_logger().info(f'Moving average: {self.moving_average}')
        self.get_logger().info(f'Max velocity: {self.max_velocity} rad/s')
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
    
    def joint_states_callback(self, msg):
        """
        Process joint state messages and update current joint positions and velocities.
        
        This callback maps the joint states from the robot to the internal representation,
        handling differences in joint ordering between simulation and real robot.
        """
        try:
            # Map joint positions by name
            self.current_joint_dict = {}
            
            for i, name in enumerate(msg.name):
                if name in self.joint_name_to_idx:
                    self.current_joint_dict[name] = msg.position[i]
            
            # Create ordered joint positions list
            if len(self.current_joint_dict) >= 6:
                # Important: Remap joints to match the simulation joint order
                self.current_joint_positions = [0.0] * 6
                for name, idx in self.joint_name_to_idx.items():
                    if name in self.current_joint_dict:
                        self.current_joint_positions[idx] = self.current_joint_dict[name]
                
                self.current_joint_velocities = [0.0] * 6  # Initialize velocities if not available
                
                # Extract velocities if available
                if msg.velocity and len(msg.velocity) == len(msg.name):
                    vel_dict = {}
                    for i, name in enumerate(msg.name):
                        if name in self.joint_name_to_idx:
                            vel_dict[name] = msg.velocity[i]
                    
                    # Map velocities to the correct indices
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
        """
        Process policy output and send trajectory commands to the robot.
        
        This callback handles the arm actions from the policy node (6 values).
        If a gripper is present, the gripper action should be sent separately
        via the gripper_command_callback.
        """
        try:
            # Skip if we don't have joint positions yet
            if self.current_joint_positions is None:
                self.get_logger().warn('Received policy output but no joint positions available yet')
                return
            
            # Get actions from the message
            actions = np.array(msg.data)
            self.get_logger().debug(f'Received policy output: {actions}')
            
            # Verify action dimension matches expected (6 for arm)
            if len(actions) != 6:
                self.get_logger().warn(f'Expected 6 arm actions, got {len(actions)}')
                if len(actions) > 6:
                    # Take only the first 6 values for arm control
                    self.get_logger().warn(f'Truncating to first 6 values for arm control')
                    actions = actions[:6]
                elif len(actions) < 6:
                    # Not enough values, cannot proceed safely
                    self.get_logger().error(f'Not enough action values, cannot control arm')
                    return
            
            # Apply clipping if enabled (matches config.clip_actions)
            if self.clip_actions:
                pre_clip = actions.copy()
                actions = np.clip(actions, -self.clip_value, self.clip_value)
                if not np.array_equal(pre_clip, actions):
                    self.get_logger().info(f'Actions clipped from {pre_clip} to {actions}')
            
            # Create target positions
            if self.direct_joint_control:
                # In direct joint control mode, the policy outputs are the joint positions
                # IMPORTANT: Don't scale down by 0.1 as was happening before - use the full range
                target_positions = actions.copy()
                self.get_logger().info(f"Using direct joint targets: {target_positions}")
                
                # Publish directly to the joint_cmd_pub
                self.publish_commands(target_positions)
            else:
                # Traditional scaling approach
                if self.use_relative_commands:
                    # Scale actions and add to current positions
                    target_positions = np.array(self.current_joint_positions) + actions * self.scale_factor
                    self.get_logger().debug(f'Target positions (relative): {target_positions}')
                else:
                    # For absolute commands, directly use the actions as offset from zeros
                    target_positions = actions * self.scale_factor
                    self.get_logger().debug(f'Target positions (absolute): {target_positions}')
                
                # Create and publish trajectory
                self.publish_trajectory(target_positions)
            
        except Exception as e:
            self.get_logger().error(f'Error in policy_output_callback: {str(e)}')
            self.get_logger().error(traceback.format_exc())
    
    def gripper_command_callback(self, msg):
        """
        Process gripper commands from the policy node.
        
        This method handles the gripper control signal from the policy output.
        It assumes the gripper command is a single float value in the range [-1, 1],
        where -1 is fully closed and 1 is fully open (or vice versa, depending on
        the specific gripper implementation).
        """
        try:
            if not self.has_gripper:
                self.get_logger().warn("Received gripper command but gripper is not enabled")
                return
                
            # Extract the gripper command value
            if len(msg.data) > 0:
                gripper_command = msg.data[0]
                self.get_logger().info(f"Received gripper command: {gripper_command}")
                
                # Apply clipping if enabled (matches config.clip_actions)
                if self.clip_actions:
                    pre_clip = gripper_command
                    gripper_command = np.clip(gripper_command, -self.clip_value, self.clip_value)
                    if pre_clip != gripper_command:
                        self.get_logger().info(f'Gripper command clipped from {pre_clip} to {gripper_command}')
                
                # Here you would implement the actual gripper control
                # This implementation depends on your specific gripper hardware
                
                # Example: If using a ROS 2 compatible gripper with Float32MultiArray:
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
        """
        Convert policy actions to real robot joint target angles.
        
        The policy outputs actions in the range [-1, 1]. This function
        converts these values to appropriate joint targets, either as
        absolute positions or relative changes to current positions.
        
        Args:
            actions: Numpy array of 6 values representing arm actions
        
        Returns:
            Numpy array of 6 target joint positions
        """
        # If using relative commands, these will be delta angles to current position
        if self.use_relative_commands:
            target_pos = np.array(self.current_joint_positions).copy()
            
            # Map the policy actions to delta joint angles (in correct joint order)
            for i, angle in enumerate(actions):
                # Scale the actions from policy (-1, 1) to smaller radian delta
                # Using a smaller action scale for smoother, safer movement
                delta = angle * 0.05 * self.action_scale
                target_pos[i] += delta
                
                # Log the delta being applied
                self.get_logger().debug(f'Joint {i}: action={angle:.4f}, delta={delta:.4f}, current={self.current_joint_positions[i]:.4f}, target={target_pos[i]:.4f}')
                
            # Clip to joint limits
            for i in range(len(target_pos)):
                _, _, inversed = self.sim_dof_angle_limits[i]
                A, B = self.servo_angle_limits[i]
                
                if inversed:
                    target_pos[i] = -target_pos[i]
                
                # Store the pre-clipped value for logging
                pre_clip = target_pos[i]
                
                # Use configured clip value, but still respect joint limits
                target_pos[i] = np.clip(target_pos[i], 
                                       max(A, -self.clip_value), 
                                       min(B, self.clip_value))
                
                # Log if clipping occurred
                if pre_clip != target_pos[i]:
                    self.get_logger().warn(f'Joint {i} clipped from {pre_clip:.4f} to {target_pos[i]:.4f}')
        else:
            # For absolute commands - map policy outputs to absolute joint targets
            target_pos = np.zeros(6)
            
            # Using a more reasonable scale for absolute positioning
            # This maps the policy output range [-1, 1] to a portion of the joint range
            scale_factor = 0.5 * self.pi * self.action_scale  # Using half pi (90 degrees) as base scale
            
            for i, angle in enumerate(actions):
                # Scale the actions from policy (-1, 1) to a reasonable angle range
                scaled_angle = angle * scale_factor
                
                # Map from simulation angle limits to real robot limits
                L, U, inversed = self.sim_dof_angle_limits[i]
                A, B = self.servo_angle_limits[i]
                
                # Apply any needed inversions
                if inversed:
                    scaled_angle = -scaled_angle
                    
                # Clip to joint limits
                pre_clip = scaled_angle
                
                # Use configured clip value, but still respect joint limits
                target_pos[i] = np.clip(scaled_angle, 
                                      max(A, -self.clip_value), 
                                      min(B, self.clip_value))
                
                # Log the conversion
                self.get_logger().debug(f'Joint {i}: action={angle:.4f}, absolute target={target_pos[i]:.4f}')
                
                # Log if clipping occurred
                if pre_clip != target_pos[i]:
                    self.get_logger().warn(f'Joint {i} clipped from {pre_clip:.4f} to {target_pos[i]:.4f}')
            
        return target_pos
    
    def remap_joint_positions_to_real_robot(self, target_pos):
        """
        Map joint positions from simulation order to real robot order.
        
        Args:
            target_pos: Numpy array of 6 joint positions in simulation order
        
        Returns:
            List of 6 joint positions in real robot order
        """
        # Create a dictionary for the real robot's joint ordering
        real_positions = {}
        
        # Map from simulation indices to joint names
        for name, sim_idx in self.joint_name_to_idx.items():
            real_positions[name] = target_pos[sim_idx]
            
        # Create a list ordered by self.joint_names for the trajectory message
        ordered_positions = [real_positions.get(name, 0.0) for name in self.joint_names]
        
        return ordered_positions
    
    def publish_commands(self, actions):
        """
        Publish actions as joint commands with velocity-based trajectory duration.
        
        Args:
            actions: Numpy array of 6 values representing arm actions
        """
        try:
            # Determine if we should convert based on control mode
            if self.direct_joint_control:
                # When in direct joint control mode, the actions are already joint targets
                # Skip the conversion and use them directly
                target_pos = actions.copy()  # Make a copy to avoid modifying the original
                
                # Clip the targets to joint limits for safety
                for i in range(len(target_pos)):
                    A, B = self.servo_angle_limits[i]
                    
                    # Store pre-clip value for logging
                    pre_clip = target_pos[i]
                    
                    # Apply clipping respecting both joint limits and config clip value
                    target_pos[i] = np.clip(target_pos[i], 
                                          max(A, -self.clip_value), 
                                          min(B, self.clip_value))
                    
                    # Log if clipping occurred
                    if pre_clip != target_pos[i]:
                        self.get_logger().warn(f'Joint {i} clipped from {pre_clip:.4f} to {target_pos[i]:.4f}')
                
                self.get_logger().info(f"Using raw joint targets from policy: {target_pos}")
            else:
                # Convert policy actions to real robot joint angles
                target_pos = self.convert_sim_to_real_angles(actions)
            
            # Create trajectory message
            traj_msg = JointTrajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.joint_names = self.joint_names
            
            # Create trajectory point with moving average for smoother motion
            point = JointTrajectoryPoint()
            
            # Remap joint positions to real robot order for trajectory message
            real_positions = self.remap_joint_positions_to_real_robot(target_pos)
            
            # Apply moving average if current positions are available
            if self.current_joint_positions is not None:
                # Map current joints to real robot order
                current_real = self.remap_joint_positions_to_real_robot(self.current_joint_positions)
                
                # Calculate smoothed positions
                positions = []
                total_distance = 0.0  # Track total distance for reporting
                for i, name in enumerate(self.joint_names):
                    # Apply moving average for smoother motion
                    current_pos = current_real[i]
                    target = real_positions[i]
                    
                    # Calculate distance for this joint (for logging)
                    distance = abs(target - current_pos)
                    total_distance += distance
                    
                    # For shoulder_pan_joint (first joint), add extra logging
                    if i == 0:
                        self.get_logger().info(f"Shoulder pan: current={current_pos:.4f}, target={target:.4f}, " +
                                              f"diff={distance:.4f}, moving_avg={self.moving_average:.2f}")
                    
                    # Blend target and current - higher moving_average means more aggressive movement
                    cmd_pos = current_pos * (1 - self.moving_average) + target * self.moving_average
                    positions.append(cmd_pos)
                
                # Log total movement distance
                self.get_logger().info(f"Total movement distance: {total_distance:.4f} radians")
            else:
                positions = real_positions
                self.get_logger().warn("No current joint positions available, using direct target positions")
            
            # Calculate durations based on movement distance and max velocity
            durations = []
            
            # For each joint in the real robot joint order
            for i in range(len(positions)):
                if self.current_joint_positions is not None:
                    # Get current position in real robot order
                    current_real = self.remap_joint_positions_to_real_robot(self.current_joint_positions)
                    current_pos = current_real[i]
                    
                    # Calculate duration based on distance and max velocity
                    distance = abs(positions[i] - current_pos)
                    duration = distance / self.max_velocity
                    durations.append(max(duration, self.min_trajectory_duration))
                else:
                    durations.append(self.trajectory_duration)
            
            # Set positions
            point.positions = positions
            
            # Set the time to reach the point (based on the max duration)
            max_duration = max(durations)
            # Ensure reasonable minimum duration
            max_duration = max(max_duration, self.trajectory_duration)
            
            # Convert to seconds and nanoseconds
            sec = int(max_duration)
            nanosec = int((max_duration - sec) * 1e9)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nanosec
            
            # Set velocities and accelerations
            point.velocities = [0.0] * len(positions)  # Zero velocity at target point
            point.accelerations = [0.0] * len(positions)  # Zero acceleration at target point
            
            traj_msg.points.append(point)
            
            # Log the trajectory message
            self.get_logger().info(f'Publishing trajectory: {traj_msg.joint_names}')
            self.get_logger().info(f'Target positions: {positions}')
            self.get_logger().info(f'Duration: {max_duration:.4f} seconds ({sec}s {nanosec}ns)')
            
            # Publish the trajectory
            self.joint_cmd_pub.publish(traj_msg)
            self.get_logger().debug('Trajectory published successfully')
        except Exception as e:
            self.get_logger().error(f'Error in publish_commands: {str(e)}')
            self.get_logger().error(traceback.format_exc())

    def publish_trajectory(self, positions):
        """
        Create and publish a trajectory message.
        
        Args:
            positions: Numpy array of 6 joint target positions
        """
        try:
            # Create trajectory message
            trajectory = JointTrajectory()
            trajectory.joint_names = self.joint_names
            
            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = positions.tolist()
            
            # Set velocities and accelerations to zeros
            point.velocities = [0.0] * len(positions)
            point.accelerations = [0.0] * len(positions)
            
            # Set timing
            sec = int(self.trajectory_time_from_start)
            nanosec = int((self.trajectory_time_from_start - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            
            # Add point to trajectory
            trajectory.points.append(point)
            
            # Log the trajectory
            self.get_logger().info(f'Publishing trajectory: {trajectory.joint_names}')
            self.get_logger().info(f'Target positions: {positions}')
            
            # Use joint_cmd_pub instead of trajectory_pub
            self.joint_cmd_pub.publish(trajectory)
            self.get_logger().debug('Trajectory published successfully')
        except Exception as e:
            self.get_logger().error(f'Error in publish_trajectory: {str(e)}')
            self.get_logger().error(traceback.format_exc())

    def timer_callback(self):
        """Timer callback to enforce consistent command rate.
        
        This is only used to publish debug information at a consistent rate.
        The actual control happens in response to incoming messages.
        """
        # Simply log current state for debugging if needed
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