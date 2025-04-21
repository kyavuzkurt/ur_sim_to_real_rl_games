#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32MultiArray
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
        
        # Declare parameters
        self.declare_parameter('action_scale', 1.0)
        self.declare_parameter('trajectory_duration', 0.5)
        self.declare_parameter('min_trajectory_duration', 0.2)
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('moving_average', 0.5)
        
        self.declare_parameter('command_topic', '/joint_trajectory_controller/joint_trajectory')  
        self.declare_parameter('state_topic', '/joint_states')
        self.declare_parameter('use_relative_commands', True)
        self.declare_parameter('trajectory_topic', '/scaled_joint_trajectory_controller/joint_trajectory')
        self.declare_parameter('command_rate', 5.0)
        self.declare_parameter('policy_output_topic', '/policy_output')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('scale_factor', 0.5)
        
        self.declare_parameter('direct_joint_control', False)
        
        self.declare_parameter('has_gripper', False)
        self.declare_parameter('gripper_command_topic', '/gripper_command')
        
        self.declare_parameter('clip_actions', False)
        self.declare_parameter('clip_value', 100.0)
        self.declare_parameter('clip_observations', 100.0)
        
        # Get parameters
        self.action_scale = self.get_parameter('action_scale').value
        self.trajectory_duration = self.get_parameter('trajectory_duration').value
        self.min_trajectory_duration = self.get_parameter('min_trajectory_duration').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.moving_average = self.get_parameter('moving_average').value
        
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
        self.last_action_time = self.get_clock().now()
        self.expected_action_dim = 7 if self.has_gripper else 6
        
        self.trajectory_time_from_start = 1.0 / self.command_rate
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
                    target_positions = np.array(self.current_joint_positions) + actions * self.scale_factor
                    self.get_logger().debug(f'Target positions (relative): {target_positions}')
                else:
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
    
    def publish_commands(self, actions):
        """Publish actions as joint commands with velocity-based trajectory duration."""
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
            
            traj_msg = JointTrajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            
            real_positions = self.remap_joint_positions_to_real_robot(target_pos)
            
            if self.current_joint_positions is not None:
                current_real = self.remap_joint_positions_to_real_robot(self.current_joint_positions)
                
                positions = []
                total_distance = 0.0
                for i, name in enumerate(self.joint_names):
                    current_pos = current_real[i]
                    target = real_positions[i]
                    
                    distance = abs(target - current_pos)
                    total_distance += distance
                    
                    if i == 0:
                        self.get_logger().info(f"Shoulder pan: current={current_pos:.4f}, target={target:.4f}, " +
                                              f"diff={distance:.4f}, moving_avg={self.moving_average:.2f}")
                    
                    cmd_pos = current_pos * (1 - self.moving_average) + target * self.moving_average
                    positions.append(cmd_pos)
                
                self.get_logger().info(f"Total movement distance: {total_distance:.4f} radians")
            else:
                positions = real_positions
                self.get_logger().warn("No current joint positions available, using direct target positions")
            
            durations = []
            
            for i in range(len(positions)):
                if self.current_joint_positions is not None:
                    current_real = self.remap_joint_positions_to_real_robot(self.current_joint_positions)
                    current_pos = current_real[i]
                    
                    distance = abs(positions[i] - current_pos)
                    duration = distance / self.max_velocity
                    durations.append(max(duration, self.min_trajectory_duration))
                else:
                    durations.append(self.trajectory_duration)
            
            point.positions = positions
            
            max_duration = max(durations)
            max_duration = max(max_duration, self.trajectory_duration)
            
            sec = int(max_duration)
            nanosec = int((max_duration - sec) * 1e9)
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = nanosec
            
            point.velocities = [0.0] * len(positions)
            point.accelerations = [0.0] * len(positions)
            
            traj_msg.points.append(point)
            
            self.get_logger().info(f'Publishing trajectory: {traj_msg.joint_names}')
            self.get_logger().info(f'Target positions: {positions}')
            self.get_logger().info(f'Duration: {max_duration:.4f} seconds ({sec}s {nanosec}ns)')
            
            self.joint_cmd_pub.publish(traj_msg)
            self.get_logger().debug('Trajectory published successfully')
        except Exception as e:
            self.get_logger().error(f'Error in publish_commands: {str(e)}')
            self.get_logger().error(traceback.format_exc())

    def publish_trajectory(self, positions):
        """Create and publish a trajectory message."""
        try:
            trajectory = JointTrajectory()
            trajectory.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            point.positions = positions.tolist()
            
            point.velocities = [0.0] * len(positions)
            point.accelerations = [0.0] * len(positions)
            
            sec = int(self.trajectory_time_from_start)
            nanosec = int((self.trajectory_time_from_start - sec) * 1e9)
            point.time_from_start = Duration(sec=sec, nanosec=nanosec)
            
            trajectory.points.append(point)
            
            self.get_logger().info(f'Publishing trajectory: {trajectory.joint_names}')
            self.get_logger().info(f'Target positions: {positions}')
            
            self.joint_cmd_pub.publish(trajectory)
            self.get_logger().debug('Trajectory published successfully')
        except Exception as e:
            self.get_logger().error(f'Error in publish_trajectory: {str(e)}')
            self.get_logger().error(traceback.format_exc())

    def timer_callback(self):
        """Timer callback to enforce consistent command rate."""
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