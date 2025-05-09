#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import random as rd
import math
from transforms3d.euler import euler2quat

class TargetPosePublisher(Node):
    def __init__(self):
        super().__init__('target_pose_publisher')
        
        # Track robot initialization state
        self.robot_ready = False
        
        # Declare parameters
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('publish_rate', 20.0)  # seconds
        
        # Declare position range parameters with specified ranges for UR3 workspace
        self.declare_parameter('pos_x', [0.35, 0.45])  # Table workspace range for x position
        self.declare_parameter('pos_y', [-0.2, 0.2])   # Table workspace range for y position
        self.declare_parameter('pos_z', [0.15, 0.35])   # Height above table for z position
        
        # Declare orientation range parameters with fixed roll and pitch for UR3
        self.declare_parameter('roll', [0.0, 0.0])        # Fixed at 0.0 for roll
        self.declare_parameter('pitch', [math.pi/2, math.pi/2])  # Fixed at π/2 (90°) for UR3
        self.declare_parameter('yaw', [-3.14, 3.14])      # Full rotation range for yaw
        
        # Get parameters
        self.frame_id = self.get_parameter('frame_id').value
        publish_rate = self.get_parameter('publish_rate').value
        
        # Get position range parameters
        self.pos_x = self.get_parameter('pos_x').value
        self.pos_y = self.get_parameter('pos_y').value
        self.pos_z = self.get_parameter('pos_z').value
        
        # Get orientation range parameters
        self.roll = self.get_parameter('roll').value
        self.pitch = self.get_parameter('pitch').value
        self.yaw = self.get_parameter('yaw').value
        
        # Subscribe to robot ready status
        self.robot_ready_sub = self.create_subscription(
            Bool,
            '/robot_ready',
            self.robot_ready_callback,
            10)
        self.get_logger().info('Subscribed to /robot_ready for initialization status')
        
        # Publishers
        self.target_pose_publisher_ = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.timer = self.create_timer(publish_rate, self.timer_callback)
        
        self.get_logger().info(f'Target pose publisher initialized with frame_id: {self.frame_id}')
        self.get_logger().info(f'Publishing new pose every {publish_rate} seconds (after robot is ready)')
        self.get_logger().info(f'Position ranges: x={self.pos_x}, y={self.pos_y}, z={self.pos_z}')
        self.get_logger().info(f'Orientation ranges: roll={self.roll}, pitch={self.pitch}, yaw={self.yaw}')
        self.get_logger().info('Waiting for robot to complete initialization...')
        
    def robot_ready_callback(self, msg):
        """Handle robot ready status messages from the controller."""
        ready = msg.data
        if ready and not self.robot_ready:
            self.robot_ready = True
            self.get_logger().info("Robot initialization complete - starting target pose generation")
        elif not ready and self.robot_ready:
            self.robot_ready = False
            self.get_logger().info("Robot not ready - pausing target pose generation")
        
    def publish_target_pose(self, pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose = pose
        self.target_pose_publisher_.publish(msg)
        
        # Log the pose details
        self.get_logger().info(
            f'Published target pose: position=[{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}], '
            f'orientation=[{pose.orientation.w:.3f}, {pose.orientation.x:.3f}, {pose.orientation.y:.3f}, {pose.orientation.z:.3f}]'
        )
    
    def generate_pose(self):
        # Generate a random pose within the specified parameter ranges
        pose = PoseStamped().pose
        
        # Generate random positions within the specified ranges
        pose.position.x = rd.uniform(self.pos_x[0], self.pos_x[1])
        pose.position.y = rd.uniform(self.pos_y[0], self.pos_y[1])
        pose.position.z = rd.uniform(self.pos_z[0], self.pos_z[1])
        
        # Generate random orientation angles within the specified ranges
        # Note: For fixed values (roll and pitch), we just use the first value
        roll = self.roll[0]  # Fixed roll value
        pitch = self.pitch[0]  # Fixed pitch value
        yaw = rd.uniform(self.yaw[0], self.yaw[1])  # Random yaw within range
        
        # Convert Euler angles to quaternion (using transforms3d)
        quat = euler2quat(roll, pitch, yaw, 'sxyz')
        pose.orientation.w = quat[0]
        pose.orientation.x = quat[1]
        pose.orientation.y = quat[2]
        pose.orientation.z = quat[3]
        
        return pose

    def timer_callback(self):
        # Only generate and publish poses when the robot is ready
        if not self.robot_ready:
            self.get_logger().debug('Waiting for robot initialization to complete before publishing poses')
            return
            
        pose = self.generate_pose()
        self.publish_target_pose(pose)

def main(args=None):
    rclpy.init(args=args)
    target_pose_publisher = TargetPosePublisher()
    rclpy.spin(target_pose_publisher)
    target_pose_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()