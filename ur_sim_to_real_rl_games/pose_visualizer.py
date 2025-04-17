#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import math

class PoseVisualizer(Node):
    def __init__(self):
        super().__init__('pose_visualizer')
        
        # Subscription to target pose
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.pose_callback,
            10
        )
        
        # Publisher for visualization markers
        self.marker_pub = self.create_publisher(
            Marker,
            '/visualization_marker',
            10
        )
        
        self.get_logger().info('Pose visualizer initialized')
        
    def pose_callback(self, msg):
        # Create a marker for the target pose
        marker = Marker()
        marker.header = msg.header
        
        # If no frame_id is specified, use the base frame of the robot
        if not marker.header.frame_id:
            marker.header.frame_id = "base_link"
            
        marker.ns = "target_poses"
        marker.id = 0
        marker.type = Marker.ARROW  # Using an arrow to show position and orientation
        marker.action = Marker.ADD
        
        # Set the pose from the message
        marker.pose = msg.pose
        
        # Set the scale - this defines the size of the arrow
        marker.scale.x = 0.1  # Shaft length
        marker.scale.y = 0.02  # Shaft diameter
        marker.scale.z = 0.02  # Head diameter
        
        # Set the color (red, semi-transparent)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # Alpha (transparency)
        
        # Set the lifetime (0 = forever)
        marker.lifetime.sec = 0
        
        # Publish the marker
        self.marker_pub.publish(marker)
        self.get_logger().debug(f'Published marker for pose at: [{msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}]')
        
        # Also create a coordinate frame marker to better visualize orientation
        self.publish_axis_marker(msg)
    
    def publish_axis_marker(self, pose_msg):
        """Publishes a coordinate frame marker to show XYZ axes at the target pose"""
        
        # Create markers for X, Y, Z axes
        for i, (axis, color) in enumerate(zip(['x', 'y', 'z'], [(1,0,0), (0,1,0), (0,0,1)])):
            marker = Marker()
            marker.header = pose_msg.header
            
            # If no frame_id is specified, use the base frame of the robot
            if not marker.header.frame_id:
                marker.header.frame_id = "base_link"
                
            marker.ns = f"target_axes"
            marker.id = i + 1  # Different ID for each axis
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Start from the target pose
            marker.pose = pose_msg.pose
            
            # Set the scale - this defines the size of the arrow
            marker.scale.x = 0.1  # Length
            marker.scale.y = 0.01  # Width
            marker.scale.z = 0.01  # Height
            
            # Set the color based on axis
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = 1.0
            
            # Set the lifetime (0 = forever)
            marker.lifetime.sec = 0
            
            # Publish the marker
            self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = PoseVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main() 