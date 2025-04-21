#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import torch
import numpy as np
import torch.nn as nn
import os
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_pose
from transforms3d.euler import quat2euler  

# Define a custom actor-critic model that matches RL-Games architecture
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, has_gripper=False):
        super(ActorCritic, self).__init__()
        
        # Store if we have a gripper for proper action handling
        self.has_gripper = has_gripper
        
        # Shared layers (same as in RL-Games config: units=[64, 64], activation=elu)
        # This matches network.mlp.units: [64, 64] and activation: elu
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU()
        )
        
        # Actor mean (mu) branch - no activation as per mu_activation: None
        self.mu = nn.Linear(64, action_dim)
        
        # Fixed sigma as per fixed_sigma: True
        # Initialized to zeros as per sigma_init.name: const_initializer, val: 0
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value function branch
        self.value = nn.Linear(64, 1)
    
    def forward(self, obs):
        shared_features = self.shared_net(obs)
        
        # Actor outputs - no activation applied (mu_activation: None)
        mu = self.mu(shared_features)
        
        # Value output
        value = self.value(shared_features)
        
        return mu, value, {'log_std': self.log_std}

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        
        # Setup device for inference
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Log device information
        if self.device.type == 'cuda':
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            self.get_logger().info(f"Using GPU: {device_name} (Device {current_device}, Total GPUs: {gpu_count})")
            
            # Log CUDA memory information
            try:
                allocated = torch.cuda.memory_allocated(current_device) / (1024 ** 3)  # Convert to GB
                reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 3)    # Convert to GB
                self.get_logger().info(f"CUDA Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            except Exception as e:
                self.get_logger().warn(f"Could not get CUDA memory info: {e}")
        else:
            self.get_logger().warn("GPU not available, running on CPU")
        
        # Joint name mapping and indices
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
        
        # Declare parameters
        self.declare_parameter('model_path', 'reach_ur10.pth')
        self.declare_parameter('clip_actions', False)
        self.declare_parameter('clip_value', 100.0)
        self.declare_parameter('clip_observations', 100.0)
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('state_topic', '/joint_states')
        self.declare_parameter('normalize_observations', True)
        self.declare_parameter('invert_target_pose', False)
        self.declare_parameter('use_tf_transform', False)
        self.declare_parameter('direct_joint_control', False)
        self.declare_parameter('has_gripper', False)
        self.declare_parameter('obs_dim', 25)
        self.declare_parameter('command_rate', 30.0)  
        
        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.clip_actions = self.get_parameter('clip_actions').value
        self.clip_value = self.get_parameter('clip_value').value
        self.clip_observations = self.get_parameter('clip_observations').value
        self.target_frame = self.get_parameter('target_frame').value
        state_topic = self.get_parameter('state_topic').value
        self.normalize_observations = self.get_parameter('normalize_observations').value
        self.invert_target_pose = self.get_parameter('invert_target_pose').value
        self.use_tf_transform = self.get_parameter('use_tf_transform').value
        self.direct_joint_control = self.get_parameter('direct_joint_control').value
        self.has_gripper = self.get_parameter('has_gripper').value
        self.obs_dim = self.get_parameter('obs_dim').value
        self.command_rate = self.get_parameter('command_rate').value
        
        # Set up TF buffer and listener if needed
        if self.use_tf_transform:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.get_logger().info("TF transformation enabled")
        
        # Check if model exists
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the checkpoint
        self.get_logger().info(f"Loading checkpoint: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.get_logger().info(f"Successfully loaded checkpoint: {list(checkpoint.keys())}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise
        
        try:
            # Calculate and verify expected dimensions
            self.verify_dimensions()
            
            # Use parameter for observation dimension
            self.get_logger().info(f"Using observation dimension: {self.obs_dim}")
            
            # Action dimension is 6 for UR robots (6 joints) + optional gripper
            self.action_dim = 7 if self.has_gripper else 6
            self.get_logger().info(f"Using action dimension: {self.action_dim}")
            
            # Create our custom actor-critic model with the correct dimensions
            self.model = ActorCritic(self.obs_dim, self.action_dim, has_gripper=self.has_gripper)
            
            # Explicitly move model to the specified device
            self.model = self.model.to(self.device)
            self.get_logger().info(f"Model moved to device: {self.device}")
            
            # Extract and load weights from checkpoint
            self.load_weights_from_checkpoint(checkpoint)
            
            # Set model to evaluation mode
            self.model.eval()
            self.get_logger().info("Model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize model: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise
        
        # Create subscriptions
        self.joint_state_sub = self.create_subscription(
            JointState, 
            state_topic, 
            self.joint_state_callback, 
            10)
        self.get_logger().info(f'Subscribed to {state_topic}')
        
        self.target_pose_sub = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.target_pose_callback,
            10)
        self.get_logger().info('Subscribed to /target_pose')
        
        # Create publisher for policy outputs
        self.policy_output_pub = self.create_publisher(
            Float32MultiArray,
            '/policy_output',
            10)
        self.get_logger().info('Publishing to /policy_output')
        
        # Create publisher for gripper control if applicable
        if self.has_gripper:
            self.gripper_pub = self.create_publisher(
                Float32MultiArray,
                '/gripper_command',
                10)
            self.get_logger().info('Publishing to /gripper_command')
        
        # Debug publisher for transformed target pose
        self.transformed_pose_pub = self.create_publisher(
            PoseStamped, 
            '/transformed_target_pose', 
            10)
        
        # Debug publisher for observation vectors
        self.observation_pub = self.create_publisher(
            Float32MultiArray,
            '/policy_observation',
            10)
        
        # State variables
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.target_pose = None
        self.current_joint_dict = {}  # Store joint states by name
        self.last_action_time = self.get_clock().now()
        self.command_counter = 0
        self.previous_actions = np.zeros(self.action_dim)  # Store previous actions for observation
        
        # Create a timer to enforce command rate
        self.timer = self.create_timer(1.0 / self.command_rate, self.timer_callback)
        self.get_logger().info(f'Created timer at {self.command_rate} Hz')
        
        # Observation normalization values (realistic ranges)
        # For joint positions: [-π, π] -> [-1, 1]
        # For joint velocities: [-3, 3] -> [-1, 1]
        # For target positions: [-1.0, 1.0] -> [-1, 1]
        self.joint_pos_scale = np.pi
        self.joint_vel_scale = 3.0
        self.target_pos_scale = 1.0
        
        self.get_logger().info('Policy node initialized')
        self.get_logger().info(f'Target frame: {self.target_frame}')
        self.get_logger().info(f'Normalizing observations: {self.normalize_observations}')
        self.get_logger().info(f'Inverting target pose: {self.invert_target_pose}')
        self.get_logger().info(f'Using TF transform: {self.use_tf_transform}')
        self.get_logger().info(f'Direct joint control mode: {self.direct_joint_control}')
        self.get_logger().info(f'Has gripper: {self.has_gripper}')
        self.get_logger().info(f'Observation dimension: {self.obs_dim}')
        self.get_logger().info(f'Action dimension: {self.action_dim}')
        self.get_logger().info(f'Command rate: {self.command_rate} Hz')
    
    def verify_dimensions(self):
        """
        Verify that the specified dimensions match the expected neural network structure.
        
        The expected structure is:
        - Input: 24-25 elements (6 joint pos + 6 joint vel + 6 pose command + 6-7 prev actions)
        - Output: 6-7 elements (6 arm actions + optional gripper action)
        """
        # Calculate expected observation dimension
        expected_obs_dim = 6 + 6 + 6 + (7 if self.has_gripper else 6)  # 24 or 25
        
        # Check if obs_dim parameter matches expected
        if self.obs_dim != expected_obs_dim:
            self.get_logger().warn(
                f"Observation dimension parameter ({self.obs_dim}) doesn't match expected ({expected_obs_dim}). "
                f"This may cause issues if the model was trained with a different dimension."
            )
        
        # Calculate expected action dimension
        expected_action_dim = 7 if self.has_gripper else 6
        
        # Log the verified dimensions
        self.get_logger().info(f"Verified dimensions - Input: {expected_obs_dim}, Output: {expected_action_dim}")
        
        return expected_obs_dim, expected_action_dim
    
    def load_weights_from_checkpoint(self, checkpoint):
        """Load weights from the checkpoint into our custom model."""
        # Extract model state dict from checkpoint
        rl_games_state_dict = checkpoint['model']
        
        # Create a new state dict that matches our custom model's structure
        new_state_dict = {}
        
        # Print the keys in the RL-Games state dict for debugging
        self.get_logger().info(f"RL-Games model keys: {list(rl_games_state_dict.keys())}")
        
        # Map the parameter names from RL-Games format to our custom model
        key_mapping = {
            'a2c_network.actor_mlp.0.weight': 'shared_net.0.weight',
            'a2c_network.actor_mlp.0.bias': 'shared_net.0.bias',
            'a2c_network.actor_mlp.2.weight': 'shared_net.2.weight',
            'a2c_network.actor_mlp.2.bias': 'shared_net.2.bias',
            'a2c_network.mu.weight': 'mu.weight',
            'a2c_network.mu.bias': 'mu.bias',
            'a2c_network.value.weight': 'value.weight',
            'a2c_network.value.bias': 'value.bias',
            'a2c_network.sigma': 'log_std',
        }
        
        # Transfer weights according to the mapping
        for rl_key, custom_key in key_mapping.items():
            if rl_key in rl_games_state_dict:
                # Move tensor to the correct device as we load it
                new_state_dict[custom_key] = rl_games_state_dict[rl_key].to(self.device)
                self.get_logger().debug(f"Mapped {rl_key} -> {custom_key} with shape {rl_games_state_dict[rl_key].shape}")
            else:
                # If a direct match is not found, try to find a close match
                possible_matches = [k for k in rl_games_state_dict.keys() if rl_key.split('.')[-1] in k]
                if possible_matches:
                    self.get_logger().warn(f"Using approximate match for {rl_key}: {possible_matches[0]}")
                    new_state_dict[custom_key] = rl_games_state_dict[possible_matches[0]].to(self.device)
                else:
                    self.get_logger().warn(f"No match found for {rl_key}")
        
        # Load the mapped weights into our model
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        
        # Log any issues with loading
        if missing:
            self.get_logger().warn(f"Missing keys: {missing}")
        if unexpected:
            self.get_logger().warn(f"Unexpected keys: {unexpected}")
            
        # Log device information for confirmation
        for name, param in self.model.named_parameters():
            self.get_logger().debug(f"Parameter {name} on device: {param.device}")
    
    def timer_callback(self):
        """Timer callback to run the policy at a consistent rate."""
        # Only run policy if we have the necessary data
        if self.current_joint_positions is not None and self.target_pose is not None:
            self.run_policy()
        else:
            self.get_logger().debug('Timer tick - missing data (joint positions or target pose)')
    
    def joint_state_callback(self, msg):
        """Process joint state messages and update current joint positions."""
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
            
            # Note: No longer calling run_policy directly here - timer will handle it
        else:
            self.get_logger().warn(f'Incomplete joint states received: {msg.name}')
    
    def target_pose_callback(self, msg):
        """Process target pose messages."""
        # Save original pose for logging
        original_pose = msg
        transformed_pose = None
        
        # Handle frame transformation if needed
        if self.use_tf_transform and msg.header.frame_id != self.target_frame:
            try:
                # Look up transform and apply it
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    msg.header.frame_id,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=1.0))
                
                transformed_pose = do_transform_pose(msg, transform)
                transformed_pose.header.frame_id = self.target_frame
                
                self.get_logger().info(f'Transformed pose from {msg.header.frame_id} to {self.target_frame}')
                self.target_pose = transformed_pose
            except TransformException as ex:
                self.get_logger().error(f'Failed to transform pose: {ex}')
                # Fall back to original pose
                self.target_pose = msg
        else:
            # Just check the frame
            if msg.header.frame_id and msg.header.frame_id != self.target_frame:
                self.get_logger().warn(f"Received pose in frame '{msg.header.frame_id}', but expected '{self.target_frame}'")
            
            self.target_pose = msg
        
        # Apply any coordinate inversion if configured
        if self.invert_target_pose:
            # Create a copy to avoid modifying the original
            inverted_pose = PoseStamped()
            inverted_pose.header = self.target_pose.header
            # Invert coordinates - may need to adjust this based on how Isaac sim defines coordinates
            inverted_pose.pose.position.x = -self.target_pose.pose.position.x
            inverted_pose.pose.position.y = -self.target_pose.pose.position.y
            inverted_pose.pose.position.z = self.target_pose.pose.position.z  # Usually height stays the same
            
            # For orientation, we need to adjust the quaternion for inverted coords
            # This is a simple inversion and might need more sophisticated transformation
            inverted_pose.pose.orientation.w = self.target_pose.pose.orientation.w
            inverted_pose.pose.orientation.x = -self.target_pose.pose.orientation.x
            inverted_pose.pose.orientation.y = -self.target_pose.pose.orientation.y
            inverted_pose.pose.orientation.z = self.target_pose.pose.orientation.z
            
            self.target_pose = inverted_pose
            self.get_logger().info("Applied coordinate inversion to target pose")
        
        # Log the final target pose being used
        self.get_logger().info(f'Using target pose in frame {self.target_pose.header.frame_id}: pos=[{self.target_pose.pose.position.x:.3f}, {self.target_pose.pose.position.y:.3f}, {self.target_pose.pose.position.z:.3f}], '
                              f'orient=[{self.target_pose.pose.orientation.w:.3f}, {self.target_pose.pose.orientation.x:.3f}, {self.target_pose.pose.orientation.y:.3f}, {self.target_pose.pose.orientation.z:.3f}]')
        
        # Publish the transformed pose for debugging
        self.transformed_pose_pub.publish(self.target_pose)
        
        # Note: No longer calling run_policy directly here - timer will handle it
        self.get_logger().debug('Target pose received, will be used in next timer cycle')
    
    def run_policy(self):
        """
        Run the policy model and publish the actions.
        
        This method:
        1. Prepares the observation vector (24-25 elements)
        2. Runs the model to get actions (6-7 elements)
        3. Publishes arm actions (6 elements) to the controller
        4. Optionally publishes gripper action (1 element) if gripper is present
        """
        try:
            # Skip if we don't have the necessary data
            if self.current_joint_positions is None:
                self.get_logger().warn("Missing joint positions, cannot run policy")
                return
                
            if self.target_pose is None:
                self.get_logger().warn("Missing target pose, cannot run policy")
                return
                
            # Prepare observation vector that matches the expected input size of the model
            obs = self.prepare_observation()
            
            # Publish observation for debugging
            self.publish_observation(obs)
            
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                self.get_logger().debug('Running model inference...')
                # Get model output (actions mean, value, extras)
                mu, value, extras = self.model(obs_tensor)
                
                # Convert to numpy array
                action_np = mu.cpu().numpy()[0]
                value_np = value.cpu().numpy()[0]
                
                # Log full information about the policy output
                self.get_logger().info(f'Raw model output (action): {action_np}')
                self.get_logger().info(f'Value estimate: {value_np}')
                
                # Optionally log additional info like variance
                if 'log_std' in extras:
                    log_std = extras['log_std'].cpu().numpy()
                    self.get_logger().debug(f'Action log std: {log_std}')
            
            # Apply clipping if enabled (matches config.clip_actions)
            if self.clip_actions:
                orig_action = action_np.copy()
                action_np = np.clip(action_np, -self.clip_value, self.clip_value)
                if not np.array_equal(orig_action, action_np):
                    self.get_logger().debug(f'Actions clipped from {orig_action} to {action_np}')
            
            # Verify action dimension
            expected_action_dim = 7 if self.has_gripper else 6
            if len(action_np) != expected_action_dim:
                self.get_logger().warn(
                    f"Action dimension mismatch: got {len(action_np)}, expected {expected_action_dim}"
                )
                # Handle dimension mismatch
                if len(action_np) < expected_action_dim:
                    # Pad with zeros
                    pad_count = expected_action_dim - len(action_np)
                    self.get_logger().warn(f"Padding action with {pad_count} zeros")
                    action_np = np.append(action_np, np.zeros(pad_count))
                elif len(action_np) > expected_action_dim:
                    # Truncate
                    self.get_logger().warn(f"Truncating action from {len(action_np)} to {expected_action_dim}")
                    action_np = action_np[:expected_action_dim]
            
            # Store current actions as previous actions for next observation
            self.previous_actions = action_np.copy()
            
            # Handle arm and optional gripper actions
            if self.has_gripper:
                # Split into arm actions and gripper action
                arm_actions = action_np[:6]
                gripper_action = action_np[6:7]
                
                # Log the arm and gripper actions
                self.get_logger().info(f'Arm actions: {arm_actions}')
                self.get_logger().info(f'Gripper action: {gripper_action}')
                
                # Publish arm actions
                self.publish_policy_output(arm_actions)
                
                # Publish gripper action separately
                gripper_msg = Float32MultiArray()
                gripper_msg.layout.dim.append(MultiArrayDimension())
                gripper_msg.layout.dim[0].label = "gripper"
                gripper_msg.layout.dim[0].size = len(gripper_action)
                gripper_msg.layout.dim[0].stride = 1
                gripper_msg.data = gripper_action.tolist()
                self.gripper_pub.publish(gripper_msg)
                self.get_logger().debug(f'Published gripper command: {gripper_action}')
            else:
                # Use all actions for arm control (should be 6 values)
                arm_actions = action_np
                
                # If in direct joint control mode, note that the output is already in the joint angle space
                if self.direct_joint_control:
                    self.get_logger().info("Using direct joint control mode - policy outputs are direct joint targets")
                    
                    # If we're using normalized observations, we need to properly interpret the outputs
                    if self.normalize_observations:
                        # Check if the outputs are within a ±π range, which suggests they might be
                        # normalized to match the input normalization
                        if np.all(np.abs(arm_actions) <= self.clip_value):
                            self.get_logger().debug("Actions appear to be in expected range for joint targets")
                        else:
                            self.get_logger().warn("Actions out of typical joint angle range, they may need scaling")
                
                # Publish the policy output for the arm
                self.publish_policy_output(arm_actions)
            
        except Exception as e:
            self.get_logger().error(f'Error in run_policy: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def prepare_observation(self):
        """
        Prepare observation vector that matches the expected dimension for the model.
        
        Observation structure (24-25 elements):
        - 6 joint positions (one per DoF)
        - 6 joint velocities (one per DoF)
        - 6 pose command values:
            - 3 position (x,y,z)
            - 3 orientation (roll, pitch, yaw)
        - 6-7 previous actions (arm + optional gripper if present)
        
        Returns:
            Numpy array with observation vector
        """
        # Create the observation vector
        obs = []
        
        # 1. Joint positions (6 values, one per DoF)
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint positions available, using zeros")
            joint_pos = np.zeros(6)
        elif len(self.current_joint_positions) > 6:
            joint_pos = np.array(self.current_joint_positions[:6])
        else:
            joint_pos = np.array(self.current_joint_positions.copy())
            
        # Normalize joint positions if needed
        if self.normalize_observations:
            joint_pos = joint_pos / self.joint_pos_scale  # Map from [-π, π] to [-1, 1]
            # Clip values according to configuration
            joint_pos = np.clip(joint_pos, -self.clip_observations, self.clip_observations)
            
        self.get_logger().debug(f"Joint positions: {joint_pos}")
        obs.extend(joint_pos)
            
        # 2. Joint velocities (6 values, one per DoF)
        if self.current_joint_velocities is None:
            self.get_logger().warn("No joint velocities available, using zeros")
            joint_vel = np.zeros(6)
        elif len(self.current_joint_velocities) > 6:
            joint_vel = np.array(self.current_joint_velocities[:6])
        else:
            joint_vel = np.array(self.current_joint_velocities.copy())
            
        # Normalize joint velocities if needed
        if self.normalize_observations:
            joint_vel = joint_vel / self.joint_vel_scale  # Map to [-1, 1]
            # Clip values according to configuration
            joint_vel = np.clip(joint_vel, -self.clip_observations, self.clip_observations)
            
        self.get_logger().debug(f"Joint velocities: {joint_vel}")
        obs.extend(joint_vel)
            
        # 3. Target position (3 values for x, y, z)
        if self.target_pose is None:
            self.get_logger().warn("No target pose available, using zeros")
            target_pos = np.zeros(3)
        else:
            target_pos = np.array([
                self.target_pose.pose.position.x,
                self.target_pose.pose.position.y,
                self.target_pose.pose.position.z
            ])
        
        # Normalize target positions if needed
        if self.normalize_observations:
            target_pos = target_pos / self.target_pos_scale  # Map to [-1, 1]
            # Clip values according to configuration
            target_pos = np.clip(target_pos, -self.clip_observations, self.clip_observations)
            
        self.get_logger().debug(f"Target position: {target_pos}")
        obs.extend(target_pos)
        
        # 4. Target orientation (3 values for roll, pitch, yaw)
        if self.target_pose is None:
            self.get_logger().warn("No target orientation available, using zeros")
            euler_angles = np.zeros(3)
        else:
            # Extract quaternion (x, y, z, w)
            quat = [
                self.target_pose.pose.orientation.x,
                self.target_pose.pose.orientation.y,
                self.target_pose.pose.orientation.z,
                self.target_pose.pose.orientation.w
            ]
            
            # Convert quaternion to Euler angles (roll, pitch, yaw)
            euler_angles = np.array(quat2euler(quat))
            
            # Normalize Euler angles if needed
            if self.normalize_observations:
                # Map Euler angles from [-π, π] to [-1, 1]
                euler_angles = euler_angles / np.pi
                # Clip values according to configuration
                euler_angles = np.clip(euler_angles, -self.clip_observations, self.clip_observations)
            
        self.get_logger().debug(f"Target orientation (euler): {euler_angles}")
        obs.extend(euler_angles)
        
        # 5. Previous actions (6-7 values)
        # Use stored previous actions, which already have the correct dimension
        prev_actions = self.previous_actions.copy()
        
        # Ensure correct dimensionality (6 arm actions + optional gripper)
        if self.has_gripper and len(prev_actions) < 7:
            # Add a zero for missing gripper action
            self.get_logger().warn("Previous actions missing gripper action, adding zero")
            prev_actions = np.append(prev_actions, 0.0)
        elif not self.has_gripper and len(prev_actions) > 6:
            # Trim extra actions
            self.get_logger().warn("Previous actions has extra values, trimming to 6")
            prev_actions = prev_actions[:6]
        
        # Previous actions should already be normalized and clipped by run_policy
        self.get_logger().debug(f"Previous actions: {prev_actions}")
        obs.extend(prev_actions)
        
        # 6. Ensure observation has correct dimension
        expected_obs_len = 6 + 6 + 3 + 3 + (7 if self.has_gripper else 6)  # 24 or 25
        
        if len(obs) != expected_obs_len:
            self.get_logger().warn(
                f"Observation dimension mismatch: got {len(obs)}, expected {expected_obs_len}. "
                f"Using requested obs_dim={self.obs_dim}"
            )
            
        if len(obs) > self.obs_dim:
            self.get_logger().warn(f"Truncating observation from {len(obs)} to {self.obs_dim} features")
            obs = obs[:self.obs_dim]
        elif len(obs) < self.obs_dim:
            pad_count = self.obs_dim - len(obs)
            self.get_logger().debug(f"Padding observation with {pad_count} zeros")
            obs.extend([0.0] * pad_count)
        
        # Log the observation dimensions
        self.get_logger().debug(f"Final observation dimension: {len(obs)}")
        
        return np.array(obs, dtype=np.float32)
    
    def publish_observation(self, obs_array):
        """Publish the observation vector for debugging."""
        msg = Float32MultiArray()
        
        # Set up layout
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "observation"
        msg.layout.dim[0].size = len(obs_array)
        msg.layout.dim[0].stride = 1
        
        # Set the data
        msg.data = obs_array.tolist()
        
        # Publish
        self.observation_pub.publish(msg)
    
    def publish_policy_output(self, action_np):
        """Publish the policy output as a Float32MultiArray message."""
        # Create the message
        msg = Float32MultiArray()
        
        # Set up layout (optional but good practice)
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "actions"
        msg.layout.dim[0].size = len(action_np)
        msg.layout.dim[0].stride = 1
        
        # Set the data
        msg.data = action_np.tolist()
        
        # Publish the message
        self.policy_output_pub.publish(msg)
        self.get_logger().debug(f'Published policy output: {action_np}')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = PolicyNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main() 