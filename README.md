# UR Sim-to-Real Reinforcement Learning

A ROS 2 package for transferring reinforcement learning policies from IsaacLab to Universal Robots (UR) hardware.

## Overview

This package implements a bridge between trained IsaacLab policies and real UR Robots. Beforehand, you need to train a policy using IsaacLab and save it in the models/ directory. 

## Dependencies



- ROS 2
- Python 3
- PyTorch
- [Universal_Robots_ROS2_GZ_Simulation](https://github.com/UniversalRobots/Universal_Robots_ROS2_GZ_Simulation) package installed in your ROS 2 workspace

You need to have a working CUDA device with at least version 11.8 to run this package. For pytorch CUDA support, you can follow the instructions [here](https://pytorch.org/get-started/locally/).

## Installation

1. Clone this repository to your ROS 2 workspace:
   ```
   cd ~/ros2_ws/src
   git clone https://github.com/kyavuzkurt/ur_sim_to_real_rl_games.git
   ```

2. Install dependencies:
   ```
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. Build the package:
   ```
   colcon build --packages-select ur_sim_to_real_rl_games
   ```

## Usage
Put your trained model from IsaacLab in the models/ directory.


### Simulation

To run in simulation mode:
```
ros2 launch ur_sim_to_real_rl_games launch_gazebo.launch.py
```

### Real Robot

To deploy on a real UR robot, while the ur_robot_driver is running, you can run the following command:
```
ros2 launch ur_sim_to_real_rl_games real_robot.launch.py
```

## Configuration

Configuration files are available in the `config/` directory:
- `simulation.yaml`: Settings for simulation environment
- `real_robot.yaml`: Settings for real robot deployment

## License

MIT Licenses

## Author

Kadir Yavuz Kurt (k.yavuzkurt1@gmail.com)
