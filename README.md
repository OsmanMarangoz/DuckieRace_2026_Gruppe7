# DuckieRace

Student project at **HKA — Karlsruhe University of Applied Sciences**, programme *Robotics and AI in Production*.

Autonomous driving project for the Duckiebot platform: lane following with a PID controller, red-line stopping, and AprilTag-based decision making at intersections.

**Students:** Osman, Bernhard, Pascal

## Setup

### Virtual machine with ROS Noetic

1. Download the Ubuntu 20.04 image from <https://www.releases.ubuntu.com/focal/>.
2. Create a virtual machine with the image. Set the network adapter to **bridged** so the VM shares the network with the host (required for ROS multi-machine communication with the Duckiebot).
3. Install ROS Noetic following <https://wiki.ros.org/noetic/Installation/Ubuntu>.
4. Clone this repository and point the remote at your own fork:

```bash
   git clone https://github.com/DuckieBotIRAS/DuckieRace_2026.git
   git remote set-url origin <your-github-repository-url>
```

### Devcontainer (alternative)

The repository includes a `.devcontainer/` setup for VS Code. Open the project in VS Code with the Remote Containers extension installed and select **Reopen in Container** — Docker handles the ROS Noetic environment, dependencies, and the `apriltag` library build.

## Build and run

Source ROS and configure the environment for your Duckiebot:

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://<duckiebot-name>.local:11311
export ROS_IP=<your-host-ip>
export VEHICLE_NAME=<duckiebot-name>
```

Build the catkin workspace:

```bash
catkin_make
source devel/setup.bash
```

Run a single node:

```bash
rosrun follow_lane detect_lane_node.py
```

Run the full pipeline via the launch file:

```bash
roslaunch follow_lane follow_lane.launch
```

## Architecture

Five ROS nodes form the autonomous driving pipeline:

- **`detect_lane_node`** — processes the camera feed, builds HSV masks for white/yellow/red, computes lane-center error, and reports red-line pixel count.
- **`detect_apriltag_node`** — runs the AprilTag detector on the camera feed and publishes the last detected tag ID (with timeout).
- **`switch_control_node`** — high-level state machine. Switches between `Lane` and `Obstacle` mode based on red-line detection; waits for a `done` signal from `decision_node` to resume.
- **`control_lane_node`** — PID controller that turns the lane-center error into velocity/omega commands. Only publishes while in Lane mode.
- **`decision_node`** — owns the obstacle phase: stops the bot, executes an action chosen by the last AprilTag ID, then signals completion.

## Live tuning

Most parameters (HSV thresholds, PID gains, crop polygon, AprilTag timeout) are exposed via JSON config files in `src/packages/follow_lane/config/`. Run the configuration GUI for live tuning with sliders:

```bash
rosrun follow_lane configuration_node.py
```

## Debug visualization

Debug image topics follow the `image_transport` convention (`.../compressed` suffix) and can be viewed in `rqt`:

```bash
rqt
```

Add multiple `Image View` plugins and subscribe to:

- `/<vehicle>/debug/lane_croped`
- `/<vehicle>/debug/lane_white`
- `/<vehicle>/debug/lane_yellow`
- `/<vehicle>/debug/lane_red`
- `/<vehicle>/apriltag/debug`

## Code structure

This repository is a catkin workspace.

- `src/packages/follow_lane/` — the autonomous-driving package (nodes, configs, launch files).
- `src/packages/duckietown_msgs/` — message and service definitions for communication with the Duckiebot.
- `.devcontainer/` — Docker-based development environment.
- `launchers/` — convenience shell scripts.