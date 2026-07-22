# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DuckieRace is an autonomous driving project for the Duckiebot platform at HKA — Karlsruhe University of Applied Sciences. The system implements lane following with PID control, red-line stopping at intersections, and AprilTag-based decision making.

**AprilTag System:**
- Tags 1-4: Intersection tags that define available directions at each junction
- Tags 5-14: Gate tags on roads between intersections (used for mapping)
- The city graph defines 3 intersections (A, B, C) with 5 roads/edges

## Build and Run Commands

```bash
# Build the catkin workspace
catkin_make
source devel/setup.bash

# Run the basic lane-following pipeline
roslaunch follow_lane follow_lane.launch

# Run the mapping phase (Phase A of Challenge 4)
roslaunch follow_lane mapping.launch start_node:=A start_exit_arm:=1

# Run a single node
rosrun follow_lane <node_name>.py

# Run tests (city_graph.py is fully testable without ROS)
python3 src/packages/follow_lane/src/test_city_graph.py

# Debug visualization
rqt
# Then add Image View plugins and subscribe to debug topics

# Live parameter tuning
rosrun follow_lane configuration_node.py
```

## Architecture

### Core Pipeline (follow_lane.launch)
The basic autonomous driving stack follows this flow:

1. **detect_lane_node** → Processes camera feed via HSV thresholding for white/yellow lane lines and red stop lines. Publishes:
   - `/<veh>/detect/lane` (Float64): Lane-center error (-1 to 1)
   - `/<veh>/detect/red_line` (Float64): Red pixel count at bottom of image
   - Debug topics: `/<veh>/debug/lane_croped`, `/<veh>/debug/lane_white`, etc.

2. **detect_apriltag_node** → AprilTag detection using the apriltag library. Publishes:
   - `/<veh>/apriltag/id` (Int32): Last detected tag ID (-1 = no tag)
   - `/<veh>/apriltag/detections` (String): JSON array of all detections with area for distance estimation

3. **switch_control_node** → High-level state machine with ControlType enum (Lane=1, Obstacle=2, Stop=3). Publishes `/<veh>/switch/control`. Listens for:
   - Red line detection → switches to Obstacle mode
   - Explorer halt signal → switches to Stop mode

4. **control_lane_node** → PID controller that turns lane error into velocity/omega commands. Only publishes when in Lane mode.

5. **decision_node** → Executes maneuvers at intersections. Publishes `/<veh>/decision/action` ("stopping", "turn_left", "turn_right", "move_forward", "skip"). When `direction_source=external`, reads from `/<veh>/explore/suggested_action`.

### Mapping Phase (mapping.launch)
Adds three nodes for autonomous exploration and mapping:

1. **localization_node** → Passively tracks position on the city graph. Publishes:
   - `/<veh>/mapping/pose` (String, JSON): Current position, visited edges, coverage stats
   - `/<veh>/mapping/complete` (Bool): Whether all edges have been visited
   - Listens to `/<veh>/decision/action` and `/<veh>/apriltag/id`

2. **explorer_node** → Calculates next turn recommendation for edge coverage. Publishes:
   - `/<veh>/explore/suggested_action` (String): Recommended maneuver
   - `/<veh>/explore/state` (String, JSON): Sweep progress, gate completion status
   - `/<veh>/explore/halt` (Bool): Signal to stop when mapping complete
   - Listens to `/<veh>/mapping/pose`

3. **gate_mapper_node** → Maps gate tags (5-14) to edges. Publishes:
   - `/<veh>/mapping/gates` (String, JSON): All discovered gate-to-edge mappings
   - `/<veh>/mapping/validated_gates` (String, JSON): Only gates matching expected_gates from city_graph.json

### City Graph System (city_graph.py)
**Purposefully written without ROS imports** — pure Python, testable offline.

Key concepts:
- **Graph Format**: `node -> { arm: (neighbor, neighbor_arm) }` — undirected, every road appears from both directions
- **Arm Convention**: Arms 1-4 numbered counterclockwise (settable via `CLOCKWISE_NUMBERING = False`)
- **Arm Math**: 
  - Forward = opposite arm (1↔3, 2↔4)
  - Left turn = identity (entry arm = exit arm)
  - Right turn = adjacent arm
- **Edge IDs**: Canonical format `"A1-B1"` (sorted alphabetically)

Key classes:
- `CityGraph`: Graph structure with BFS shortest-path and edge-coverage logic
- `GraphTracker`: Dead-reckoning on the graph — tracks current edge/node, status transitions WAITING_FIRST_STOP → ON_EDGE → AT_INTERSECTION → ON_EDGE
- `ExplorerPolicy`: Multi-sweep exploration (up to `max_sweeps` passes to catch gate tags missed due to ~90% detection rate)
- `ExpectedGatesMap`: Validates discovered gates against expected positions from city_graph.json

## Configuration Files

- `src/packages/follow_lane/config/city_graph.json` — City graph definition + expected_gates mapping
- `src/packages/follow_lane/config/detect_lane_node.json` — HSV thresholds, crop polygon for lane detection
- `src/packages/follow_lane/config/detect_apriltag_node.json` — Camera intrinsics, tag timeout
- `src/packages/follow_lane/config/control_lane_node.json` — PID gains, max velocity

## Key Files

| File | Purpose |
|------|---------|
| `city_graph.py` | Core graph logic, arm math, ExplorerPolicy, GraphTracker |
| `localization_node.py` | Position tracking using start_node/start_exit_arm parameters |
| `explorer_node.py` | Navigation decisions for edge coverage |
| `gate_mapper_node.py` | Maps AprilTag IDs to graph edges |
| `switch_control_node.py` | Control mode state machine |
| `decision_node.py` | Maneuver execution at intersections |
| `detect_lane_node.py` | HSV lane and red-line detection |
| `detect_apriltag_node.py` | AprilTag detection with distance via area |
| `test_city_graph.py` | Comprehensive simulation tests (run without ROS) |

## Important Constants

```python
GATE_TAG_IDS = set(range(5, 15))          # AprilTag IDs for road gates
INTERSECTION_TAG_IDS = set(range(1, 5))   # AprilTag IDs for intersections
ID_FUNCTIONS = {
    1: ['turn_left', 'turn_right', 'move_forward'],
    2: ['turn_left', 'turn_right'],
    3: ['turn_left', 'move_forward'],
    4: ['turn_right', 'move_forward'],
}
```

## Common Message Flows

**Lane Following**:
```
camera → detect_lane_node → /detect/lane → control_lane_node → /car_cmd_switch_node/cmd
```

**Intersection Handling**:
```
detect_lane_node → /detect/red_line → switch_control_node → /switch/control
switch_control_node → decision_node (Obstacle trigger)
decision_node → /decision/action → localization_node (tracks position)
                                        → explorer_node (decides turn)
                                        → control_lane_node (stops)
```

**Gate Mapping**:
```
camera → detect_apriltag_node → /apriltag/id → gate_mapper_node → /mapping/gates
```
