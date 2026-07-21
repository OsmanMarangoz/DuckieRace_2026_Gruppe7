cmake_minimum_required(VERSION 2.8.3)
project(follow_lane)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

install(PROGRAMS
  src/detect_lane_node.py
  src/detect_apriltag_node.py
  src/switch_control_node.py
  src/control_lane_node.py
  src/decision_node.py
  src/configuration_node.py
  src/city_graph.py
  src/util.py
  src/explorer_node.py
  src/localization_node.py
  src/gate_mapper_node.py
  src/mapping_debug_overlay.py
  src/mapping_debug_terminal.py
  DESTINATION lib/${PROJECT_NAME}
)