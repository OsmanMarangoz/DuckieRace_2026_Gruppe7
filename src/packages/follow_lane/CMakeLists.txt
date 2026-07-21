cmake_minimum_required(VERSION 2.8.3)
project(follow_lane)

find_package(catkin REQUIRED COMPONENTS
  duckietown_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/kreiselfahrt_node.py
  src/kreiselfahrt_dashboard.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  src/kreiselfahrt_logic.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  config/kreiselfahrt_node.json
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(FILES
  launch/kreiselfahrt.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(test/kreiselfahrt_node.test)
endif()
