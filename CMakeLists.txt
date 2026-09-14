cmake_minimum_required(VERSION 3.0.2)
project(rigid_mask_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/rigid_mask_node
  scripts/rigid_mask_node.py
  scripts/rigid_mask_env_check
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
