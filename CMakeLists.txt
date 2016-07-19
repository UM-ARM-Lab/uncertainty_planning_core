cmake_minimum_required(VERSION 2.8.3)
project(uncertainty_planning_core)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy geometry_msgs visualization_msgs std_msgs sdf_tools arc_utilities thruster_robot_controllers message_generation)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(cmake_modules REQUIRED)
find_package(Eigen REQUIRED)
find_package(OpenMP)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
#catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg FILES ObstacleConfig.msg)

## Generate services in the 'srv' folder
add_service_files(DIRECTORY srv FILES Simple6dofRobotMove.srv)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES geometry_msgs visualization_msgs std_msgs)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(CATKIN_DEPENDS roscpp rospy geometry_msgs visualization_msgs std_msgs sdf_tools arc_utilities thruster_robot_controllers message_runtime DEPENDS system_lib rt Eigen)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(include SYSTEM ${catkin_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -Wall -Wextra -Wconversion")
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wall -Wextra -Wconversion")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")

set(UNCERTAINTY_PLANNING_CORE_SOURCES
    include/${PROJECT_NAME}/nomdp_planner_state.hpp
    include/${PROJECT_NAME}/simple_pid_controller.hpp
    include/${PROJECT_NAME}/simple_uncertainty_models.hpp
    include/${PROJECT_NAME}/simple_particle_contact_simulator.hpp
    include/${PROJECT_NAME}/nomdp_contact_planning.hpp
    include/${PROJECT_NAME}/execution_policy.hpp)

# Robot-type specific sources
set(LINKED_SOURCES ${UNCERTAINTY_PLANNING_CORE_SOURCES} include/${PROJECT_NAME}/simplelinked_robot_helpers.hpp)
set(SE3_SOURCES ${UNCERTAINTY_PLANNING_CORE_SOURCES} include/${PROJECT_NAME}/simplese3_robot_helpers.hpp)
set(SE2_SOURCES ${UNCERTAINTY_PLANNING_CORE_SOURCES} include/${PROJECT_NAME}/simplese2_robot_helpers.hpp)

# Planning executable sources
set(LINKED_PLANNER_SOURCES ${LINKED_SOURCES} src/common_config.hpp src/baxter_joint_actuator_model.hpp src/baxter_linked_common_config.hpp src/linked_contact_planning_node.cpp)
set(SE3_PLANNER_SOURCES ${SE3_SOURCES} src/common_config.hpp src/se3_common_config.hpp src/se3_contact_planning_node.cpp)
set(SE3_CLUTTERED_PLANNER_SOURCES ${SE3_SOURCES} src/common_config.hpp src/se3_cluttered_common_config.hpp src/se3_cluttered_contact_planning_node.cpp)
set(SE3_NEW_CLUTTERED_PLANNER_SOURCES ${SE3_SOURCES} src/common_config.hpp src/se3_new_cluttered_common_config.hpp src/se3_new_cluttered_contact_planning_node.cpp)
set(SE2_PLANNER_SOURCES ${SE2_SOURCES} src/common_config.hpp src/se2_common_config.hpp src/se2_contact_planning_node.cpp)

# Policy execution sources
set(LINKED_EXECUTION_SOURCES ${LINKED_SOURCES} src/common_config.hpp src/baxter_joint_actuator_model.hpp src/baxter_linked_common_config.hpp src/linked_contact_execution_node.cpp)
set(SE3_EXECUTION_SOURCES ${SE3_SOURCES} src/common_config.hpp src/se3_common_config.hpp src/se3_contact_execution_node.cpp)
set(SE3_CLUTTERED_EXECUTION_SOURCES ${SE3_SOURCES} src/common_config.hpp src/se3_cluttered_common_config.hpp src/se3_cluttered_contact_execution_node.cpp)
set(SE3_NEW_CLUTTERED_EXECUTION_SOURCES ${SE3_SOURCES} src/common_config.hpp src/se3_new_cluttered_common_config.hpp src/se3_new_cluttered_contact_execution_node.cpp)
set(SE2_EXECUTION_SOURCES ${SE2_SOURCES} src/common_config.hpp src/se2_common_config.hpp src/se2_contact_execution_node.cpp)

###################################################################################################################
# ROS-ENABLED VERSIONS
###################################################################################################################

add_executable(linked_contact_planning_node_parallel ${LINKED_PLANNER_SOURCES})
add_dependencies(linked_contact_planning_node_parallel ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(linked_contact_planning_node_parallel ${catkin_LIBRARIES} rt)
set_target_properties(linked_contact_planning_node_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Og -g -DENABLE_PARALLEL -DENABLE_DEBUG_PRINTS -DEIGEN_DONT_PARALLELIZE")

add_executable(se3_contact_planning_node_parallel ${SE3_PLANNER_SOURCES})
add_dependencies(se3_contact_planning_node_parallel ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(se3_contact_planning_node_parallel ${catkin_LIBRARIES} rt)
set_target_properties(se3_contact_planning_node_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DENABLE_DEBUG_PRINTS -DEIGEN_DONT_PARALLELIZE")

add_executable(se3_cluttered_contact_planning_node_parallel ${SE3_CLUTTERED_PLANNER_SOURCES})
add_dependencies(se3_cluttered_contact_planning_node_parallel ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(se3_cluttered_contact_planning_node_parallel ${catkin_LIBRARIES} rt)
set_target_properties(se3_cluttered_contact_planning_node_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DENABLE_DEBUG_PRINTS -DEIGEN_DONT_PARALLELIZE")

add_executable(se3_new_cluttered_contact_planning_node_parallel ${SE3_NEW_CLUTTERED_PLANNER_SOURCES})
add_dependencies(se3_new_cluttered_contact_planning_node_parallel ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(se3_new_cluttered_contact_planning_node_parallel ${catkin_LIBRARIES} rt)
set_target_properties(se3_new_cluttered_contact_planning_node_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DENABLE_DEBUG_PRINTS -DEIGEN_DONT_PARALLELIZE")

add_executable(se2_contact_planning_node_parallel ${SE2_PLANNER_SOURCES})
add_dependencies(se2_contact_planning_node_parallel ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(se2_contact_planning_node_parallel ${catkin_LIBRARIES} rt)
set_target_properties(se2_contact_planning_node_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DENABLE_DEBUG_PRINTS -DEIGEN_DONT_PARALLELIZE")

###################################################################################################################
# ROS-DISABLED VERSIONS
###################################################################################################################

add_executable(linked_contact_planning_node_no_ros_parallel ${LINKED_PLANNER_SOURCES})
add_dependencies(linked_contact_planning_node_no_ros_parallel ${catkin_EXPORTED_TARGETS})
#target_link_libraries(linked_contact_planning_node_no_ros_parallel ${catkin_LIBRARIES} rt)
set_target_properties(linked_contact_planning_node_no_ros_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE -DDISABLE_ROS_INTERFACE -DENABLE_ROS_PARAMETERS")

add_executable(se3_contact_planning_node_no_ros_parallel ${SE3_PLANNER_SOURCES})
add_dependencies(se3_contact_planning_node_no_ros_parallel ${catkin_EXPORTED_TARGETS})
#target_link_libraries(se3_contact_planning_node_no_ros_parallel ${catkin_LIBRARIES} rt)
set_target_properties(se3_contact_planning_node_no_ros_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE -DDISABLE_ROS_INTERFACE -DENABLE_ROS_PARAMETERS")

add_executable(se3_cluttered_contact_planning_node_no_ros_parallel ${SE3_CLUTTERED_PLANNER_SOURCES})
add_dependencies(se3_cluttered_contact_planning_node_no_ros_parallel ${catkin_EXPORTED_TARGETS})
#target_link_libraries(se3_cluttered_contact_planning_node_no_ros_parallel ${catkin_LIBRARIES} rt)
set_target_properties(se3_cluttered_contact_planning_node_no_ros_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE -DDISABLE_ROS_INTERFACE -DENABLE_ROS_PARAMETERS")

add_executable(se3_new_cluttered_contact_planning_node_no_ros_parallel ${SE3_NEW_CLUTTERED_PLANNER_SOURCES})
add_dependencies(se3_new_cluttered_contact_planning_node_no_ros_parallel ${catkin_EXPORTED_TARGETS})
#target_link_libraries(se3_new_cluttered_contact_planning_node_no_ros_parallel ${catkin_LIBRARIES} rt)
set_target_properties(se3_new_cluttered_contact_planning_node_no_ros_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE -DDISABLE_ROS_INTERFACE -DENABLE_ROS_PARAMETERS")

add_executable(se2_contact_planning_node_no_ros_parallel ${SE2_PLANNER_SOURCES})
add_dependencies(se2_contact_planning_node_no_ros_parallel ${catkin_EXPORTED_TARGETS})
#target_link_libraries(se2_contact_planning_node_no_ros_parallel ${catkin_LIBRARIES} rt)
set_target_properties(se2_contact_planning_node_no_ros_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE -DDISABLE_ROS_INTERFACE -DENABLE_ROS_PARAMETERS")

###################################################################################################################
# SIMULATION AND EXECUTION LINKED
###################################################################################################################

add_executable(linked_contact_execution_node ${LINKED_EXECUTION_SOURCES})
add_dependencies(linked_contact_execution_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(linked_contact_execution_node ${catkin_LIBRARIES} rt)
set_target_properties(linked_contact_execution_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE")

###################################################################################################################
# SIMULATION AND EXECUTION SE(3) PEG IN HOLE
###################################################################################################################

add_executable(se3_contact_execution_node ${SE3_EXECUTION_SOURCES})
add_dependencies(se3_contact_execution_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(se3_contact_execution_node ${catkin_LIBRARIES} rt)
set_target_properties(se3_contact_execution_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DFORCE_DEBUG -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE")

###################################################################################################################
# SIMULATION AND EXECUTION SE(3) CLUTTERED
###################################################################################################################

add_executable(se3_cluttered_contact_execution_node ${SE3_CLUTTERED_EXECUTION_SOURCES})
add_dependencies(se3_cluttered_contact_execution_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(se3_cluttered_contact_execution_node ${catkin_LIBRARIES} rt)
set_target_properties(se3_cluttered_contact_execution_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DFORCE_DEBUG -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE")

###################################################################################################################
# SIMULATION AND EXECUTION SE(3) NEW CLUTTERED
###################################################################################################################

add_executable(se3_new_cluttered_contact_execution_node ${SE3_NEW_CLUTTERED_EXECUTION_SOURCES})
add_dependencies(se3_new_cluttered_contact_execution_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(se3_new_cluttered_contact_execution_node ${catkin_LIBRARIES} rt)
set_target_properties(se3_new_cluttered_contact_execution_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DFORCE_DEBUG -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE")

###################################################################################################################
# SIMULATION AND EXECUTION SE(2)
###################################################################################################################

add_executable(se2_contact_execution_node_debug ${SE2_EXECUTION_SOURCES})
add_dependencies(se2_contact_execution_node_debug ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(se2_contact_execution_node_debug ${catkin_LIBRARIES} rt)
set_target_properties(se2_contact_execution_node_debug PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

add_executable(se2_contact_execution_node ${SE2_EXECUTION_SOURCES})
add_dependencies(se2_contact_execution_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(se2_contact_execution_node ${catkin_LIBRARIES} rt)
set_target_properties(se2_contact_execution_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DFORCE_DEBUG -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE")

