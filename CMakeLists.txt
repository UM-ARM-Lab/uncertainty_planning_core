cmake_minimum_required(VERSION 2.8.3)
project(uncertainty_planning_core)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp visualization_msgs sdf_tools arc_utilities)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
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
#add_message_files(DIRECTORY msg FILES MS.msg)

## Generate services in the 'srv' folder
#add_service_files(DIRECTORY srv FILES SR.srv)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
#generate_messages(DEPENDENCIES geometry_msgs visualization_msgs std_msgs control_msgs)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(INCLUDE_DIRS include LIBRARIES ${PROJECT_NAME} CATKIN_DEPENDS roscpp visualization_msgs sdf_tools arc_utilities DEPENDS system_lib rt Eigen3 OpenMP)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(include SYSTEM ${catkin_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -Wall -Wextra -Wconversion")
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wall -Wextra -Wconversion -O3 -g -DENABLE_PARALLEL_DISTANCE_MATRIX -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wconversion")
#set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -O3 -g -DENABLE_PARALLEL -DEIGEN_DONT_PARALLELIZE")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

set(UNCERTAINTY_PLANNING_CORE_SOURCES
    include/${PROJECT_NAME}/simple_pid_controller.hpp
    include/${PROJECT_NAME}/simple_uncertainty_models.hpp
    include/${PROJECT_NAME}/simple_robot_models.hpp
    include/${PROJECT_NAME}/simple_samplers.hpp
    include/${PROJECT_NAME}/simple_simulator_interface.hpp
    include/${PROJECT_NAME}/uncertainty_planner_state.hpp
    include/${PROJECT_NAME}/uncertainty_contact_planning.hpp
    include/${PROJECT_NAME}/execution_policy.hpp
    include/${PROJECT_NAME}/uncertainty_planning_core.hpp
    src/${PROJECT_NAME}/uncertainty_planning_core.cpp)

###################################################################################################################
# CORE LIBRARY
###################################################################################################################

add_library(${PROJECT_NAME} ${UNCERTAINTY_PLANNING_CORE_SOURCES})
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} rt)

