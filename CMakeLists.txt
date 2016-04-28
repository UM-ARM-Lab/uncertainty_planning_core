cmake_minimum_required(VERSION 2.8.3)
project(nomdp_planning)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy geometry_msgs visualization_msgs std_msgs sdf_tools arc_utilities)

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
catkin_package(CATKIN_DEPENDS roscpp rospy geometry_msgs visualization_msgs std_msgs sdf_tools arc_utilities DEPENDS system_lib rt Eigen)

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

set(NOMDP_CORE_SOURCES
    include/${PROJECT_NAME}/nomdp_planner_state.hpp
    include/${PROJECT_NAME}/simple_pid_controller.hpp
    include/${PROJECT_NAME}/simple_uncertainty_models.hpp
    include/${PROJECT_NAME}/simple_particle_contact_simulator.hpp
    include/${PROJECT_NAME}/nomdp_contact_planning.hpp
    include/${PROJECT_NAME}/execution_policy.hpp
    include/${PROJECT_NAME}/eigenvector3d_robot_helpers.hpp
    include/${PROJECT_NAME}/eigenvectorxd_robot_helpers.hpp
    include/${PROJECT_NAME}/simple6dof_robot_helpers.hpp
    include/${PROJECT_NAME}/proper6dof_robot_helpers.hpp
    include/${PROJECT_NAME}/simplese2_robot_helpers.hpp
    include/${PROJECT_NAME}/simplelinked_robot_helpers.hpp)

set(NOMDP_SOURCES ${NOMDP_CORE_SOURCES} src/nomdp_contact_planning_node.cpp)

#add_executable(se2_simulator_test_node ${NOMDP_CORE_SOURCES} src/simulator_test_node.cpp)
#add_dependencies(se2_simulator_test_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
#target_link_libraries(se2_simulator_test_node ${catkin_LIBRARIES} rt)
#set_target_properties(se2_simulator_test_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DUSE_SE2")

#add_executable(r3_simulator_test_node ${NOMDP_CORE_SOURCES} src/simulator_test_node.cpp)
#add_dependencies(r3_simulator_test_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
#target_link_libraries(r3_simulator_test_node ${catkin_LIBRARIES} rt)
#set_target_properties(r3_simulator_test_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DUSE_R3")

add_executable(sixdof_simulator_test_node ${NOMDP_CORE_SOURCES} src/simulator_test_node.cpp)
add_dependencies(sixdof_simulator_test_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(sixdof_simulator_test_node ${catkin_LIBRARIES} rt)
set_target_properties(sixdof_simulator_test_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DUSE_6DOF")

add_executable(linked_simulator_test_node ${NOMDP_CORE_SOURCES} src/simulator_test_node.cpp)
add_dependencies(linked_simulator_test_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(linked_simulator_test_node ${catkin_LIBRARIES} rt)
set_target_properties(linked_simulator_test_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DUSE_LINKED")

add_executable(nomdp_contact_planning_node_no_ros_debug ${NOMDP_SOURCES})
add_dependencies(nomdp_contact_planning_node_no_ros_debug ${catkin_EXPORTED_TARGETS})
set_target_properties(nomdp_contact_planning_node_no_ros_debug PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DDISABLE_ROS_INTERFACE -DFORCE_DEBUG")

add_executable(nomdp_contact_planning_node_debug ${NOMDP_SOURCES})
add_dependencies(nomdp_contact_planning_node_debug ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(nomdp_contact_planning_node_debug ${catkin_LIBRARIES} rt)
set_target_properties(nomdp_contact_planning_node_debug PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DFORCE_DEBUG")

add_executable(nomdp_contact_planning_node_parallel_debug ${NOMDP_SOURCES})
add_dependencies(nomdp_contact_planning_node_parallel_debug ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(nomdp_contact_planning_node_parallel_debug ${catkin_LIBRARIES} rt)
set_target_properties(nomdp_contact_planning_node_parallel_debug PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -DFORCE_DEBUG -DENABLE_PARALLEL")

add_executable(nomdp_contact_planning_node ${NOMDP_SOURCES})
add_dependencies(nomdp_contact_planning_node ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(nomdp_contact_planning_node ${catkin_LIBRARIES} rt)
set_target_properties(nomdp_contact_planning_node PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")

add_executable(nomdp_contact_planning_node_parallel ${NOMDP_SOURCES})
add_dependencies(nomdp_contact_planning_node_parallel_debug ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencpp)
target_link_libraries(nomdp_contact_planning_node_parallel ${catkin_LIBRARIES} rt)
set_target_properties(nomdp_contact_planning_node_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DENABLE_PARALLEL")

add_executable(nomdp_contact_planning_node_no_ros ${NOMDP_SOURCES})
add_dependencies(nomdp_contact_planning_node_no_ros ${catkin_EXPORTED_TARGETS})
set_target_properties(nomdp_contact_planning_node_no_ros PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DDISABLE_ROS_INTERFACE")

add_executable(nomdp_contact_planning_node_no_ros_parallel ${NOMDP_SOURCES})
add_dependencies(nomdp_contact_planning_node_no_ros_parallel ${catkin_EXPORTED_TARGETS})
set_target_properties(nomdp_contact_planning_node_no_ros_parallel PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DENABLE_PARALLEL -DDISABLE_ROS_INTERFACE")

if(BUILD_XEON_PHI_NATIVE)
    message(STATUS "Building executable for Xeon Phi (Native mode)")
    add_executable(nomdp_contact_planning_node_mic ${NOMDP_SOURCES})
    add_dependencies(nomdp_contact_planning_node_mic ${catkin_EXPORTED_TARGETS})
    set_target_properties(nomdp_contact_planning_node_mic PROPERTIES COMPILE_FLAGS "-I/usr/include -I/usr/include/x86_64-linux-gnu -mmic -openmp -std=c++0x ${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -DENABLE_PARALLEL -DDISABLE_ROS_INTERFACE")
else()
    message(WARNING "Not building executable for Xeon Phi (Native mode)")
endif()

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS deformable_ompl deformable_ompl_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_deformable_ompl.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
