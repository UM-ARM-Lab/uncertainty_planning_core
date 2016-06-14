#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <random>
#include <Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include "arc_utilities/eigen_helpers.hpp"
#include "arc_utilities/eigen_helpers_conversions.hpp"
#include "arc_utilities/pretty_print.hpp"
#include "arc_utilities/voxel_grid.hpp"
#include "arc_utilities/simple_rrt_planner.hpp"
#include "nomdp_planning/simple_pid_controller.hpp"
#include "nomdp_planning/simple_uncertainty_models.hpp"
#include "nomdp_planning/nomdp_contact_planning.hpp"
#include "nomdp_planning/eigenvector3d_robot_helpers.hpp"
#include "nomdp_planning/simple6dof_robot_helpers.hpp"
#include "nomdp_planning/simplese2_robot_helpers.hpp"
#include "nomdp_planning/simplelinked_robot_helpers.hpp"
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>

//#define USE_6DOF
//#define USE_R3
//#define USE_SE2
//#define USE_LINKED

#ifndef USE_6DOF
    #ifndef USE_R3
        #ifndef USE_SE2
            #ifndef USE_LINKED
                #error ROBOT TYPE MUST BE SELECTED VIA DEFINE
            #endif
        #endif
    #endif
#endif

using namespace nomdp_contact_planning;

std::vector<nomdp_planning_tools::OBSTACLE_CONFIG> make_environment(const int32_t environment_code)
{
    if (environment_code == 0)
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.75, 2.0, 2.0), 0x55, 0x6b, 0x2f, 0xff)};
    }
    else if (environment_code == 1)
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(0.75, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.75, 2.0, 2.0), 0x55, 0x6b, 0x2f, 0xff)};
    }
    else if (environment_code == 2)
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.75, 2.0, 2.0), 0x55, 0x6b, 0x2f, 0xff),
                                                                    nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(-1.0, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.75, 2.0, 2.0), 0x55, 0x6b, 0x2f, 0xff)};
    }
    else if (environment_code == 3)
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(1.25, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.75, 2.0, 2.0), 0x55, 0x6b, 0x2f, 0xff),
                                                                    nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(-1.0, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.75, 2.0, 2.0), 0x55, 0x6b, 0x2f, 0xff)};
    }
    else if (environment_code == 4)
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.75, 2.0, 2.0), 0x55, 0x6b, 0x2f, 0xff),
                                                                    nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(-1.0, -1.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.75, 2.0, 2.0), 0x55, 0x6b, 0x2f, 0xff)};
    }
    else if (environment_code == 5)
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(-1.0, 0.0, 0.0), Eigen::Quaterniond(Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d::UnitZ())), Eigen::Vector3d(1.0, 1.0, 1.0), 0x55, 0x6b, 0x2f, 0xff)};
    }
    else if (environment_code == 6)
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(-1.0, 0.0, 0.0), Eigen::Quaterniond(Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d::UnitY())) * Eigen::Quaterniond(Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d::UnitZ())), Eigen::Vector3d(1.0, 1.0, 1.0), 0x55, 0x6b, 0x2f, 0xff)};
    }
    else
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.125, 2.0, 0.125), 0x55, 0x6b, 0x2f, 0xff)};
    }
}

std::vector<nomdp_planning_tools::OBSTACLE_CONFIG> make_linked_environment(const int32_t environment_code)
{
    if (environment_code == 0)
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(0.75, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.5, 1.0, 0.5), 0x55, 0x6b, 0x2f, 0xff),
                                                                    nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(-0.75, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.5, 1.0, 0.5), 0x55, 0x6b, 0x2f, 0xff)};
    }
    else if (environment_code == 1)
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(0.75, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.5, 1.0, 0.5), 0x55, 0x6b, 0x2f, 0xff),
                                                                    nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(-0.75, 0.5, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.5, 1.0, 0.5), 0x55, 0x6b, 0x2f, 0xff),
                                                                    nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(0.25, 1.825, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.125, 0.125, 0.125), 0x55, 0x6b, 0x2f, 0xff)};
    }
    else
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(10.0, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.5, 1.0, 0.5), 0x55, 0x6b, 0x2f, 0xff)};
    }
}

EigenHelpers::VectorVector3d make_robot(const int32_t robot_code)
{
    if (robot_code == 0)
    {
        EigenHelpers::VectorVector3d robot_points;
        robot_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.25, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.375, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.5, 0.0, 0.0));
        return robot_points;
    }
    else if (robot_code == 1)
    {
        EigenHelpers::VectorVector3d robot_points;
        robot_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        return robot_points;
    }
    else if (robot_code == 2)
    {
        EigenHelpers::VectorVector3d robot_points;
        robot_points.push_back(Eigen::Vector3d(-1.0, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.875, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.75, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.625, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.5, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.375, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.25, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.0, 0.0, 0.0));
        return robot_points;
    }
    else if (robot_code == 3)
    {
        EigenHelpers::VectorVector3d robot_points;
        robot_points.push_back(Eigen::Vector3d(0.0, -0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.25, 0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.25, 0.125, 0.0));
        return robot_points;
    }
    else if (robot_code == 4)
    {
        EigenHelpers::VectorVector3d robot_points;
        robot_points.push_back(Eigen::Vector3d(-0.125, -0.125, -0.125));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.0, -0.125));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.125, -0.125));
        robot_points.push_back(Eigen::Vector3d(0.0, -0.125, -0.125));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.0, -0.125));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.125, -0.125));
        robot_points.push_back(Eigen::Vector3d(0.125, -0.125, -0.125));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.0, -0.125));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.125, -0.125));
        robot_points.push_back(Eigen::Vector3d(-0.125, -0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.0, -0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.125, -0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.125, 0.0));
        robot_points.push_back(Eigen::Vector3d(-0.125, -0.125, 0.125));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.0, 0.125));
        robot_points.push_back(Eigen::Vector3d(-0.125, 0.125, 0.125));
        robot_points.push_back(Eigen::Vector3d(0.0, -0.125, 0.125));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.125));
        robot_points.push_back(Eigen::Vector3d(0.0, 0.125, 0.125));
        robot_points.push_back(Eigen::Vector3d(0.125, -0.125, 0.125));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.125));
        robot_points.push_back(Eigen::Vector3d(0.125, 0.125, 0.125));
        return robot_points;
    }
    else
    {
        return EigenHelpers::VectorVector3d(1, Eigen::Vector3d(0.0, 0.0, 0.0));
    }
}

std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedRobot> make_linked_robot(const int32_t robot_code, const simplelinked_robot_helpers::ROBOT_CONFIG& joint_config)
{
    if (robot_code == 0)
    {
        // Make the reference configuration
        simplelinked_robot_helpers::SimpleLinkedConfiguration reference_configuration = {simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE)};
        // Make the robot model
        simplelinked_robot_helpers::RobotLink link1;
        link1.link_name = "link1";
        link1.link_points = EigenHelpers::VectorVector3d();
        simplelinked_robot_helpers::RobotLink link2;
        link2.link_name = "link2";
        link2.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link3;
        link3.link_name = "link3";
        link3.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link4;
        link4.link_name = "link4";
        link4.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        std::vector<simplelinked_robot_helpers::RobotLink> links = {link1, link2, link3, link4};
        std::vector<std::pair<size_t, size_t>> allowed_self_collisions = {std::pair<size_t, size_t>(0, 1), std::pair<size_t, size_t>(1, 2), std::pair<size_t, size_t>(2, 3)};
        simplelinked_robot_helpers::RobotJoint joint1;
        joint1.parent_link_index = 0;
        joint1.child_link_index = 1;
        joint1.joint_axis = Eigen::Vector3d::UnitZ();
        joint1.joint_transform = Eigen::Translation3d(0.0, 0.0, 0.0) * Eigen::Quaterniond::Identity();
        joint1.joint_model = reference_configuration[0];
        joint1.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        simplelinked_robot_helpers::RobotJoint joint2;
        joint2.parent_link_index = 1;
        joint2.child_link_index = 2;
        joint2.joint_axis = Eigen::Vector3d::UnitZ();
        joint2.joint_transform = Eigen::Translation3d(0.75, 0.0, 0.125) * Eigen::Quaterniond::Identity();
        joint2.joint_model = reference_configuration[1];
        joint2.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        simplelinked_robot_helpers::RobotJoint joint3;
        joint3.parent_link_index = 2;
        joint3.child_link_index = 3;
        joint3.joint_axis = Eigen::Vector3d::UnitZ();
        joint3.joint_transform = Eigen::Translation3d(0.75, 0.0, -0.125) * Eigen::Quaterniond::Identity();
        joint3.joint_model = reference_configuration[2];
        joint3.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        std::vector<simplelinked_robot_helpers::RobotJoint>joints = {joint1, joint2, joint3};
        simplelinked_robot_helpers::SimpleLinkedRobot robot(links, joints, allowed_self_collisions, reference_configuration);
        return std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedRobot>(reference_configuration, robot);
    }
    else if (robot_code == 1)
    {
        // Make the reference configuration
        simplelinked_robot_helpers::SimpleLinkedConfiguration reference_configuration = {simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE)};
        // Make the robot model
        simplelinked_robot_helpers::RobotLink link1;
        link1.link_name = "link1";
        link1.link_points = EigenHelpers::VectorVector3d();
        simplelinked_robot_helpers::RobotLink link2;
        link2.link_name = "link2";
        link2.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link3;
        link3.link_name = "link3";
        link3.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link4;
        link4.link_name = "link4";
        link4.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        std::vector<simplelinked_robot_helpers::RobotLink> links = {link1, link2, link3, link4};
        std::vector<std::pair<size_t, size_t>> allowed_self_collisions = {std::pair<size_t, size_t>(0, 1), std::pair<size_t, size_t>(1, 2), std::pair<size_t, size_t>(2, 3)};
        simplelinked_robot_helpers::RobotJoint joint1;
        joint1.parent_link_index = 0;
        joint1.child_link_index = 1;
        joint1.joint_axis = Eigen::Vector3d::UnitZ();
        joint1.joint_transform = Eigen::Translation3d(0.0, 0.0, 0.0) * Eigen::Quaterniond::Identity();
        joint1.joint_model = reference_configuration[0];
        joint1.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        simplelinked_robot_helpers::RobotJoint joint2;
        joint2.parent_link_index = 1;
        joint2.child_link_index = 2;
        joint2.joint_axis = Eigen::Vector3d::UnitZ();
        joint2.joint_transform = Eigen::Translation3d(1.0, 0.0, 0.125) * Eigen::Quaterniond::Identity();
        joint2.joint_model = reference_configuration[1];
        joint2.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        simplelinked_robot_helpers::RobotJoint joint3;
        joint3.parent_link_index = 2;
        joint3.child_link_index = 3;
        joint3.joint_axis = Eigen::Vector3d::UnitZ();
        joint3.joint_transform = Eigen::Translation3d(0.75, 0.0, -0.125) * Eigen::Quaterniond::Identity();
        joint3.joint_model = reference_configuration[2];
        joint3.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        std::vector<simplelinked_robot_helpers::RobotJoint>joints = {joint1, joint2, joint3};
        simplelinked_robot_helpers::SimpleLinkedRobot robot(links, joints, allowed_self_collisions, reference_configuration);
        return std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedRobot>(reference_configuration, robot);
    }
    else
    {
        // Make the reference configuration
        simplelinked_robot_helpers::SimpleLinkedConfiguration reference_configuration = {simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE)};
        // Make the robot model
        simplelinked_robot_helpers::RobotLink link1;
        link1.link_name = "link1";
        link1.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link1.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link1.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link1.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link1.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link1.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link1.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link1.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link1.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link2;
        link2.link_name = "link2";
        link2.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link3;
        link3.link_name = "link3";
        link3.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link4;
        link4.link_name = "link4";
        link4.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link5;
        link5.link_name = "link5";
        link5.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link5.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link5.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link5.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link5.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link5.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link5.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link5.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link5.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link6;
        link6.link_name = "link6";
        link6.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link6.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link6.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link6.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link6.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link6.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link6.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link6.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link6.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link7;
        link7.link_name = "link7";
        link7.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        std::vector<simplelinked_robot_helpers::RobotLink> links = {link1, link2, link3, link4, link5, link6, link7};
        std::vector<std::pair<size_t, size_t>> allowed_self_collisions = {std::pair<size_t, size_t>(0, 1), std::pair<size_t, size_t>(1, 2), std::pair<size_t, size_t>(2, 3), std::pair<size_t, size_t>(3, 4), std::pair<size_t, size_t>(4, 5), std::pair<size_t, size_t>(5, 6)};
        simplelinked_robot_helpers::RobotJoint joint1;
        joint1.parent_link_index = 0;
        joint1.child_link_index = 1;
        joint1.joint_axis = Eigen::Vector3d::UnitZ();
        joint1.joint_transform = Eigen::Translation3d(1.0, 0.0, 0.125) * Eigen::Quaterniond::Identity();
        joint1.joint_model = reference_configuration[0];
        joint1.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        simplelinked_robot_helpers::RobotJoint joint2;
        joint2.parent_link_index = 1;
        joint2.child_link_index = 2;
        joint2.joint_axis = Eigen::Vector3d::UnitZ();
        joint2.joint_transform = Eigen::Translation3d(1.0, 0.0, -0.125) * Eigen::Quaterniond::Identity();
        joint2.joint_model = reference_configuration[1];
        joint2.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        simplelinked_robot_helpers::RobotJoint joint3;
        joint3.parent_link_index = 2;
        joint3.child_link_index = 3;
        joint3.joint_axis = Eigen::Vector3d::UnitZ();
        joint3.joint_transform = Eigen::Translation3d(1.0, 0.0, 0.125) * Eigen::Quaterniond::Identity();
        joint3.joint_model = reference_configuration[2];
        joint3.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        simplelinked_robot_helpers::RobotJoint joint4;
        joint4.parent_link_index = 3;
        joint4.child_link_index = 4;
        joint4.joint_axis = Eigen::Vector3d::UnitZ();
        joint4.joint_transform = Eigen::Translation3d(1.0, 0.0, -0.125) * Eigen::Quaterniond::Identity();
        joint4.joint_model = reference_configuration[3];
        joint4.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        simplelinked_robot_helpers::RobotJoint joint5;
        joint5.parent_link_index = 4;
        joint5.child_link_index = 5;
        joint5.joint_axis = Eigen::Vector3d::UnitZ();
        joint5.joint_transform = Eigen::Translation3d(1.0, 0.0, 0.125) * Eigen::Quaterniond::Identity();
        joint5.joint_model = reference_configuration[4];
        joint5.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        simplelinked_robot_helpers::RobotJoint joint6;
        joint6.parent_link_index = 5;
        joint6.child_link_index = 6;
        joint6.joint_axis = Eigen::Vector3d::UnitZ();
        joint6.joint_transform = Eigen::Translation3d(1.0, 0.0, -0.125) * Eigen::Quaterniond::Identity();
        joint6.joint_model = reference_configuration[5];
        joint6.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        std::vector<simplelinked_robot_helpers::RobotJoint>joints = {joint1, joint2, joint3, joint4, joint5, joint6};
        simplelinked_robot_helpers::SimpleLinkedRobot robot(links, joints, allowed_self_collisions, reference_configuration);
        return std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedRobot>(reference_configuration, robot);
    }
}

std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfiguration> make_start_and_goal(const int32_t robot_code)
{
    if (robot_code == 0)
    {
        // Start
        simplelinked_robot_helpers::SimpleLinkedConfiguration start = {simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), -M_PI_4, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), M_PI_2, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), -(M_PI_4 * 1.5), simplelinked_robot_helpers::SimpleJointModel::REVOLUTE)};
        // Goal
        simplelinked_robot_helpers::SimpleLinkedConfiguration goal = {simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE)};
        return std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfiguration>(start, goal);
    }
    else if (robot_code == 1)
    {
        // Start
        simplelinked_robot_helpers::SimpleLinkedConfiguration start = {simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE)};
        // Goal
        simplelinked_robot_helpers::SimpleLinkedConfiguration goal = {simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), (M_PI * 0.65), simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                        simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), (M_PI * 0.8), simplelinked_robot_helpers::SimpleJointModel::REVOLUTE)};
        return std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfiguration>(start, goal);
    }
    else
    {
        simplelinked_robot_helpers::SimpleLinkedConfiguration start = {simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), -M_PI_4, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), M_PI_4, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE)};
        simplelinked_robot_helpers::SimpleLinkedConfiguration goal = {simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), M_PI_4, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), 0.0, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE),
                                                                                    simplelinked_robot_helpers::SimpleJointModel(std::pair<double, double>(-M_PI, M_PI), -M_PI_4, simplelinked_robot_helpers::SimpleJointModel::REVOLUTE)};
        return std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfiguration>(start, goal);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "simulator_test_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Simulator Test Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    // Get environment and robot types
    int32_t env_code = 0; //0;
    int32_t robot_code = 0; //4; //0;
    if (argc >= 2)
    {
        env_code = (int32_t)atoi(argv[1]);
    }
    if (argc >= 3)
    {
        robot_code = (int32_t)atoi(argv[2]);
    }
    // Make the environment
#ifdef USE_LINKED
    const std::vector<nomdp_planning_tools::OBSTACLE_CONFIG> env_objects = make_linked_environment(env_code);
#endif
#ifndef USE_LINKED
    const std::vector<nomdp_planning_tools::OBSTACLE_CONFIG> env_objects = make_environment(env_code);
#endif
    // Initialize the planning space
    double env_resolution = 0.125;
    double step_size = 10 * env_resolution;
    double goal_distance_threshold = 1.0 * env_resolution;
    double goal_probability_threshold = 0.51;
    double signature_matching_threshold = 0.125;
    double distance_clustering_threshold = 1.0;
    double env_min_x = 0.0 + (env_resolution);
    double env_max_x = 10.0 - (env_resolution);
    double env_min_y = 0.0 + (env_resolution);
    double env_max_y = 10.0 - (env_resolution);
    double env_min_z = 0.0 + (env_resolution);
    double env_max_z = 10.0 - (env_resolution);
#ifdef USE_6DOF
    simple6dof_robot_helpers::Simple6DOFBaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
#endif
#ifdef USE_R3
    eigenvector3d_robot_helpers::EigenVector3dBaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
#endif
#ifdef USE_SE2
    UNUSED(env_min_z);
    UNUSED(env_max_z);
    simplese2_robot_helpers::SimpleSE2BaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y));
#endif
#ifdef USE_LINKED
    UNUSED(env_min_x);
    UNUSED(env_max_x);
    UNUSED(env_min_y);
    UNUSED(env_max_y);
    UNUSED(env_min_z);
    UNUSED(env_max_z);
#endif
    double kp = 0.25;
    double ki = 0.0;
    double kd = 0.05;
    double i_clamp = 0.0;
    double velocity_limit = env_resolution * 2.0;
    double max_sensor_noise = 0.0; //env_resolution * 0.01;
    double max_actuator_noise = 0.0; //env_resolution * 1.0; //1.0;
    double feasibility_alpha = 0.75;
    double variance_alpha = 0.75;
    // Make the actual robot
#ifdef USE_6DOF
    // Make the robot geometry
    const EigenHelpers::VectorVector3d robot_points = make_robot(robot_code);
    simple6dof_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, (velocity_limit * 0.125), (max_sensor_noise * 0.125), (max_actuator_noise * 0.125));
    Eigen::Matrix<double, 6, 1> initial_config = Eigen::Matrix<double, 6, 1>::Zero();
    simple6dof_robot_helpers::Simple6DOFRobot robot(robot_points, initial_config, robot_config);
#endif
#ifdef USE_R3
    // Make the robot geometry
    const EigenHelpers::VectorVector3d robot_points = make_robot(robot_code);
    eigenvector3d_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    eigenvector3d_robot_helpers::SimpleEigenVector3dRobot robot(robot_points, Eigen::Vector3d::Zero(), robot_config);
#endif
#ifdef USE_SE2
    // Make the robot geometry
    const EigenHelpers::VectorVector3d robot_points = make_robot(robot_code);
    simplese2_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, (velocity_limit * 0.125), (max_sensor_noise * 0.125), (max_actuator_noise * 0.125));
    Eigen::Matrix<double, 3, 1> initial_config = Eigen::Matrix<double, 3, 1>::Zero();
    simplese2_robot_helpers::SimpleSE2Robot robot(robot_points, initial_config, robot_config);
#endif
#ifdef USE_LINKED
    // Make the robot
    simplelinked_robot_helpers::ROBOT_CONFIG joint_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedRobot> built_robot = make_linked_robot(robot_code, joint_config);
    simplelinked_robot_helpers::SimpleLinkedConfiguration reference_configuration = built_robot.first;
    simplelinked_robot_helpers::SimpleLinkedRobot robot = built_robot.second;
    const Eigen::Affine3d base_transform = Eigen::Translation3d(0.275, 2.5, 0.0) * Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitZ()));
    robot.SetBaseTransform(base_transform);
    simplelinked_robot_helpers::SimpleLinkedBaseSampler sampler(reference_configuration);
#endif
#ifdef USE_6DOF
    NomdpPlanningSpace<simple6dof_robot_helpers::Simple6DOFRobot, simple6dof_robot_helpers::Simple6DOFBaseSampler, Eigen::Matrix<double, 6, 1>, simple6dof_robot_helpers::EigenMatrixD61Serializer, simple6dof_robot_helpers::Simple6DOFAverager, simple6dof_robot_helpers::Simple6DOFDistancer, simple6dof_robot_helpers::Simple6DOFDimDistancer, simple6dof_robot_helpers::Simple6DOFInterpolator, std::allocator<Eigen::Matrix<double, 6, 1>>, std::mt19937_64> planning_space(true, 1u, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, distance_clustering_threshold, feasibility_alpha, variance_alpha, robot, sampler, env_objects, env_resolution);
#endif
#ifdef USE_R3
    NomdpPlanningSpace<eigenvector3d_robot_helpers::SimpleEigenVector3dRobot, eigenvector3d_robot_helpers::EigenVector3dBaseSampler, Eigen::Vector3d, eigenvector3d_robot_helpers::EigenVector3dSerializer, eigenvector3d_robot_helpers::EigenVector3dAverager, eigenvector3d_robot_helpers::EigenVector3dDistancer, eigenvector3d_robot_helpers::EigenVector3dDimDistancer, eigenvector3d_robot_helpers::EigenVector3dInterpolator, Eigen::aligned_allocator<Eigen::Vector3d>, std::mt19937_64> planning_space(true, 1u, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, distance_clustering_threshold, feasibility_alpha, variance_alpha, robot, sampler, env_objects, env_resolution);
#endif
#ifdef USE_SE2
    NomdpPlanningSpace<simplese2_robot_helpers::SimpleSE2Robot, simplese2_robot_helpers::SimpleSE2BaseSampler, Eigen::Matrix<double, 3, 1>, simple6dof_robot_helpers::EigenMatrixD61Serializer, simplese2_robot_helpers::SimpleSE2Averager, simplese2_robot_helpers::SimpleSE2Distancer, simplese2_robot_helpers::SimpleSE2DimDistancer, simplese2_robot_helpers::SimpleSE2Interpolator, std::allocator<Eigen::Matrix<double, 3, 1>>, std::mt19937_64> planning_space(true, 1u, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, distance_clustering_threshold, feasibility_alpha, variance_alpha, robot, sampler, env_objects, env_resolution);
#endif
#ifdef USE_LINKED
    NomdpPlanningSpace<simplelinked_robot_helpers::SimpleLinkedRobot, simplelinked_robot_helpers::SimpleLinkedBaseSampler, simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfigurationSerializer, simplelinked_robot_helpers::SimpleLinkedAverager, simplelinked_robot_helpers::SimpleLinkedDistancer, simplelinked_robot_helpers::SimpleLinkedDimDistancer, simplelinked_robot_helpers::SimpleLinkedInterpolator, std::allocator<simplelinked_robot_helpers::SimpleLinkedConfiguration>, std::mt19937_64> planning_space(false, 1u, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, distance_clustering_threshold, feasibility_alpha, variance_alpha, robot, sampler, env_objects, env_resolution);
#endif
    // Now, run a series of simulator tests
#ifdef USE_6DOF
    Eigen::Matrix<double, 6, 1> start;
    start << 0.0, 3.0, 0.0, 0.0, 0.0, 0.0;
    Eigen::Matrix<double, 6, 1> goal;
    goal << 0.0, -3.0, 0.0, 0.0, 0.0, 0.0;
#endif
#ifdef USE_R3
    Eigen::Vector3d start(0.0, 3.0, 0.0);
    Eigen::Vector3d goal(0.0, -3.0, 0.0);
#endif
#ifdef USE_SE2
    Eigen::Matrix<double, 3, 1> start;
    start << 0.0, 3.0, 0.0;
    Eigen::Matrix<double, 3, 1> goal;
    goal << 0.0, -3.0, 0.0;
#endif
#ifdef USE_LINKED
    std::pair<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfiguration> start_and_goal = make_start_and_goal(robot_code);
    const simplelinked_robot_helpers::SimpleLinkedConfiguration& start = start_and_goal.first;
    const simplelinked_robot_helpers::SimpleLinkedConfiguration& goal = start_and_goal.second;
#endif
    const bool enable_contact_manifold_target_adjustment = false;
    planning_space.DemonstrateSimulator(start, goal, enable_contact_manifold_target_adjustment, display_debug_publisher);
    return 0;
}
