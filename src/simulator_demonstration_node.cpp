#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <functional>
#include <random>
#include <Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/voxel_grid.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>
#include <uncertainty_planning_core/simple_pid_controller.hpp>
#include <uncertainty_planning_core/simple_uncertainty_models.hpp>
#include <uncertainty_planning_core/uncertainty_contact_planning.hpp>
#include <uncertainty_planning_core/simplese3_robot_helpers.hpp>
#include <thruster_robot_controllers/SetActuationError.h>
#include <uncertainty_planning_core/se2_common_config.hpp>
#include <uncertainty_planning_core/se3_common_config.hpp>
#include <uncertainty_planning_core/baxter_linked_common_config.hpp>
#include <uncertainty_planning_core/ur5_linked_common_config.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

using namespace uncertainty_contact_planning;

void demonstrate_se3(ros::Publisher& display_debug_publisher)
{
    std::cout << "Demonstrating SE(3)..." << std::endl;
    const uncertainty_planning_core::OPTIONS options = se3_common_config::GetOptions();
    std::cout << PrettyPrint::PrettyPrint(options) << std::endl;
    const std::pair<Eigen::Affine3d, Eigen::Affine3d> start_and_goal = se3_common_config::GetStartAndGoal();
    const simplese3_robot_helpers::SimpleSE3BaseSampler sampler = se3_common_config::GetSampler();
    const simplese3_robot_helpers::ROBOT_CONFIG robot_config = se3_common_config::GetDefaultRobotConfig(options);
    const simplese3_robot_helpers::SimpleSE3Robot robot = se3_common_config::GetRobot(robot_config);
    uncertainty_planning_core::DemonstrateSE3Simulator(options, robot, sampler, start_and_goal.first, start_and_goal.second, display_debug_publisher);
}

void demonstrate_se2(ros::Publisher& display_debug_publisher)
{
    std::cout << "Demonstrating SE(2)..." << std::endl;
    const uncertainty_planning_core::OPTIONS options = se2_common_config::GetOptions();
    std::cout << PrettyPrint::PrettyPrint(options) << std::endl;
    const std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1>> start_and_goal = se2_common_config::GetStartAndGoal();
    const simplese2_robot_helpers::SimpleSE2BaseSampler sampler = se2_common_config::GetSampler();
    const simplese2_robot_helpers::ROBOT_CONFIG robot_config = se2_common_config::GetDefaultRobotConfig(options);
    const simplese2_robot_helpers::SimpleSE2Robot robot = se2_common_config::GetRobot(robot_config);
    uncertainty_planning_core::DemonstrateSE2Simulator(options, robot, sampler, start_and_goal.first, start_and_goal.second, display_debug_publisher);
}

void demonstrate_baxter(ros::Publisher& display_debug_publisher)
{
    std::cout << "Demonstrating Baxter..." << std::endl;
    const uncertainty_planning_core::OPTIONS options = baxter_linked_common_config::GetOptions();
    std::cout << PrettyPrint::PrettyPrint(options) << std::endl;
    const std::vector<double> joint_uncertainty_params = baxter_linked_common_config::GetJointUncertaintyParams(options);
    assert(joint_uncertainty_params.size() == 7);
    const std::pair<baxter_linked_common_config::SLC, baxter_linked_common_config::SLC> start_and_goal = baxter_linked_common_config::GetStartAndGoal();
    const simplelinked_robot_helpers::SimpleLinkedBaseSampler sampler = baxter_linked_common_config::GetSampler();
    const simplelinked_robot_helpers::ROBOT_CONFIG robot_config = baxter_linked_common_config::GetDefaultRobotConfig(options);
    const Eigen::Affine3d base_transform = baxter_linked_common_config::GetBaseTransform();
    const simplelinked_robot_helpers::SimpleLinkedRobot<baxter_linked_common_config::BaxterJointActuatorModel> robot = baxter_linked_common_config::GetRobot(base_transform, robot_config, joint_uncertainty_params);
    uncertainty_planning_core::DemonstrateBaxterSimulator(options, robot, sampler, start_and_goal.first, start_and_goal.second, display_debug_publisher);
}

void demonstrate_ur5(ros::Publisher& display_debug_publisher)
{
    std::cout << "Demonstrating UR5..." << std::endl;
    const uncertainty_planning_core::OPTIONS options = ur5_linked_common_config::GetOptions();
    std::cout << PrettyPrint::PrettyPrint(options) << std::endl;
    const std::vector<double> joint_uncertainty_params = ur5_linked_common_config::GetJointUncertaintyParams(options);
    assert(joint_uncertainty_params.size() == 6);
    const std::pair<ur5_linked_common_config::SLC, ur5_linked_common_config::SLC> start_and_goal = ur5_linked_common_config::GetStartAndGoal();
    const simplelinked_robot_helpers::SimpleLinkedBaseSampler sampler = ur5_linked_common_config::GetSampler();
    const simplelinked_robot_helpers::ROBOT_CONFIG robot_config = ur5_linked_common_config::GetDefaultRobotConfig(options);
    const Eigen::Affine3d base_transform = ur5_linked_common_config::GetBaseTransform();
    const simplelinked_robot_helpers::SimpleLinkedRobot<ur5_linked_common_config::UR5JointActuatorModel> robot = ur5_linked_common_config::GetRobot(base_transform, robot_config, joint_uncertainty_params);
    uncertainty_planning_core::DemonstrateUR5Simulator(options, robot, sampler, start_and_goal.first, start_and_goal.second, display_debug_publisher);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "simulator_demonstration_node");
    ros::NodeHandle nh;
    ROS_INFO("Starting Simulator Demonstration Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    std::string robot_type;
    ros::NodeHandle nhp("~");
    nhp.param(std::string("robot_type"), robot_type, std::string("baxter"));
    if (robot_type == "se2")
    {
        demonstrate_se2(display_debug_publisher);
    }
    else if (robot_type == "se3")
    {
        demonstrate_se3(display_debug_publisher);
    }
    else if (robot_type == "baxter")
    {
        demonstrate_baxter(display_debug_publisher);
    }
    else if (robot_type == "ur5")
    {
        demonstrate_ur5(display_debug_publisher);
    }
    else
    {
        std::cout << "Robot type [" << robot_type << "] is not recognized" << std::endl;
    }
    return 0;
}
