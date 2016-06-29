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
#include "arc_utilities/eigen_helpers.hpp"
#include "arc_utilities/eigen_helpers_conversions.hpp"
#include "arc_utilities/pretty_print.hpp"
#include "arc_utilities/voxel_grid.hpp"
#include "arc_utilities/simple_rrt_planner.hpp"
#include "nomdp_planning/simple_pid_controller.hpp"
#include "nomdp_planning/simple_uncertainty_models.hpp"
#include "nomdp_planning/nomdp_contact_planning.hpp"
#include "nomdp_planning/simplese2_robot_helpers.hpp"
#include "linked_common_config.hpp"
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <nomdp_planning/Simple6dofRobotMove.h>

using namespace nomdp_contact_planning;

inline std::vector<linked_common_config::SLC, std::allocator<linked_common_config::SLC>> move_robot(const linked_common_config::SLC& target_config, const bool reset, ros::ServiceClient& robot_control_service)
{
    UNUSED(reset);
    UNUSED(robot_control_service);
    std::cout << "!!! EXECUTION NOT IMPLEMENTED YET !!!" << std::endl;
    std::vector<linked_common_config::SLC, std::allocator<linked_common_config::SLC>> configs = {target_config};
    return configs;
}

void peg_in_hole_env_linked(ros::Publisher& display_debug_publisher, ros::ServiceClient& robot_control_service)
{
    const common_config::OPTIONS options = linked_common_config::GetOptions(common_config::OPTIONS::EXECUTION);
    std::cout << PrettyPrint::PrettyPrint(options) << std::endl;
    const std::pair<linked_common_config::SLC, linked_common_config::SLC> start_and_goal = linked_common_config::GetStartAndGoal();
    const simplelinked_robot_helpers::SimpleLinkedBaseSampler sampler = linked_common_config::GetSampler();
    const simplelinked_robot_helpers::ROBOT_CONFIG robot_config = linked_common_config::GetDefaultRobotConfig(options);
    const Eigen::Affine3d base_transform = linked_common_config::GetBaseTransform();
    const simplelinked_robot_helpers::SimpleLinkedRobot robot = linked_common_config::GetRobot(base_transform, robot_config);
    NomdpPlanningSpace<simplelinked_robot_helpers::SimpleLinkedRobot, simplelinked_robot_helpers::SimpleLinkedBaseSampler, simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfigurationSerializer, simplelinked_robot_helpers::SimpleLinkedAverager, simplelinked_robot_helpers::SimpleLinkedDistancer, simplelinked_robot_helpers::SimpleLinkedDimDistancer, simplelinked_robot_helpers::SimpleLinkedInterpolator, std::allocator<simplelinked_robot_helpers::SimpleLinkedConfiguration>, std::mt19937_64> planning_space(options.clustering_type, false, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, robot, sampler, "peg_in_hole", options.environment_resolution);
    // Load the policy
    const auto policy = planning_space.LoadPolicy(options.planned_policy_file);
    std::map<std::string, double> complete_policy_stats;
    std::cout << "Press ENTER to simulate policy..." << std::endl;
    std::cin.get();
    const auto policy_simulation_stats = planning_space.SimulateExectionPolicy(policy, start_and_goal.first, start_and_goal.second, options.num_policy_simulations, options.exec_step_limit, options.enable_contact_manifold_target_adjustment, display_debug_publisher, false, 0.1, false);
    std::cout << "Policy simulation success: " << policy_simulation_stats.first.second << std::endl;
    complete_policy_stats["Policy simulation success"] = policy_simulation_stats.first.second;
    complete_policy_stats["Policy simulation successful resolves"] = policy_simulation_stats.second.first;
    complete_policy_stats["Policy simulation unsuccessful resolves"] = policy_simulation_stats.second.second;
    std::cout << "Press ENTER to execute policy..." << std::endl;
    std::cin.get();
    std::function<std::vector<linked_common_config::SLC, std::allocator<linked_common_config::SLC>>(const linked_common_config::SLC&, const bool)> robot_execution_fn = [&] (const linked_common_config::SLC& target_configuration, const bool reset) { return move_robot(target_configuration, reset, robot_control_service); };
    const auto execution_results = planning_space.ExecuteExectionPolicy(policy, start_and_goal.first, start_and_goal.second, robot_execution_fn, options.num_policy_executions, options.exec_step_limit, display_debug_publisher, false, 0.1, false);
    std::cout << "Policy execution success: " << PrettyPrint::PrettyPrint(execution_results.second, true) << std::endl;
    complete_policy_stats["Policy execution success"] = execution_results.second;
    // Save the executed policy
    planning_space.SavePolicy(execution_results.first, options.executed_policy_file);
    // Print out the results & save them to the log file
    const std::string log_results = PrettyPrint::PrettyPrint(complete_policy_stats, false, "\n");
    std::cout << "Policy results:\n" << log_results << std::endl;
    std::ofstream log_file(options.policy_log_file, std::ios_base::out | std::ios_base::app);
    if (!log_file.is_open())
    {
        std::cerr << "\x1b[31;1m Unable to create folder/file to log to: " << options.policy_log_file << "\x1b[37m \n";
        throw std::invalid_argument( "Log filename must be write-openable" );
    }
    log_file << log_results << std::endl;
    log_file.close();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "linked_contact_execution_node");
    ros::NodeHandle nh;
    ROS_INFO("Starting Nomdp Contact Execution Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    ros::ServiceClient robot_control_service = nh.serviceClient<nomdp_planning::Simple6dofRobotMove>("simple_6dof_robot_move");
    peg_in_hole_env_linked(display_debug_publisher, robot_control_service);
    return 0;
}
