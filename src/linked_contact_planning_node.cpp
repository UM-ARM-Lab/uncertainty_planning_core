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
#include "nomdp_planning/simplelinked_robot_helpers.hpp"
#include "baxter_linked_common_config.hpp"

#ifdef USE_ROS
    #include <ros/ros.h>
    #include <visualization_msgs/MarkerArray.h>
#endif

using namespace nomdp_contact_planning;

#ifdef USE_ROS
void peg_in_hole_env_linked(ros::Publisher& display_debug_publisher)
{
    const common_config::OPTIONS options = linked_common_config::GetOptions(common_config::OPTIONS::PLANNING);
#else
void peg_in_hole_env_linked(int argc, char** argv)
{
    const common_config::OPTIONS options = linked_common_config::GetOptions(argc, argv, common_config::OPTIONS::PLANNING);
#endif
    std::cout << PrettyPrint::PrettyPrint(options) << std::endl;
    const std::pair<linked_common_config::SLC, linked_common_config::SLC> start_and_goal = linked_common_config::GetStartAndGoal();
    const simplelinked_robot_helpers::SimpleLinkedBaseSampler sampler = linked_common_config::GetSampler();
    const simplelinked_robot_helpers::ROBOT_CONFIG robot_config = linked_common_config::GetDefaultRobotConfig(options);
    const Eigen::Affine3d base_transform = linked_common_config::GetBaseTransform();
    const simplelinked_robot_helpers::SimpleLinkedRobot robot = linked_common_config::GetRobot(base_transform, robot_config);
    NomdpPlanningSpace<simplelinked_robot_helpers::SimpleLinkedRobot, simplelinked_robot_helpers::SimpleLinkedBaseSampler, simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfigurationSerializer, simplelinked_robot_helpers::SimpleLinkedAverager, simplelinked_robot_helpers::SimpleLinkedDistancer, simplelinked_robot_helpers::SimpleLinkedDimDistancer, simplelinked_robot_helpers::SimpleLinkedInterpolator, std::allocator<simplelinked_robot_helpers::SimpleLinkedConfiguration>, std::mt19937_64> planning_space(options.clustering_type, false, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, robot, sampler, "baxter_env", options.environment_resolution);
    // Plan
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
#ifdef USE_ROS
    auto planner_result = planning_space.Plan(start_and_goal.first, start_and_goal.second, options.goal_bias, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, options.enable_contact_manifold_target_adjustment, display_debug_publisher);
#else
    auto planner_result = planning_space.Plan(start_and_goal.first, start_and_goal.second, options.goal_bias, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, options.enable_contact_manifold_target_adjustment);
#endif
    const auto& policy = planner_result.first;
    const std::map<std::string, double> planner_stats = planner_result.second;
    const double p_goal_reached = planner_stats.at("P(goal reached)");
    if (p_goal_reached >= options.goal_probability_threshold)
    {
        std::cout << "Planner reached goal, saving & loading policy" << std::endl;
        // Save the policy
        assert(planning_space.SavePolicy(policy, options.planned_policy_file));
        const auto loaded_policy = planning_space.LoadPolicy(options.planned_policy_file);
        std::vector<uint8_t> policy_buffer;
        policy.SerializeSelf(policy_buffer);
        std::vector<uint8_t> loaded_policy_buffer;
        loaded_policy.SerializeSelf(loaded_policy_buffer);
        assert(policy_buffer.size() == loaded_policy_buffer.size());
        for (size_t idx = 0; idx > policy_buffer.size(); idx++)
        {
            const uint8_t policy_buffer_byte = policy_buffer[idx];
            const uint8_t loaded_policy_buffer_byte = loaded_policy_buffer[idx];
            assert(policy_buffer_byte == loaded_policy_buffer_byte);
        }
        assert(policy.GetRawPreviousIndexMap().size() == loaded_policy.GetRawPreviousIndexMap().size());
    }
    else
    {
        std::cout << "Planner failed to reach goal" << std::endl;
    }
    // Print out the results & save them to the log file
    const std::string log_results = PrettyPrint::PrettyPrint(planner_stats, false, "\n");
    std::cout << "Planner results for " << options.num_particles << " particles:\n" << log_results << std::endl;
    std::ofstream log_file(options.planner_log_file, std::ios_base::out | std::ios_base::app);
    if (!log_file.is_open())
    {
        std::cerr << "\x1b[31;1m Unable to create folder/file to log to: " << options.planner_log_file << "\x1b[37m \n";
        throw std::invalid_argument( "Log filename must be write-openable" );
    }
    log_file << log_results << std::endl;
    log_file.close();
}

int main(int argc, char** argv)
{
#ifdef USE_ROS
    ros::init(argc, argv, "linked_contact_planning_node");
    ros::NodeHandle nh;
    ROS_INFO("Starting Nomdp Contact Planning Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    peg_in_hole_env_linked(display_debug_publisher);
#else
    peg_in_hole_env_linked(argc, argv);
#endif
    return 0;
}
