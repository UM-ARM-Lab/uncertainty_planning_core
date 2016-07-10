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
#include "nomdp_planning/simplese3_robot_helpers.hpp"
#include "se3_common_config.hpp"

#ifdef USE_ROS
    #include <ros/ros.h>
    #include <visualization_msgs/MarkerArray.h>
    #include <geometry_msgs/PoseStamped.h>
    #include <nomdp_planning/Simple6dofRobotMove.h>
#endif

using namespace nomdp_contact_planning;

#ifdef USE_ROS
inline EigenHelpers::VectorAffine3d move_robot(const Eigen::Affine3d& target_transform, const bool reset, ros::ServiceClient& robot_control_service)
{
    const geometry_msgs::PoseStamped target_pose = EigenHelpersConversions::EigenAffine3dToGeometryPoseStamped(target_transform, "world");
    // Put together service call
    nomdp_planning::Simple6dofRobotMove::Request req;
    req.target = target_pose;
    req.time_limit = ros::Duration(10.0);
    if (reset)
    {
        std::cout << "Resetting robot to transform: " << PrettyPrint::PrettyPrint(target_transform) << std::endl;
        req.reset = true;
    }
    else
    {
        std::cout << "Commanding robot to transform: " << PrettyPrint::PrettyPrint(target_transform) << std::endl;
        req.reset = false;
    }
    nomdp_planning::Simple6dofRobotMove::Response res;
    // Call service
    try
    {
        robot_control_service.call(req, res);
    }
    catch (...)
    {
        ROS_ERROR("Move service failed");
    }
    // Unpack result
    const std::vector<geometry_msgs::PoseStamped>& poses = res.trajectory;
    EigenHelpers::VectorAffine3d transforms(poses.size());
    for (size_t idx = 0; idx < poses.size(); idx++)
    {
        transforms[idx] = EigenHelpersConversions::GeometryPoseToEigenAffine3d(poses[idx].pose);
    }
    std::cout << "Reached transform: " << PrettyPrint::PrettyPrint(transforms.back()) << std::endl;
    return transforms;
}
#endif

void peg_in_hole_env_se3(ros::Publisher& display_debug_publisher, ros::ServiceClient& robot_control_service)
{
    const common_config::OPTIONS options = se3_common_config::GetOptions(common_config::OPTIONS::EXECUTION);
    std::cout << PrettyPrint::PrettyPrint(options) << std::endl;
    const std::pair<Eigen::Affine3d, Eigen::Affine3d> start_and_goal = se3_common_config::GetStartAndGoal();
    const simplese3_robot_helpers::SimpleSE3BaseSampler sampler = se3_common_config::GetSampler();
    const simplese3_robot_helpers::ROBOT_CONFIG robot_config = se3_common_config::GetDefaultRobotConfig(options);
    const simplese3_robot_helpers::SimpleSE3Robot robot = se3_common_config::GetRobot(robot_config);
    NomdpPlanningSpace<simplese3_robot_helpers::SimpleSE3Robot, simplese3_robot_helpers::SimpleSE3BaseSampler, Eigen::Affine3d, simplese3_robot_helpers::EigenAffine3dSerializer, simplese3_robot_helpers::SimpleSE3Averager, simplese3_robot_helpers::SimpleSE3Distancer, simplese3_robot_helpers::SimpleSE3DimDistancer, simplese3_robot_helpers::SimpleSE3Interpolator, Eigen::aligned_allocator<Eigen::Affine3d>, std::mt19937_64> planning_space(options.clustering_type, false, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, robot, sampler, "peg_in_hole", options.environment_resolution);
    // Load the policy
    try
    {
        auto policy = planning_space.LoadPolicy(options.planned_policy_file);
        policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
        std::map<std::string, double> complete_policy_stats;
    #ifndef FORCE_DEBUG
        std::cout << "Press ENTER to simulate policy..." << std::endl;
        std::cin.get();
    #endif
        const auto policy_simulation_results = planning_space.SimulateExectionPolicy(policy, start_and_goal.first, start_and_goal.second, options.num_policy_simulations, options.max_exec_actions, options.enable_contact_manifold_target_adjustment, display_debug_publisher, false, 0.001, false);
        const std::map<std::string, double> policy_simulation_stats = policy_simulation_results.second.first;
        const std::vector<int64_t> policy_simulation_step_counts = policy_simulation_results.second.second.first;
        std::cout << "Policy simulation success: " << PrettyPrint::PrettyPrint(policy_simulation_stats) << std::endl;
        complete_policy_stats.insert(policy_simulation_stats.begin(), policy_simulation_stats.end());
    #ifndef FORCE_DEBUG
        std::cout << "Press ENTER to execute policy..." << std::endl;
        std::cin.get();
    #endif
        std::function<EigenHelpers::VectorAffine3d(const Eigen::Affine3d&, const bool)> robot_execution_fn = [&] (const Eigen::Affine3d& target_configuration, const bool reset) { return move_robot(target_configuration, reset, robot_control_service); };
        const auto policy_execution_results = planning_space.ExecuteExectionPolicy(policy, start_and_goal.first, start_and_goal.second, robot_execution_fn, options.num_policy_executions, options.max_exec_actions, display_debug_publisher, false, 0.001, false);
        const std::map<std::string, double> policy_execution_stats = policy_execution_results.second.first;
        const std::vector<int64_t> policy_execution_step_counts = policy_execution_results.second.second.first;
        const std::vector<double> policy_execution_times = policy_execution_results.second.second.second;
        std::cout << "Policy execution success: " << PrettyPrint::PrettyPrint(policy_execution_stats) << std::endl;
        complete_policy_stats.insert(policy_execution_stats.begin(), policy_execution_stats.end());
        // Save the executed policy
        planning_space.SavePolicy(policy_execution_results.first, options.executed_policy_file);
        // Print out the results & save them to the log file
        const std::string log_results = "++++++++++\n" + PrettyPrint::PrettyPrint(options) + "\nRESULTS:\n" + PrettyPrint::PrettyPrint(complete_policy_stats, false, "\n") + "\nSimulation step counts: " + PrettyPrint::PrettyPrint(policy_simulation_step_counts) + "\nExecution step counts: " + PrettyPrint::PrettyPrint(policy_execution_step_counts) + "\nExecution times: " + PrettyPrint::PrettyPrint(policy_execution_times);
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
    catch (...)
    {
        std::cout << "!!! Policy file does not exist, skipping !!!" << std::endl;
        const std::string log_results = "++++++++++\n" + PrettyPrint::PrettyPrint(options) + "\nRESULTS:\n(Execution) Policy success: 0\n(Simulation) Policy success: 0\n(Simulation) Policy successful simulator resolves: 0\n(Simulation) Policy unsuccessful simulator environment resolves: 0\n(Simulation) Policy unsuccessful simulator resolves: 0\n(Simulation) Policy unsuccessful simulator self-collision resolves: 0\nSimulation step counts: -1\nExecution step counts: -1\nExecution times: -0.0";
        std::ofstream log_file(options.policy_log_file, std::ios_base::out | std::ios_base::app);
        if (!log_file.is_open())
        {
            std::cerr << "\x1b[31;1m Unable to create folder/file to log to: " << options.policy_log_file << "\x1b[37m \n";
            throw std::invalid_argument( "Log filename must be write-openable" );
        }
        log_file << log_results << std::endl;
        log_file.close();
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "se3_contact_execution_node");
    ros::NodeHandle nh;
    ROS_INFO("Starting Nomdp Contact Execution Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    ros::ServiceClient robot_control_service = nh.serviceClient<nomdp_planning::Simple6dofRobotMove>("simple_6dof_robot_move");
    peg_in_hole_env_se3(display_debug_publisher, robot_control_service);
    return 0;
}
