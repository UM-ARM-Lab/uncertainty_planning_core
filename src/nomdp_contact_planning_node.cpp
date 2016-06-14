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
#include "nomdp_planning/eigenvector3d_robot_helpers.hpp"
#include "nomdp_planning/simple6dof_robot_helpers.hpp"

#ifdef USE_ROS
    #include <ros/ros.h>
    #include <visualization_msgs/MarkerArray.h>
#endif

using namespace nomdp_contact_planning;

struct PLANNER_OPTIONS
{
    // Time limits
    double planner_time_limit;
    // Standard planner control params
    double goal_bias;
    double step_size;
    double goal_probability_threshold;
    double goal_distance_threshold;
    // Distance function control params/weights
    double signature_matching_threshold;
    double distance_clustering_threshold;
    double feasibility_alpha;
    double variance_alpha;
    // Reverse/repeat params
    uint32_t max_attempt_count;
    // Particle/execution limits
    uint32_t num_particles;
    // Execution limits
    uint32_t exec_step_limit;
    uint32_t policy_action_attempt_count;
    // Control flags
    bool use_contact;
    bool use_reverse;
    bool use_spur_actions;
    bool enable_contact_manifold_target_adjustment;
};

#ifdef USE_ROS
std::map<std::string, double> peg_in_hole_env_6dof(const PLANNER_OPTIONS& planner_options, const simple6dof_robot_helpers::ROBOT_CONFIG& robot_config, const std::string& policy_filename, ros::Publisher& display_debug_publisher)
{
#else
std::map<std::string, double> peg_in_hole_env_6dof(const PLANNER_OPTIONS& planner_options, const simple6dof_robot_helpers::ROBOT_CONFIG& robot_config, const std::string& policy_filename)
{
#endif
    // Define the goals of the plan
    Eigen::Matrix<double, 6, 1> start;
    start << 9.0, 9.0, 9.0, 0.0, 0.0, 0.0;
    Eigen::Matrix<double, 6, 1> goal;
    goal << 2.25, 2.25, 0.5, 0.0, 0.0, 0.0;
    // Turn the time limits into durations
    std::chrono::duration<double> planner_time_limit(planner_options.planner_time_limit);
    // Fixed parameters for testing
    const double env_resolution = 0.125;
    const double env_min_x = 0.0 + (env_resolution);
    const double env_max_x = 10.0 - (env_resolution);
    const double env_min_y = 0.0 + (env_resolution);
    const double env_max_y = 10.0 - (env_resolution);
    const double env_min_z = 0.0 + (env_resolution);
    const double env_max_z = 10.0 - (env_resolution);
    // Make the sampler
    simple6dof_robot_helpers::Simple6DOFBaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
    // Make the robot geometry
    EigenHelpers::VectorVector3d robot_points;
    const std::vector<double> x_pos = {-0.1875, -0.0625, 0.0625, 0.1875};
    const std::vector<double> y_pos = {-0.1875, -0.0625, 0.0625, 0.1875};
    const std::vector<double> z_pos = {-0.4375, -0.3125, -0.1875, -0.0625, 0.0625, 0.1875, 0.3125, 0.4375};
    for (size_t xpdx = 0; xpdx < x_pos.size(); xpdx++)
    {
        for (size_t ypdx = 0; ypdx < y_pos.size(); ypdx++)
        {
            for (size_t zpdx = 0; zpdx < z_pos.size(); zpdx++)
            {
                robot_points.push_back(Eigen::Vector3d(x_pos[xpdx], y_pos[ypdx], z_pos[zpdx]));
            }
        }
    }
    // Make the actual robot
    Eigen::Matrix<double, 6, 1> initial_config = Eigen::Matrix<double, 6, 1>::Zero();
    simple6dof_robot_helpers::Simple6DOFRobot robot(robot_points, initial_config, robot_config);
    // Build the planning space
    NomdpPlanningSpace<simple6dof_robot_helpers::Simple6DOFRobot, simple6dof_robot_helpers::Simple6DOFBaseSampler, Eigen::Matrix<double, 6, 1>, simple6dof_robot_helpers::EigenMatrixD61Serializer, simple6dof_robot_helpers::Simple6DOFAverager, simple6dof_robot_helpers::Simple6DOFDistancer, simple6dof_robot_helpers::Simple6DOFDimDistancer, simple6dof_robot_helpers::Simple6DOFInterpolator, std::allocator<Eigen::Matrix<double, 6, 1>>, std::mt19937_64> planning_space(false, planner_options.num_particles, planner_options.step_size, planner_options.goal_distance_threshold, planner_options.goal_probability_threshold, planner_options.signature_matching_threshold, planner_options.distance_clustering_threshold, planner_options.feasibility_alpha, planner_options.variance_alpha, robot, sampler, "peg_in_hole", env_resolution);
    // Plan & execute
#ifdef USE_ROS
    auto planner_result = planning_space.Plan(start, goal, planner_options.goal_bias, planner_time_limit, planner_options.max_attempt_count, planner_options.policy_action_attempt_count, planner_options.use_contact, planner_options.use_reverse, planner_options.use_spur_actions, planner_options.enable_contact_manifold_target_adjustment, display_debug_publisher);
#else
    auto planner_result = planning_space.Plan(start, goal, planner_options.goal_bias, planner_time_limit, planner_options.max_attempt_count, planner_options.policy_action_attempt_count, planner_options.use_contact, planner_options.use_reverse, planner_options.use_spur_actions, planner_options.enable_contact_manifold_target_adjustment);
#endif
    const auto& policy = planner_result.first;
    const std::map<std::string, double> planner_stats = planner_result.second;
    const double p_goal_reached = planner_stats.at("P(goal reached)");
    if (p_goal_reached >= planner_options.goal_probability_threshold)
    {
        std::cout << "Planner reached goal, saving & loading policy" << std::endl;
        // Save the policy
        assert(planning_space.SavePolicy(policy, policy_filename));
        const auto loaded_policy = planning_space.LoadPolicy(policy_filename);
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
    return planner_stats;
}

int main(int argc, char** argv)
{
    const double env_resolution = 0.125;
#ifdef USE_ROS
    ros::init(argc, argv, "nomdp_contact_planning_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Nomdp Contact Planning Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    const uint32_t num_particles = 50u;
    const double goal_probability_threshold = 0.8;
    const double goal_bias = 0.1;
    const double signature_matching_threshold = 0.125;
    const double distance_clustering_threshold = 8.0 * env_resolution;
    const double feasibility_alpha = 0.25;
    const double variance_alpha = 0.25;
    const uint32_t max_attempt_count = 50u;
    const bool use_spur_actions = true;
    const uint32_t num_repeats = 1u;
    std::string log_filename;
    std::string policy_filename;
    nhp.param(std::string("log_filename"), log_filename, std::string("/tmp/nomdp_planner_log.txt"));
    nhp.param(std::string("policy_filename"), policy_filename, std::string("/tmp/nomdp_planner_policy.policy"));
#else
    const uint32_t num_repeats = (argc > 1) ? (uint32_t)atoi(argv[1]) : 1u;
    const std::string log_filename = (argc > 2) ? std::string(argv[2]) : std::string("/tmp/nomdp_planner_log.txt");
    const std::string policy_filename = (argc > 3) ? std::string(argv[3]) : std::string("/tmp/nomdp_planner_policy.policy");
    const uint32_t num_particles = (argc > 4) ? (uint32_t)atoi(argv[4]) : 50u;
    const double goal_probability_threshold = (argc > 5) ? atof(argv[5]) : 0.51;
    const double goal_bias = (argc > 6) ? atof(argv[6]) : 0.1;
    const double signature_matching_threshold = (argc > 7) ? atof(argv[7]) : 0.125;
    const double distance_clustering_threshold = (argc > 8) ? atof(argv[8]) : 8.0 * env_resolution;
    const double feasibility_alpha = (argc > 9) ? atof(argv[9]) : 0.75;
    const double variance_alpha = (argc > 10) ? atof(argv[10]) : 0.75;
    const uint32_t max_attempt_count = (argc > 11) ? (uint32_t)atoi(argv[11]) : 50u;
    const bool use_spur_actions = (argc > 12) ? (bool)atoi(argv[12]) : true;
#endif
    // Planner params
    PLANNER_OPTIONS planner_options;
    planner_options.planner_time_limit = 600.0;
    planner_options.exec_step_limit = 100u;
    planner_options.goal_bias = goal_bias;
    planner_options.step_size = 24.0 * env_resolution;
    planner_options.goal_probability_threshold = goal_probability_threshold;
    planner_options.goal_distance_threshold = 1.0 * env_resolution;
    planner_options.signature_matching_threshold = signature_matching_threshold;
    planner_options.distance_clustering_threshold = distance_clustering_threshold;
    planner_options.feasibility_alpha = feasibility_alpha;
    planner_options.variance_alpha = variance_alpha;
    planner_options.max_attempt_count = max_attempt_count;
    planner_options.policy_action_attempt_count = 10u;
    planner_options.num_particles = num_particles;
    planner_options.use_contact = true;
    planner_options.use_reverse = true;
    planner_options.use_spur_actions = use_spur_actions;
    planner_options.enable_contact_manifold_target_adjustment = false;
    // Fixed controller params
    const double kp = 0.1;
    const double ki = 0.0;
    const double kd = 0.01;
    const double i_clamp = 0.0;
    const double velocity_limit = env_resolution * 2.0;
    const double max_sensor_noise = env_resolution * 0.01;
    const double max_actuator_noise = env_resolution * 1.0; //1.0;
    const simple6dof_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, (velocity_limit * 0.125), (max_sensor_noise * 0.125), (max_actuator_noise * 0.125));
    // Run the planner
    std::vector<std::map<std::string, double>> planner_performance(num_repeats);
    for (size_t idx = 0; idx < planner_performance.size(); idx++)
    {
    #ifdef USE_ROS
        planner_performance[idx] = peg_in_hole_env_6dof(planner_options, robot_config, policy_filename, display_debug_publisher);
    #else
        planner_performance[idx] = peg_in_hole_env_6dof(planner_options, robot_config, policy_filename);
    #endif
    }
    // Print out the results & save them to the log file
    const std::string log_results = PrettyPrint::PrettyPrint(planner_performance, false, "\n");
    std::cout << "Planner results for " << num_particles << " particles:\nAll: " << log_results << std::endl;
    std::ofstream log_file(log_filename, std::ios_base::out | std::ios_base::app);
    if (!log_file.is_open())
    {
        std::cerr << "\x1b[31;1m Unable to create folder/file to log to: " << log_filename << "\x1b[37m \n";
        throw std::invalid_argument( "Log filename must be write-openable" );
    }
    log_file << log_results << std::endl;
    log_file.close();
    return 0;
}
