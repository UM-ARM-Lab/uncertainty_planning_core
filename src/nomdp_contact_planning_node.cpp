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
    #include <sensor_msgs/Image.h>
    #include <geometry_msgs/PoseStamped.h>
    #include <nomdp_planning/Simple6dofRobotMove.h>
#endif

using namespace nomdp_contact_planning;

/*
double contact_test_env(int argc, char** argv, const uint32_t num_particles, const uint32_t num_threads)
{
#ifdef USE_ROS
    ros::init(argc, argv, "nomdp_contact_planning_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Nomdp Contact Planning Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    //ros::Publisher display_image_publisher = nh.advertise<sensor_msgs::Image>("nomdp_debug_display_image", 1, true);
#else
    UNUSED(argc);
    UNUSED(argv);
#endif
    // Initialize the planning space
    double planner_time_limit = 600.0;
    double env_resolution = 0.125;
    double step_size = 10 * env_resolution;
    double goal_distance_threshold = 1.0 * env_resolution;
    double goal_probability_threshold = 0.51;
    double signature_matching_threshold = 0.125;
    double env_min_x = 0.0 + (env_resolution);
    double env_max_x = 10.0 - (env_resolution);
    double env_min_y = 0.0 + (env_resolution);
    double env_max_y = 10.0 - (env_resolution);
    double env_min_z = 0.0 + (env_resolution);
    double env_max_z = 10.0 - (env_resolution);
#ifdef USE_6DOF
    simple6dof_robot_helpers::Simple6DOFBaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
#else
    eigenvector3d_robot_helpers::EigenVector3dBaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
#endif
    double kp = 0.1;
    double ki = 0.0;
    double kd = 0.01;
    double i_clamp = 0.0;
    double velocity_limit = env_resolution * 2.0;
    double max_sensor_noise = env_resolution * 0.01;
    double max_actuator_noise = env_resolution * 1.0; //1.0;
    double feasibility_alpha = 0.75;
    double variance_alpha = 0.75;
    // Make the robot geometry
    EigenHelpers::VectorVector3d robot_points = {Eigen::Vector3d(0.25, 0.0, 0.0), Eigen::Vector3d(0.125, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(-0.125, 0.0, 0.0), Eigen::Vector3d(-0.25, 0.0, 0.0)};
    // Make the actual robot
#ifdef USE_6DOF
    simple6dof_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, (velocity_limit * 0.125), (max_sensor_noise * 0.125), (max_actuator_noise * 0.125));
    Eigen::Matrix<double, 6, 1> initial_config = Eigen::Matrix<double, 6, 1>::Zero();
    simple6dof_robot_helpers::Simple6DOFRobot robot(robot_points, initial_config, robot_config);
#else
    eigenvector3d_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    eigenvector3d_robot_helpers::SimpleEigenVector3dRobot robot(robot_points, Eigen::Vector3d::Zero(), robot_config);
#endif
    bool use_contact = true;
    bool use_reverse = true;
    bool use_spur_actions = true;
#ifdef USE_6DOF
    NomdpPlanningSpace<simple6dof_robot_helpers::Simple6DOFRobot, simple6dof_robot_helpers::Simple6DOFBaseSampler, Eigen::Matrix<double, 6, 1>, simple6dof_robot_helpers::Simple6DOFAverager, simple6dof_robot_helpers::Simple6DOFDistancer, simple6dof_robot_helpers::Simple6DOFDimDistancer, simple6dof_robot_helpers::Simple6DOFInterpolator, std::allocator<Eigen::Matrix<double, 6, 1>>, std::mt19937_64> planning_space(false, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, "nested_corners", env_resolution, num_threads);
#else
    NomdpPlanningSpace<eigenvector3d_robot_helpers::SimpleEigenVector3dRobot, eigenvector3d_robot_helpers::EigenVector3dBaseSampler, Eigen::Vector3d, eigenvector3d_robot_helpers::EigenVector3dAverager, eigenvector3d_robot_helpers::EigenVector3dDistancer, eigenvector3d_robot_helpers::EigenVector3dDimDistancer, eigenvector3d_robot_helpers::EigenVector3dInterpolator, Eigen::aligned_allocator<Eigen::Vector3d>, std::mt19937_64> planning_space(false, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, "nested_corners", env_resolution, num_threads);
#endif
    // Define the goals of the plan
#ifdef USE_6DOF
    Eigen::Matrix<double, 6, 1> start;
    start << 9.0, 9.0, 9.0, 0.0, 0.0, 0.0;
    Eigen::Matrix<double, 6, 1> goal;
    goal << 1.75, 1.625, 1.625, 0.0, 0.0, 0.0;
#else
    Eigen::Vector3d start(9.0, 9.0, 9.0);
    Eigen::Vector3d goal(1.875, 1.625, 1.625);
#endif
    double goal_bias = 0.1;
    std::chrono::duration<double> time_limit(planner_time_limit);
    // Plan & execute
    std::chrono::duration<double> exec_time_limit(5.0);
#ifdef USE_ROS
    auto planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, 10u, use_contact, use_reverse, use_spur_actions, display_debug_publisher);
    std::cout << "Press ENTER to simulate policy..." << std::endl;
    std::cin.get();
    double policy_success = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit, display_debug_publisher, true, false);
#else
    auto planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, 10u, use_contact, use_reverse, use_spur_actions);
    double policy_success = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit);
#endif
    //std::cout << "Policy execution success: " << policy_success << std::endl;
    return policy_success;
}

std::pair<double, double> inset_peg_in_hole_env(int argc, char** argv, const uint32_t num_particles, const uint32_t num_threads)
{
#ifdef USE_ROS
    ros::init(argc, argv, "nomdp_contact_planning_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Nomdp Contact Planning Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    //ros::Publisher display_image_publisher = nh.advertise<sensor_msgs::Image>("nomdp_debug_display_image", 1, true);
#else
    UNUSED(argc);
    UNUSED(argv);
#endif
    // Initialize the planning space
    double planner_time_limit = 240.0;
    double env_resolution = 0.125;
    double step_size = 10 * env_resolution;
    double goal_distance_threshold = 1.0 * env_resolution;
    double goal_probability_threshold = 0.51;
    double signature_matching_threshold = 0.125;
    double env_min_x = 0.0 + (env_resolution);
    double env_max_x = 10.0 - (env_resolution);
    double env_min_y = 0.0 + (env_resolution);
    double env_max_y = 10.0 - (env_resolution);
    double env_min_z = 0.0 + (env_resolution);
    double env_max_z = 10.0 - (env_resolution);
#ifdef USE_6DOF
    simple6dof_robot_helpers::Simple6DOFBaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
#else
    eigenvector3d_robot_helpers::EigenVector3dBaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
#endif
    double kp = 0.1;
    double ki = 0.0;
    double kd = 0.01;
    double i_clamp = 0.0;
    double velocity_limit = env_resolution * 2.0;
    double max_sensor_noise = env_resolution * 0.1;
    double max_actuator_noise = env_resolution * 1.0; //1.0;
    double feasibility_alpha = 0.75;
    double variance_alpha = 0.75;
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
#ifdef USE_6DOF
    simple6dof_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, (velocity_limit * 0.125), (max_sensor_noise * 0.125), (max_actuator_noise * 0.125));
    Eigen::Matrix<double, 6, 1> initial_config = Eigen::Matrix<double, 6, 1>::Zero();
    simple6dof_robot_helpers::Simple6DOFRobot robot(robot_points, initial_config, robot_config);
#else
    eigenvector3d_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    eigenvector3d_robot_helpers::SimpleEigenVector3dRobot robot(robot_points, Eigen::Vector3d::Zero(), robot_config);
#endif
    bool use_contact = true;
    bool use_reverse = true;
    bool use_spur_actions = true;
#ifdef USE_6DOF
    NomdpPlanningSpace<simple6dof_robot_helpers::Simple6DOFRobot, simple6dof_robot_helpers::Simple6DOFBaseSampler, Eigen::Matrix<double, 6, 1>, simple6dof_robot_helpers::Simple6DOFAverager, simple6dof_robot_helpers::Simple6DOFDistancer, simple6dof_robot_helpers::Simple6DOFDimDistancer, simple6dof_robot_helpers::Simple6DOFInterpolator, std::allocator<Eigen::Matrix<double, 6, 1>>, std::mt19937_64> planning_space(false, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, "inset_peg_in_hole", env_resolution, num_threads);
#else
    NomdpPlanningSpace<eigenvector3d_robot_helpers::SimpleEigenVector3dRobot, eigenvector3d_robot_helpers::EigenVector3dBaseSampler, Eigen::Vector3d, eigenvector3d_robot_helpers::EigenVector3dAverager, eigenvector3d_robot_helpers::EigenVector3dDistancer, eigenvector3d_robot_helpers::EigenVector3dDimDistancer, eigenvector3d_robot_helpers::EigenVector3dInterpolator, Eigen::aligned_allocator<Eigen::Vector3d>, std::mt19937_64> planning_space(false, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, "inset_peg_in_hole", env_resolution, num_threads);
#endif
    // Define the goals of the plan
#ifdef USE_6DOF
    Eigen::Matrix<double, 6, 1> start;
    start << 9.0, 9.0, 9.0, 0.0, 0.0, 0.0;
    Eigen::Matrix<double, 6, 1> goal;
    goal << 2.25, 2.25, 0.5, 0.0, 0.0, 0.0;
#else
    Eigen::Vector3d start(9.0, 9.0, 9.0);
    Eigen::Vector3d goal(1.875, 1.625, 1.625);
#endif
    double goal_bias = 0.1;
    std::chrono::duration<double> time_limit(planner_time_limit);
    // Plan & execute
    std::chrono::duration<double> exec_time_limit(5.0);
#ifdef USE_ROS
    auto planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, 10u, use_contact, use_reverse, use_spur_actions, display_debug_publisher);
    //std::cout << "Press ENTER to simulate policy..." << std::endl;
    //std::cin.get();
    //double policy_success = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit, display_debug_publisher, true);
#else
    auto planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, 10u, use_contact, use_reverse, use_spur_actions);
    //double policy_success = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit);
#endif
    //std::cout << "Policy execution success: " << policy_success << std::endl;
    //return policy_success;
    return std::pair<double, double>(planner_result.second["P(goal reached)"], planner_result.second["solutions"]);
}
*/

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
    double feasibility_alpha;
    double variance_alpha;
    // Reverse/repeat params
    uint32_t max_attempt_count;
    // Particle/execution limits
    uint32_t num_particles;
    uint32_t num_policy_simulations;
    uint32_t num_policy_executions;
    // Execution limits
    uint32_t exec_step_limit;
    // Control flags
    bool use_contact;
    bool use_reverse;
    bool use_spur_actions;
};

inline Eigen::Matrix<double, 6, 1> ConvertRobotPoseToConfiguration(const Eigen::Affine3d& pose)
{
    const Eigen::Vector3d position = pose.translation();
    const Eigen::Vector3d rotation = EigenHelpers::EulerAnglesFromAffine3d(pose);
    Eigen::Matrix<double, 6, 1> config;
    config << position, rotation;
    return config;
}

inline double execution_distance_fn(const Eigen::Matrix<double, 6, 1>& q1, const Eigen::Matrix<double, 6, 1>& q2, simple6dof_robot_helpers::Simple6DOFRobot robot)
{
    robot.UpdatePosition(q1);
    const Eigen::Affine3d q1t = robot.GetLinkTransform("robot");
    robot.UpdatePosition(q2);
    const Eigen::Affine3d q2t = robot.GetLinkTransform("robot");
    return (EigenHelpers::Distance(q1t, q2t, 0.5) * 2.0);
}

#ifdef USE_ROS
inline std::vector<Eigen::Matrix<double, 6, 1>> move_robot(const Eigen::Matrix<double, 6, 1>& target_configuration, simple6dof_robot_helpers::Simple6DOFRobot robot, ros::ServiceClient& robot_control_service)
{
    std::cout << "Commanding robot to configuration: " << PrettyPrint::PrettyPrint(target_configuration) << std::endl;
    robot.UpdatePosition(target_configuration);
    const Eigen::Affine3d target_transform = robot.GetLinkTransform("robot");
    std::cout << "Commanding robot to transform: " << PrettyPrint::PrettyPrint(target_transform) << std::endl;
    const geometry_msgs::PoseStamped target_pose = EigenHelpersConversions::EigenAffine3dToGeometryPoseStamped(target_transform, "world");
    // Put together service call
    nomdp_planning::Simple6dofRobotMove::Request req;
    req.target = target_pose;
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
    // Convert the result back to robot configurations (THIS IS A TERRIBLE HACK)
    std::vector<Eigen::Matrix<double, 6, 1>> configurations(transforms.size());
    for (size_t idx = 0; idx < transforms.size(); idx++)
    {
        configurations[idx] = ConvertRobotPoseToConfiguration(transforms[idx]);
    }
    std::cout << "Reached configuration: " << PrettyPrint::PrettyPrint(configurations.back()) << std::endl;
    return configurations;
}
#endif

#ifdef USE_ROS
std::map<std::string, double> peg_in_hole_env_6dof(const PLANNER_OPTIONS& planner_options, const simple6dof_robot_helpers::ROBOT_CONFIG& robot_config, ros::Publisher& display_debug_publisher, ros::ServiceClient& robot_control_service)
{
#else
std::map<std::string, double> peg_in_hole_env_6dof(const PLANNER_OPTIONS& planner_options, const simple6dof_robot_helpers::ROBOT_CONFIG& robot_config)
{
#endif
    // Define the goals of the plan
    Eigen::Matrix<double, 6, 1> start;
    start << 9.0, 9.0, 9.0, 0.0, 0.0, 0.0;
    Eigen::Matrix<double, 6, 1> goal;
    goal << 2.25, 2.25, 0.5, 0.0, 0.0, 0.0;
    double goal_bias = 0.1;
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
    NomdpPlanningSpace<simple6dof_robot_helpers::Simple6DOFRobot, simple6dof_robot_helpers::Simple6DOFBaseSampler, Eigen::Matrix<double, 6, 1>, simple6dof_robot_helpers::EigenMatrixD61Serializer, simple6dof_robot_helpers::Simple6DOFAverager, simple6dof_robot_helpers::Simple6DOFDistancer, simple6dof_robot_helpers::Simple6DOFDimDistancer, simple6dof_robot_helpers::Simple6DOFInterpolator, std::allocator<Eigen::Matrix<double, 6, 1>>, std::mt19937_64> planning_space(false, planner_options.num_particles, planner_options.step_size, planner_options.goal_distance_threshold, planner_options.goal_probability_threshold, planner_options.signature_matching_threshold, planner_options.feasibility_alpha, planner_options.variance_alpha, robot, sampler, "peg_in_hole", env_resolution);
    // Plan & execute
#ifdef USE_ROS
    auto planner_result = planning_space.Plan(start, goal, goal_bias, planner_time_limit, planner_options.max_attempt_count, planner_options.use_contact, planner_options.use_reverse, planner_options.use_spur_actions, display_debug_publisher);
    std::map<std::string, double> planner_stats = planner_result.second;
    std::cout << "Press ENTER to simulate policy..." << std::endl;
    std::cin.get();
    const auto policy_stats = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, planner_options.num_policy_simulations, planner_options.exec_step_limit); //, display_debug_publisher, true, false);
    std::cout << "Policy simulation success: " << PrettyPrint::PrettyPrint(policy_stats, true) << std::endl;
    planner_stats["Policy simulation success"] = policy_stats.first;
    planner_stats["Policy simulation successful resolves"] = policy_stats.second.first;
    planner_stats["Policy simulation unsuccessful resolves"] = policy_stats.second.second;
    std::cout << "Press ENTER to execute policy..." << std::endl;
    std::cin.get();
    std::function<std::vector<Eigen::Matrix<double, 6, 1>>(const Eigen::Matrix<double, 6, 1>&)> robot_execution_fn = [&] (const Eigen::Matrix<double, 6, 1>& target_configuration) { return move_robot(target_configuration, robot, robot_control_service); };
    std::function<double(const Eigen::Matrix<double, 6, 1>&, const Eigen::Matrix<double, 6, 1>&)> exec_dist_fn = [&] (const Eigen::Matrix<double, 6, 1>& q1, const Eigen::Matrix<double, 6, 1>& q2) { return execution_distance_fn(q1, q2, robot); };
    const double execution_success = planning_space.ExecuteExectionPolicy(planner_result.first, start, goal, robot_execution_fn, exec_dist_fn, planner_options.num_policy_executions, planner_options.exec_step_limit);
    std::cout << "Policy execution success: " << PrettyPrint::PrettyPrint(execution_success, true) << std::endl;
    planner_stats["Policy execution success"] = execution_success;
#else
    auto planner_result = planning_space.Plan(start, goal, goal_bias, planner_time_limit, planner_options.max_attempt_count, planner_options.use_contact, planner_options.use_reverse, planner_options.use_spur_actions);
    std::map<std::string, double> planner_stats = planner_result.second;
    std::pair<double, std::pair<double, double>> policy_exec_stats = std::pair<double, std::pair<double, double>>(0.0, std::make_pair(0.0, 0.0));
    if (planner_stats["P(goal reached)"] > 0.0)
    {
        policy_exec_stats = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, planner_options.num_policy_simulations, planner_options.exec_step_limit);
    }
    else
    {
        std::cout << "Planner did not reach the goal, skipping policy execution" << std::endl;
    }
    const auto policy_stats = policy_exec_stats;
    std::cout << "Policy execution success: " << PrettyPrint::PrettyPrint(policy_stats, true) << std::endl;
    planner_stats["Policy success"] = policy_stats.first;
    planner_stats["Policy simulation successful resolves"] = policy_stats.second.first;
    planner_stats["Policy simulation unsuccessful resolves"] = policy_stats.second.second;
    // Run some checks on the serialization system
    const std::string planner_tree_test_file("/tmp/nomdp_planner_tree.tree");
    const auto& planner_tree = planning_space.GetTreeImmutable();
    assert(planning_space.SavePlannerTree(planner_tree, planner_tree_test_file));
    const auto loaded_planner_tree = planning_space.LoadPlannerTree(planner_tree_test_file);
    assert(planner_tree.size() == loaded_planner_tree.size());
    const std::string planner_policy_test_file("/tmp/nomdp_planner_policy.policy");
    const auto& policy = planner_result.first;
    assert(planning_space.SavePolicy(policy, planner_policy_test_file));
    const auto loaded_planner_policy = planning_space.LoadPolicy(planner_policy_test_file);
    assert(policy.GetRawPreviousIndexMap().size() == loaded_planner_policy.GetRawPreviousIndexMap().size());
#endif
    return planner_stats;
}

/*
#ifdef USE_ROS
std::map<std::string, double> peg_in_hole_env_3dof(ros::Publisher& display_debug_publisher, const uint32_t num_particles, const uint32_t num_threads)
{
#else
std::map<std::string, double> peg_in_hole_env_3dof(const uint32_t num_particles, const uint32_t num_threads)
{
#endif
    // Initialize the planning space
    double planner_time_limit = 240.0;
    double env_resolution = 0.125;
    double step_size = 10 * env_resolution;
    double goal_distance_threshold = 1.0 * env_resolution;
    double goal_probability_threshold = 0.51;
    double signature_matching_threshold = 0.125;
    double env_min_x = 0.0 + (env_resolution);
    double env_max_x = 10.0 - (env_resolution);
    double env_min_y = 0.0 + (env_resolution);
    double env_max_y = 10.0 - (env_resolution);
    double env_min_z = 0.0 + (env_resolution);
    double env_max_z = 10.0 - (env_resolution);
    eigenvector3d_robot_helpers::EigenVector3dBaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
    double kp = 0.1;
    double ki = 0.0;
    double kd = 0.01;
    double i_clamp = 0.0;
    double velocity_limit = env_resolution * 2.0;
    double max_sensor_noise = env_resolution * 0.1;
    double max_actuator_noise = env_resolution * 1.0; //1.0;
    double feasibility_alpha = 0.75;
    double variance_alpha = 0.75;
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
    eigenvector3d_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    eigenvector3d_robot_helpers::SimpleEigenVector3dRobot robot(robot_points, Eigen::Vector3d::Zero(), robot_config);
    bool use_contact = true;
    bool use_reverse = true;
    bool use_spur_actions = true;
    NomdpPlanningSpace<eigenvector3d_robot_helpers::SimpleEigenVector3dRobot, eigenvector3d_robot_helpers::EigenVector3dBaseSampler, Eigen::Vector3d, eigenvector3d_robot_helpers::EigenVector3dAverager, eigenvector3d_robot_helpers::EigenVector3dDistancer, eigenvector3d_robot_helpers::EigenVector3dDimDistancer, eigenvector3d_robot_helpers::EigenVector3dInterpolator, Eigen::aligned_allocator<Eigen::Vector3d>, std::mt19937_64> planning_space(false, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, "peg_in_hole", env_resolution);
    // Define the goals of the plan
    Eigen::Vector3d start(9.0, 9.0, 9.0);
    Eigen::Vector3d goal(1.875, 1.625, 1.625);
    double goal_bias = 0.1;
    std::chrono::duration<double> time_limit(planner_time_limit);
    // Plan & execute
    std::chrono::duration<double> exec_time_limit(5.0);
#ifdef USE_ROS
    auto planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, 10u, use_contact, use_reverse, use_spur_actions, display_debug_publisher);
    std::map<std::string, double> planner_stats = planner_result.second;
    std::cout << "Press ENTER to simulate policy..." << std::endl;
    std::cin.get();
    const auto policy_stats = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit, display_debug_publisher, true, false);
#else
    auto planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, 10u, use_contact, use_reverse, use_spur_actions);
    std::map<std::string, double> planner_stats = planner_result.second;
    const auto policy_stats = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit);
#endif
    std::cout << "Policy execution success: " << PrettyPrint::PrettyPrint(policy_stats, true) << std::endl;
    planner_stats["Policy success"] = policy_stats.first;
    planner_stats["Policy simulation successful resolves"] = policy_stats.second.first;
    planner_stats["Policy simulation unsuccessful resolves"] = policy_stats.second.second;
    return planner_stats;
}
*/

void clustering_test()
{
    std::vector<double> test_data = {1.0, 11.2, 13.2, 15.4, 3.0, 4.0, 2.5};
    std::function<double(const double&, const double&)> distance_fn = [](const double& x1, const double& x2) { return fabs(x1 - x2); };
    std::cout << "Clustered: " << PrettyPrint::PrettyPrint(simple_hierarchical_clustering::SimpleHierarchicalClustering::Cluster(test_data, distance_fn, 5.0), true) << std::endl;
    exit(0);
}

int main(int argc, char** argv)
{
#ifdef USE_ROS
    ros::init(argc, argv, "nomdp_contact_planning_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Nomdp Contact Planning Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    ros::ServiceClient robot_control_service = nh.serviceClient<nomdp_planning::Simple6dofRobotMove>("simple_6dof_robot_move");
    const uint32_t num_particles = 50u;
    const uint32_t num_executions = 50u;
    const double goal_probability_threshold = 0.51;
    const double goal_bias = 0.1;
    const double signature_matching_threshold = 0.125;
    const double feasibility_alpha = 0.75;
    const double variance_alpha = 0.75;
    const uint32_t max_attempt_count = 10u;
    const bool use_spur_actions = true;
    const uint32_t num_repeats = 1u;
#else
    const uint32_t num_repeats = (argc > 1) ? (uint32_t)atoi(argv[1]) : 1u;
    const std::string log_filename = (argc > 2) ? std::string(argv[2]) : std::string("/tmp/nomdp_planner_log.txt");
    const uint32_t num_particles = (argc > 3) ? (uint32_t)atoi(argv[3]) : 50u;
    const uint32_t num_executions = (argc > 4) ? (uint32_t)atoi(argv[4]) : 50u;
    const double goal_probability_threshold = (argc > 5) ? atof(argv[5]) : 0.51;
    const double goal_bias = (argc > 6) ? atof(argv[6]) : 0.1;
    const double signature_matching_threshold = (argc > 7) ? atof(argv[7]) : 0.125;
    const double feasibility_alpha = (argc > 8) ? atof(argv[8]) : 0.75;
    const double variance_alpha = (argc > 9) ? atof(argv[9]) : 0.75;
    const uint32_t max_attempt_count = (argc > 10) ? (uint32_t)atoi(argv[10]) : 10u;
    const bool use_spur_actions = (argc > 11) ? (bool)atoi(argv[11]) : true;
#endif
    const double env_resolution = 0.125;
    // Planner params
    PLANNER_OPTIONS planner_options;
    planner_options.planner_time_limit = 240.0;
    planner_options.exec_step_limit = 100u;
    planner_options.goal_bias = goal_bias;
    planner_options.step_size = 10.0 * env_resolution;
    planner_options.goal_probability_threshold = goal_probability_threshold;
    planner_options.goal_distance_threshold = 1.0 * env_resolution;
    planner_options.signature_matching_threshold = signature_matching_threshold;
    planner_options.feasibility_alpha = feasibility_alpha;
    planner_options.variance_alpha = variance_alpha;
    planner_options.max_attempt_count = max_attempt_count;
    planner_options.num_particles = num_particles;
    planner_options.num_policy_simulations = num_executions;
    planner_options.num_policy_executions = num_executions;
    planner_options.use_contact = true;
    planner_options.use_reverse = true;
    planner_options.use_spur_actions = use_spur_actions;
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
        planner_performance[idx] = peg_in_hole_env_6dof(planner_options, robot_config, display_debug_publisher, robot_control_service);
    #else
        planner_performance[idx] = peg_in_hole_env_6dof(planner_options, robot_config);
    #endif
    }
    // Print out the results & save them to the log file
    const std::string log_results = PrettyPrint::PrettyPrint(planner_performance, false, "\n");
    std::cout << "Planner results for " << num_particles << " particles:\nAll: " << log_results << std::endl;
#ifndef USE_ROS
    std::ofstream log_file(log_filename, std::ios_base::out | std::ios_base::app);
    if (!log_file.is_open())
    {
        std::cerr << "\x1b[31;1m Unable to create folder/file to log to: " << log_filename << "\x1b[37m \n";
        throw std::invalid_argument( "Log filename must be write-openable" );
    }
    log_file << log_results << std::endl;
    log_file.close();
#endif
    return 0;
}
