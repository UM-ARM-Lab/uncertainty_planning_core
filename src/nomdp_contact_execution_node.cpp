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
    #include <geometry_msgs/PoseStamped.h>
    #include <nomdp_planning/Simple6dofRobotMove.h>
#endif

using namespace nomdp_contact_planning;

struct PLANNER_OPTIONS
{
    // Standard planner control params
    double step_size;
    double goal_probability_threshold;
    double goal_distance_threshold;
    // Distance function control params/weights
    double signature_matching_threshold;
    double feasibility_alpha;
    double variance_alpha;
    // Particle/execution limits
    uint32_t num_particles;
    uint32_t num_policy_simulations;
    uint32_t num_policy_executions;
    // Execution limits
    uint32_t exec_step_limit;
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
std::map<std::string, double> peg_in_hole_env_6dof(const PLANNER_OPTIONS& planner_options, const simple6dof_robot_helpers::ROBOT_CONFIG& robot_config, const std::string& policy_filename, ros::Publisher& display_debug_publisher, ros::ServiceClient& robot_control_service)
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
    // Load the policy
    const auto policy = planning_space.LoadPolicy(policy_filename);
    std::map<std::string, double> complete_policy_stats;
#ifdef USE_ROS
    std::cout << "Press ENTER to simulate policy..." << std::endl;
    std::cin.get();
    const auto policy_simulation_stats = planning_space.SimulateExectionPolicy(policy, start, goal, planner_options.num_policy_simulations, planner_options.exec_step_limit, display_debug_publisher, true, false);
    std::cout << "Policy simulation success: " << PrettyPrint::PrettyPrint(policy_simulation_stats, true) << std::endl;
    complete_policy_stats["Policy simulation success"] = policy_simulation_stats.first;
    complete_policy_stats["Policy simulation successful resolves"] = policy_simulation_stats.second.first;
    complete_policy_stats["Policy simulation unsuccessful resolves"] = policy_simulation_stats.second.second;
    std::cout << "Press ENTER to execute policy..." << std::endl;
    std::cin.get();
    std::function<std::vector<Eigen::Matrix<double, 6, 1>>(const Eigen::Matrix<double, 6, 1>&)> robot_execution_fn = [&] (const Eigen::Matrix<double, 6, 1>& target_configuration) { return move_robot(target_configuration, robot, robot_control_service); };
    std::function<double(const Eigen::Matrix<double, 6, 1>&, const Eigen::Matrix<double, 6, 1>&)> exec_dist_fn = [&] (const Eigen::Matrix<double, 6, 1>& q1, const Eigen::Matrix<double, 6, 1>& q2) { return execution_distance_fn(q1, q2, robot); };
    const double execution_success = planning_space.ExecuteExectionPolicy(policy, start, goal, robot_execution_fn, exec_dist_fn, planner_options.num_policy_executions, planner_options.exec_step_limit);
    std::cout << "Policy execution success: " << PrettyPrint::PrettyPrint(execution_success, true) << std::endl;
    complete_policy_stats["Policy execution success"] = execution_success;
#else
    std::pair<double, std::pair<double, double>> policy_exec_stats = planning_space.SimulateExectionPolicy(policy, start, goal, planner_options.num_policy_simulations, planner_options.exec_step_limit);
    const auto policy_stats = policy_exec_stats;
    std::cout << "Policy execution success: " << PrettyPrint::PrettyPrint(policy_stats, true) << std::endl;
    complete_policy_stats["Policy success"] = policy_stats.first;
    complete_policy_stats["Policy simulation successful resolves"] = policy_stats.second.first;
    complete_policy_stats["Policy simulation unsuccessful resolves"] = policy_stats.second.second;
#endif
    return complete_policy_stats;
}

int main(int argc, char** argv)
{
#ifdef USE_ROS
    ros::init(argc, argv, "nomdp_contact_execution_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Nomdp Contact Execution Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    ros::ServiceClient robot_control_service = nh.serviceClient<nomdp_planning::Simple6dofRobotMove>("simple_6dof_robot_move");
    const uint32_t num_particles = 50u;
    const uint32_t num_executions = 50u;
    const double goal_probability_threshold = 0.51;
    const double signature_matching_threshold = 0.125;
    const double feasibility_alpha = 0.75;
    const double variance_alpha = 0.75;
    const uint32_t num_repeats = 1u;
    std::string log_filename;
    std::string policy_filename;
    nhp.param(std::string("log_filename"), log_filename, std::string("/tmp/nomdp_policy_log.txt"));
    nhp.param(std::string("policy_filename"), policy_filename, std::string("/tmp/nomdp_planner_policy.policy"));
#else
    const uint32_t num_particles = 50u;
    const uint32_t num_repeats = (argc > 1) ? (uint32_t)atoi(argv[1]) : 1u;
    const std::string log_filename = (argc > 2) ? std::string(argv[2]) : std::string("/tmp/nomdp_planner_log.txt");
    const std::string policy_filename = (argc > 3) ? std::string(argv[3]) : std::string("/tmp/nomdp_planner_policy.policy");
    const uint32_t num_executions = (argc > 4) ? (uint32_t)atoi(argv[4]) : 50u;
    const double goal_probability_threshold = (argc > 5) ? atof(argv[5]) : 0.51;
    const double signature_matching_threshold = (argc > 6) ? atof(argv[6]) : 0.125;
    const double feasibility_alpha = (argc > 7) ? atof(argv[7]) : 0.75;
    const double variance_alpha = (argc > 8) ? atof(argv[8]) : 0.75;
#endif
    const double env_resolution = 0.125;
    // Planner params
    PLANNER_OPTIONS planner_options;
    planner_options.exec_step_limit = 100u;
    planner_options.step_size = 10.0 * env_resolution;
    planner_options.goal_probability_threshold = goal_probability_threshold;
    planner_options.goal_distance_threshold = 1.0 * env_resolution;
    planner_options.signature_matching_threshold = signature_matching_threshold;
    planner_options.feasibility_alpha = feasibility_alpha;
    planner_options.variance_alpha = variance_alpha;
    planner_options.num_particles = num_particles;
    planner_options.num_policy_simulations = num_executions;
    planner_options.num_policy_executions = num_executions;
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
    std::vector<std::map<std::string, double>> policy_performance(num_repeats);
    for (size_t idx = 0; idx < policy_performance.size(); idx++)
    {
    #ifdef USE_ROS
        policy_performance[idx] = peg_in_hole_env_6dof(planner_options, robot_config, policy_filename, display_debug_publisher, robot_control_service);
    #else
        policy_performance[idx] = peg_in_hole_env_6dof(planner_options, robot_config, policy_filename);
    #endif
    }
    // Print out the results & save them to the log file
    const std::string log_results = PrettyPrint::PrettyPrint(policy_performance, false, "\n");
    std::cout << "Policy results for " << num_particles << " particles:\nAll: " << log_results << std::endl;
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
