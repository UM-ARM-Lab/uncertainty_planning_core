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

#ifdef USE_ROS
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#endif

#ifdef ENABLE_PARALLEL
#include <omp.h>
#endif

using namespace nomdp_contact_planning;

std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> contact_test_env(int argc, char** argv, const size_t num_particles, const u_int32_t num_threads)
{
#ifdef USE_ROS
    ros::init(argc, argv, "nomdp_contact_planning_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Nomdp Contact Planning Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
#else
    UNUSED(argc);
    UNUSED(argv);
#endif
    // Make the environment
    std::vector<OBSTACLE_CONFIG> env_objects;
    OBSTACLE_CONFIG object1(Eigen::Vector3d(0.0, 0.0, -4.75), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 5.0, 0.25));
    env_objects.push_back(object1);
    OBSTACLE_CONFIG object2(Eigen::Vector3d(4.75, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.25, 5.0, 5.0));
    env_objects.push_back(object2);
    OBSTACLE_CONFIG object3(Eigen::Vector3d(0.0, 4.75, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 0.25, 5.0));
    env_objects.push_back(object3);
    OBSTACLE_CONFIG object4(Eigen::Vector3d(1.25, 1.25, -4.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(3.0, 3.0, 0.25));
    env_objects.push_back(object4);
    OBSTACLE_CONFIG object5(Eigen::Vector3d(4.0, 1.25, -1.25), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.25, 3.0, 3.0));
    env_objects.push_back(object5);
    OBSTACLE_CONFIG object6(Eigen::Vector3d(1.25, 4.0, -1.25), Eigen::Quaterniond::Identity(), Eigen::Vector3d(3.0, 0.25, 3.0));
    env_objects.push_back(object6);
    // Initialize the planning space
    double planner_time_limit = 300.0;
    double env_resolution = 0.125;
    double step_size = 10 * env_resolution;
    double goal_distance_threshold = 1.0 * env_resolution;
    double goal_probability_threshold = 0.25;
    double env_min_x = -5.0;
    double env_max_x = 6.0;
    double env_min_y = -5.0;
    double env_max_y = 6.0;
    double env_min_z = -6.0;
    double env_max_z = 5.0;
    double kp = 0.1;
    double ki = 0.0;
    double kd = 0.05;
    double i_clamp = 0.0;
    double velocity_limit = env_resolution * 2.0;
    double max_sensor_noise = env_resolution * 0.1;
    double max_actuator_noise = env_resolution * 1.0;
    double max_robot_trajectory_curvature = 4.0;
    double feasibility_alpha = 0.75;
    double variance_alpha = 0.75;
    POINT_ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    bool use_contact = true;
    NomdpPlanningSpace planning_space(use_contact, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, max_robot_trajectory_curvature, feasibility_alpha, variance_alpha, robot_config, env_min_x, env_min_y, env_min_z, env_max_x, env_max_y, env_max_z, env_objects, env_resolution, num_threads);
    // Define the goals of the plan
    Eigen::Vector3d start(-4.5, 0.0, -1.5);
    Eigen::Vector3d goal(4.5, 4.5, -4.5);
    double goal_bias = 0.1;
    std::chrono::duration<double> time_limit(planner_time_limit);
    // Plan & execute
    std::chrono::duration<double> exec_time_limit(5.0);
#ifdef USE_ROS
    std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, display_debug_publisher);
    double policy_success = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit, display_debug_publisher);
    std::cout << "Policy execution success: " << policy_success << std::endl;
#else
    std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = planning_space.Plan(start, goal, goal_bias, time_limit);
    double policy_success = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit);
    std::cout << "Policy execution success: " << policy_success << std::endl;
#endif
    return planner_result;
}

std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> three_block_env(int argc, char** argv, const size_t num_particles, const u_int32_t num_threads)
{
#ifdef USE_ROS
    ros::init(argc, argv, "nomdp_contact_planning_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Nomdp Contact Planning Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
#else
    UNUSED(argc);
    UNUSED(argv);
#endif
    // Make the environment
    std::vector<OBSTACLE_CONFIG> env_objects;
    OBSTACLE_CONFIG object1(Eigen::Vector3d(0.0, 0.0, 0.5), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 1.0, 2.0));
    env_objects.push_back(object1);
    OBSTACLE_CONFIG object2(Eigen::Vector3d(0.0, 0.0, -4.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 1.0, 1.0));
    env_objects.push_back(object2);
    OBSTACLE_CONFIG object3(Eigen::Vector3d(0.0, 0.0, 4.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 1.0, 1.0));
    env_objects.push_back(object3);
    // Add sharp edges to make the task harder
    OBSTACLE_CONFIG object4(Eigen::Vector3d(0.0, -0.5, 3.125), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 1.0, 0.125));
    env_objects.push_back(object4);
    // Initialize the planning space
    double planner_time_limit = 30.0;
    double env_resolution = 0.125;
    double step_size = 10 * env_resolution;
    double goal_distance_threshold = 10.0 * env_resolution;
    double goal_probability_threshold = 0.25;
    double env_min_x = -5.0;
    double env_max_x = 5.0;
    double env_min_y = -5.0;
    double env_max_y = 5.0;
    double env_min_z = -5.0;
    double env_max_z = 5.0;
    double kp = 0.1;
    double ki = 0.0;
    double kd = 0.05;
    double i_clamp = 0.0;
    double velocity_limit = env_resolution * 2.0;
    double max_sensor_noise = env_resolution * 10.0;
    double max_actuator_noise = env_resolution * 2.0;
    double max_robot_trajectory_curvature = 4.0;
    double feasibility_alpha = 0.75;
    double variance_alpha = 0.5;
    POINT_ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    bool use_contact = true;
    NomdpPlanningSpace planning_space(use_contact, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, max_robot_trajectory_curvature, feasibility_alpha, variance_alpha, robot_config, env_min_x, env_min_y, env_min_z, env_max_x, env_max_y, env_max_z, env_objects, env_resolution, num_threads);
    // Define the goals of the plan
    Eigen::Vector3d start(-4.5, -4.5, 4.5);
    Eigen::Vector3d goal(4.5, 4.5, 4.5);
    double goal_bias = 0.1;
    std::chrono::duration<double> time_limit(planner_time_limit);
    // Plan
#ifdef USE_ROS
    std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, display_debug_publisher);
#else
    std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = planning_space.Plan(start, goal, goal_bias, time_limit);
#endif
    return planner_result;
}

std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> dual_slit_env(int argc, char** argv, const size_t num_particles, const u_int32_t num_threads)
{
#ifdef USE_ROS
    ros::init(argc, argv, "nomdp_contact_planning_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Nomdp Contact Planning Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
#else
    UNUSED(argc);
    UNUSED(argv);
#endif
    // Make the environment
    std::vector<OBSTACLE_CONFIG> env_objects;
    OBSTACLE_CONFIG object1(Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 4.0, 0.125));
    env_objects.push_back(object1);
    OBSTACLE_CONFIG object2(Eigen::Vector3d(0.0, 0.0, -4.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 4.0, 3.625));
    env_objects.push_back(object2);
    OBSTACLE_CONFIG object3(Eigen::Vector3d(0.0, 0.0, 4.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 4.0, 3.625));
    env_objects.push_back(object3);
    // Initialize the planning space
    double planner_time_limit = 30.0;
    double env_resolution = 0.125;
    double step_size = 10 * env_resolution;
    double goal_distance_threshold = 10.0 * env_resolution;
    double goal_probability_threshold = 0.25;
    double env_min_x = -5.0;
    double env_max_x = 5.0;
    double env_min_y = -5.0;
    double env_max_y = 5.0;
    double env_min_z = -5.0;
    double env_max_z = 5.0;
    double kp = 0.1;
    double ki = 0.0;
    double kd = 0.05;
    double i_clamp = 0.0;
    double velocity_limit = env_resolution * 2.0;
    double max_sensor_noise = env_resolution * 10.0;
    double max_actuator_noise = env_resolution * 2.0;
    double max_robot_trajectory_curvature = 4.0;
    double feasibility_alpha = 0.75;
    double variance_alpha = 0.5;
    POINT_ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    bool use_contact = true;
    NomdpPlanningSpace planning_space(use_contact, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, max_robot_trajectory_curvature, feasibility_alpha, variance_alpha, robot_config, env_min_x, env_min_y, env_min_z, env_max_x, env_max_y, env_max_z, env_objects, env_resolution, num_threads);
    // Define the goals of the plan
    Eigen::Vector3d start(-4.5, -4.5, 4.5);
    Eigen::Vector3d goal(4.5, 4.5, 4.5);
    double goal_bias = 0.1;
    std::chrono::duration<double> time_limit(planner_time_limit);
    // Plan
#ifdef USE_ROS
    std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, display_debug_publisher);
#else
    std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = planning_space.Plan(start, goal, goal_bias, time_limit);
#endif
    return planner_result;
}

std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> pan_env(int argc, char** argv, const size_t num_particles, const u_int32_t num_threads)
{
#ifdef USE_ROS
    ros::init(argc, argv, "nomdp_contact_planning_node");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");
    ROS_INFO("Starting Nomdp Contact Planning Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
#else
    UNUSED(argc);
    UNUSED(argv);
#endif
    // Make the environment
    std::vector<OBSTACLE_CONFIG> env_objects;
    // Bottom and sides of a "pan-shaped" environment
    OBSTACLE_CONFIG base_object(Eigen::Vector3d(5.0, 5.0, -0.125), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 5.0, 0.125));
    env_objects.push_back(base_object);
//    OBSTACLE_CONFIG top_object(Eigen::Vector3d(5.0, 5.0, 1.125), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 5.0, 0.125));
//    env_objects.push_back(top_object);
    OBSTACLE_CONFIG left_wall_object(Eigen::Vector3d(5.0, 10.125, 0.5), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 0.125, 0.5));
    env_objects.push_back(left_wall_object);
    OBSTACLE_CONFIG right_wall_object(Eigen::Vector3d(5.0, -0.125, 0.5), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 0.125, 0.5));
    env_objects.push_back(right_wall_object);
    OBSTACLE_CONFIG front_wall_object(Eigen::Vector3d(10.125, 5.0, 0.5), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.125, 5.0, 0.5));
    env_objects.push_back(front_wall_object);
    OBSTACLE_CONFIG back_wall_object(Eigen::Vector3d(-0.125, 5.0, 0.5), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.125, 5.0, 0.5));
    env_objects.push_back(back_wall_object);
    // Obstacles
    OBSTACLE_CONFIG obstacle1(Eigen::Vector3d(5.0, 5.0, 0.5), Eigen::Quaterniond::Identity(), Eigen::Vector3d(2.0, 2.0, 0.5));
    env_objects.push_back(obstacle1);
    OBSTACLE_CONFIG obstacle2(Eigen::Vector3d(5.0, 2.5, 0.5), Eigen::Quaterniond::Identity(), Eigen::Vector3d(2.0, 2.0, 0.5));
    //env_objects.push_back(obstacle2);
    // Initialize the planning space
    double planner_time_limit = 30.0;
    double env_resolution = 0.125;
    double step_size = 10 * env_resolution;
    double goal_distance_threshold = 10.0 * env_resolution;
    double goal_probability_threshold = 0.25;
    double env_min_x = 0.05;
    double env_max_x = 9.95;
    double env_min_y = 0.05;
    double env_max_y = 9.95;
    double env_min_z = 0.05;
    double env_max_z = 0.95;
    double kp = 0.1;
    double ki = 0.0;
    double kd = 0.05;
    double i_clamp = 0.0;
    double velocity_limit = env_resolution * 2.0;
    double max_sensor_noise = env_resolution * 4.0;
    double max_actuator_noise = env_resolution * 0.5;
    double max_robot_trajectory_curvature = 4.0;
    double feasibility_alpha = 0.75;
    double variance_alpha = 0.5;
    POINT_ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    bool use_contact = true;
    NomdpPlanningSpace planning_space(use_contact, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, max_robot_trajectory_curvature, feasibility_alpha, variance_alpha, robot_config, env_min_x, env_min_y, env_min_z, env_max_x, env_max_y, env_max_z, env_objects, env_resolution, num_threads);
    // Define the goals of the plan
    Eigen::Vector3d start(0.1, 0.1, 0.5);
    Eigen::Vector3d goal(9.9, 9.9, 0.5);
    double goal_bias = 0.1;
    std::chrono::duration<double> time_limit(planner_time_limit);
    // Plan
#ifdef USE_ROS
    std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, display_debug_publisher);
#else
    std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = planning_space.Plan(start, goal, goal_bias, time_limit);
#endif
    return planner_result;
}

#ifdef ENABLE_PARALLEL
int omp_test(void)
{
    int th_id, nthreads;
    #pragma omp parallel private(th_id)
    {
        th_id = omp_get_thread_num();
        #pragma omp barrier
        if (th_id == 0)
        {
            nthreads = omp_get_num_threads();
        }
    }
    return nthreads;
}
#endif

void clustering_test()
{
    std::vector<double> test_data = {1.0, 11.2, 13.2, 15.4, 3.0, 4.0, 2.5};
    SimpleHierarchicalClustering<double> clustering;
    std::function<double(const double&, const double&)> distance_fn = [](const double& x1, const double& x2) { return fabs(x1 - x2); };
    std::cout << "Clustered: " << PrettyPrint::PrettyPrint(clustering.Cluster(test_data, distance_fn, 5.0), true) << std::endl;
    exit(0);
}

int main(int argc, char** argv)
{
    //clustering_test();
    u_int32_t num_threads = 1u;
#ifdef ENABLE_PARALLEL
    num_threads = omp_test();
    std::cout << "OpenMP context has " << num_threads << " threads" << std::endl;
#endif
    size_t num_particles = 228;
    // Run different planning problems
    std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = contact_test_env(argc, argv, num_particles, num_threads);
    //std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = three_block_env(argc, argv, num_particles, num_threads);
    //std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = dual_slit_env(argc, argv, num_particles, num_threads);
    //std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> planner_result = pan_env(argc, argv, num_particles, num_threads);
    // Print out the results
    std::cout << "Planner results: " << PrettyPrint::PrettyPrint(planner_result.second) << std::endl;
    //std::cout << "Generated policy: " << PrettyPrint::PrettyPrint(planner_result.first) << std::endl;
    return 0;
}
