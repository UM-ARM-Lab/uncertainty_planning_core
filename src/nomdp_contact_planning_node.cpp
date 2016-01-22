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

#ifdef USE_ROS
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>
#endif

#ifdef ENABLE_PARALLEL
#include <omp.h>
#endif

#define USE_6DOF

using namespace nomdp_contact_planning;

double contact_test_env(int argc, char** argv, const u_int32_t num_particles, const u_int32_t num_threads)
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
    // Make the environment
    std::vector<nomdp_planning_tools::OBSTACLE_CONFIG> env_objects;
//    OBSTACLE_CONFIG object1(1u, Eigen::Vector3d(0.0, 0.0, -4.75), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 5.0, 0.25), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object1);
//    OBSTACLE_CONFIG object2(2u, Eigen::Vector3d(4.75, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.25, 5.0, 5.0), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object2);
//    OBSTACLE_CONFIG object3(3u, Eigen::Vector3d(0.0, 4.75, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 0.25, 5.0), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object3);
//    OBSTACLE_CONFIG object4(4u, Eigen::Vector3d(1.25, 1.25, -4.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(3.0, 3.0, 0.25), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object4);
//    OBSTACLE_CONFIG object5(5u, Eigen::Vector3d(4.0, 1.25, -1.25), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.25, 3.0, 3.0), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object5);
//    OBSTACLE_CONFIG object6(6u, Eigen::Vector3d(1.25, 4.0, -1.25), Eigen::Quaterniond::Identity(), Eigen::Vector3d(3.0, 0.25, 3.0), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object6);
//    OBSTACLE_CONFIG object2(2u, Eigen::Vector3d(4.75, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.25, 5.0, 5.0), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object2);
//    OBSTACLE_CONFIG object3(3u, Eigen::Vector3d(0.0, 4.75, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 0.25, 5.0), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object3);
//    OBSTACLE_CONFIG object5(5u, Eigen::Vector3d(2.25, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.25, 5.0, 5.0), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object5);
//    OBSTACLE_CONFIG object6(6u, Eigen::Vector3d(0.0, 2.25, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(5.0, 0.25, 5.0), 0x55, 0x6b, 0x2f, 0xff);
//    env_objects.push_back(object6);
    //OBSTACLE_CONFIG object7(7u, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Quaterniond(Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ())), Eigen::Vector3d(0.25, 0.25, 0.25), 0x55, 0x6b, 0x2f, 0xff);
    //env_objects.push_back(object7);
    // Fuck it, for now we're using the "default" environment that has been correctly populated (until convex segments works!)
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
    eigenvector3d_robot_helpers::SimpleEigenVector3dRobot robot(robot_points, Eigen::Vector3d::Identity(), robot_config);
#endif
    bool use_contact = true;
    bool use_reverse = false;
#ifdef USE_6DOF
    NomdpPlanningSpace<simple6dof_robot_helpers::Simple6DOFRobot, simple6dof_robot_helpers::Simple6DOFBaseSampler, Eigen::Matrix<double, 6, 1>, simple6dof_robot_helpers::Simple6DOFAverager, simple6dof_robot_helpers::Simple6DOFDistancer, simple6dof_robot_helpers::Simple6DOFDimDistancer, simple6dof_robot_helpers::Simple6DOFInterpolator, std::allocator<Eigen::Matrix<double, 6, 1>>, std::mt19937_64> planning_space(use_contact, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, env_objects, env_resolution, num_threads);
#else
    NomdpPlanningSpace<eigenvector3d_robot_helpers::SimpleEigenVector3dRobot, eigenvector3d_robot_helpers::EigenVector3dBaseSampler, Eigen::Vector3d, eigenvector3d_robot_helpers::EigenVector3dAverager, eigenvector3d_robot_helpers::EigenVector3dDistancer, eigenvector3d_robot_helpers::EigenVector3dDimDistancer, eigenvector3d_robot_helpers::EigenVector3dInterpolator, Eigen::aligned_allocator<Eigen::Vector3d>, std::mt19937_64> planning_space(use_contact, num_particles, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, env_objects, env_resolution, num_threads);
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
    auto planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, use_reverse, display_debug_publisher);
    std::cout << "Press ENTER to simulate policy..." << std::endl;
    std::cin.get();
    double policy_success = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit, display_debug_publisher, true);
#else
    auto planner_result = planning_space.Plan(start, goal, goal_bias, time_limit, use_reverse);
    double policy_success = planning_space.SimulateExectionPolicy(planner_result.first, start, goal, num_particles, exec_time_limit);
#endif
    //std::cout << "Policy execution success: " << policy_success << std::endl;
    return policy_success;
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
    std::function<double(const double&, const double&)> distance_fn = [](const double& x1, const double& x2) { return fabs(x1 - x2); };
    std::cout << "Clustered: " << PrettyPrint::PrettyPrint(simple_hierarchical_clustering::SimpleHierarchicalClustering::Cluster(test_data, distance_fn, 5.0), true) << std::endl;
    exit(0);
}

int main(int argc, char** argv)
{
    //clustering_test();
    u_int32_t num_threads = 1;
#ifdef ENABLE_PARALLEL
    num_threads = omp_test();
    std::cout << "OpenMP context has " << num_threads << " threads" << std::endl;
#endif
    const size_t num_executions = 1u;
    //size_t num_particles = 228u;
    //std::vector<u_int32_t> particle_counts = {40u, 80u, 120u, 160u, 200u, 240u, 280u, 320u, 360u, 400u, 440u, 480u, 520u, 560u};
    std::vector<u_int32_t> particle_counts = {100u}; //{200u};
    for (size_t pdx = 0; pdx < particle_counts.size(); pdx++)
    {
        const u_int32_t& num_particles = particle_counts[pdx];
        // Run different planning problems
        std::vector<double> policy_successes(num_executions, 0.0);
        for (size_t idx = 0; idx < policy_successes.size(); idx++)
        {
            policy_successes[idx] = contact_test_env(argc, argv, num_particles, num_threads);
        }
        // Print out the results
        const double max_success = *std::max_element(policy_successes.begin(), policy_successes.end());
        const double min_success = *std::min_element(policy_successes.begin(), policy_successes.end());
        std::cout << "Planner results for " << num_particles << " particles:\nSummary:\nMax success: " << max_success << " Min success: " << min_success << "\nAll: " << PrettyPrint::PrettyPrint(policy_successes) << std::endl;
    }
    return 0;
}
