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
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/Image.h>

//#define USE_6DOF
//#define USE_R3
//#define USE_SE2

#ifndef USE_6DOF
    #ifndef USE_R3
        #ifndef USE_SE2
            #error ROBOT TYPE MUST BE SELECTED VIA DEFINE
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
    else
    {
        return std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>{nomdp_planning_tools::OBSTACLE_CONFIG(1u, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.125, 2.0, 0.125), 0x55, 0x6b, 0x2f, 0xff)};
    }
}

EigenHelpers::VectorVector3d make_robot(const int32_t robot_code)
{
    if (robot_code == 0)
    {
        return EigenHelpers::VectorVector3d{Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Vector3d(0.375, 0.0, 0.0), Eigen::Vector3d(0.25, 0.0, 0.0), Eigen::Vector3d(0.125, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(-0.125, 0.0, 0.0), Eigen::Vector3d(-0.25, 0.0, 0.0), Eigen::Vector3d(-0.375, 0.0, 0.0), Eigen::Vector3d(-0.5, 0.0, 0.0)};
    }
    else if (robot_code == 1)
    {
        return EigenHelpers::VectorVector3d{Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.875, 0.0, 0.0), Eigen::Vector3d(0.75, 0.0, 0.0), Eigen::Vector3d(0.625, 0.0, 0.0), Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Vector3d(0.375, 0.0, 0.0), Eigen::Vector3d(0.25, 0.0, 0.0), Eigen::Vector3d(0.125, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0)};
    }
    else if (robot_code == 2)
    {
        return EigenHelpers::VectorVector3d{Eigen::Vector3d(-1.0, 0.0, 0.0), Eigen::Vector3d(-0.875, 0.0, 0.0), Eigen::Vector3d(-0.75, 0.0, 0.0), Eigen::Vector3d(-0.625, 0.0, 0.0), Eigen::Vector3d(-0.5, 0.0, 0.0), Eigen::Vector3d(-0.375, 0.0, 0.0), Eigen::Vector3d(-0.25, 0.0, 0.0), Eigen::Vector3d(-0.125, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0)};
    }
    else if (robot_code == 3)
    {
        return EigenHelpers::VectorVector3d{Eigen::Vector3d(0.0, -0.125, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.125, 0.0, 0.0), Eigen::Vector3d(-0.125, 0.0, 0.0), Eigen::Vector3d(0.0, 0.125, 0.0), Eigen::Vector3d(0.125, 0.125, 0.0), Eigen::Vector3d(-0.125, 0.125, 0.0), Eigen::Vector3d(0.25, 0.125, 0.0), Eigen::Vector3d(-0.25, 0.125, 0.0)};
    }
    else
    {
        return EigenHelpers::VectorVector3d{Eigen::Vector3d(0.0, 0.0, 0.0)};
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
    int32_t env_code = 5; //0;
    int32_t robot_code = 4; //0;
    if (argc >= 2)
    {
        env_code = (int32_t)atoi(argv[1]);
    }
    if (argc >= 3)
    {
        robot_code = (int32_t)atoi(argv[2]);
    }
    // Make the environment
    const std::vector<nomdp_planning_tools::OBSTACLE_CONFIG> env_objects = make_environment(env_code);
    // Initialize the planning space
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
#endif
#ifdef USE_R3
    eigenvector3d_robot_helpers::EigenVector3dBaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
#endif
#ifdef USE_SE2
    UNUSED(env_min_z);
    UNUSED(env_max_z);
    simplese2_robot_helpers::SimpleSE2BaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y));
#endif
    double kp = 0.1;
    double ki = 0.0;
    double kd = 0.01;
    double i_clamp = 0.0;
    double velocity_limit = env_resolution * 2.0;
    double max_sensor_noise = 0.0; //env_resolution * 0.01;
    double max_actuator_noise = 0.0; //env_resolution * 1.0; //1.0;
    double feasibility_alpha = 0.75;
    double variance_alpha = 0.75;
    // Make the robot geometry
    const EigenHelpers::VectorVector3d robot_points = make_robot(robot_code);
    // Make the actual robot
#ifdef USE_6DOF
    simple6dof_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, (velocity_limit * 0.125), (max_sensor_noise * 0.125), (max_actuator_noise * 0.125));
    Eigen::Matrix<double, 6, 1> initial_config = Eigen::Matrix<double, 6, 1>::Zero();
    simple6dof_robot_helpers::Simple6DOFRobot robot(robot_points, initial_config, robot_config);
#endif
#ifdef USE_R3
    eigenvector3d_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
    eigenvector3d_robot_helpers::SimpleEigenVector3dRobot robot(robot_points, Eigen::Vector3d::Identity(), robot_config);
#endif
#ifdef USE_SE2
    simplese2_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, (velocity_limit * 0.125), (max_sensor_noise * 0.125), (max_actuator_noise * 0.125));
    Eigen::Matrix<double, 3, 1> initial_config = Eigen::Matrix<double, 3, 1>::Zero();
    simplese2_robot_helpers::SimpleSE2Robot robot(robot_points, initial_config, robot_config);
#endif
    bool use_contact = true;
#ifdef USE_6DOF
    NomdpPlanningSpace<simple6dof_robot_helpers::Simple6DOFRobot, simple6dof_robot_helpers::Simple6DOFBaseSampler, Eigen::Matrix<double, 6, 1>, simple6dof_robot_helpers::Simple6DOFAverager, simple6dof_robot_helpers::Simple6DOFDistancer, simple6dof_robot_helpers::Simple6DOFDimDistancer, simple6dof_robot_helpers::Simple6DOFInterpolator, std::allocator<Eigen::Matrix<double, 6, 1>>, std::mt19937_64> planning_space(use_contact, 1u, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, env_objects, env_resolution, 1u);
#endif
#ifdef USE_R3
    NomdpPlanningSpace<eigenvector3d_robot_helpers::SimpleEigenVector3dRobot, eigenvector3d_robot_helpers::EigenVector3dBaseSampler, Eigen::Vector3d, eigenvector3d_robot_helpers::EigenVector3dAverager, eigenvector3d_robot_helpers::EigenVector3dDistancer, eigenvector3d_robot_helpers::EigenVector3dDimDistancer, eigenvector3d_robot_helpers::EigenVector3dInterpolator, Eigen::aligned_allocator<Eigen::Vector3d>, std::mt19937_64> planning_space(use_contact, 1u, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, env_objects, env_resolution, 1u);
#endif
#ifdef USE_SE2
    NomdpPlanningSpace<simplese2_robot_helpers::SimpleSE2Robot, simplese2_robot_helpers::SimpleSE2BaseSampler, Eigen::Matrix<double, 3, 1>, simplese2_robot_helpers::SimpleSE2Averager, simplese2_robot_helpers::SimpleSE2Distancer, simplese2_robot_helpers::SimpleSE2DimDistancer, simplese2_robot_helpers::SimpleSE2Interpolator, std::allocator<Eigen::Matrix<double, 3, 1>>, std::mt19937_64> planning_space(use_contact, 1u, step_size, goal_distance_threshold, goal_probability_threshold, signature_matching_threshold, feasibility_alpha, variance_alpha, robot, sampler, env_objects, env_resolution, 1u);
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
    planning_space.DemonstrateSimulator(start, goal, display_debug_publisher);
    return 0;
}
