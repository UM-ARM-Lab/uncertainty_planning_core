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
#include "common_config.hpp"

namespace linked_common_config
{
    typedef simplelinked_robot_helpers::SimpleJointModel SJM;
    typedef simplelinked_robot_helpers::SimpleLinkedConfiguration SLC;

    inline common_config::OPTIONS GetDefaultOptions()
    {
        common_config::OPTIONS options;
        options.environment_resolution = 0.125;
        options.planner_time_limit = 1200.0;
        options.goal_bias = 0.1;
        options.step_size = 15.0 * options.environment_resolution;
        options.goal_probability_threshold = 0.51;
        options.goal_distance_threshold = 2.0 * options.environment_resolution;
        options.signature_matching_threshold = 0.5;
        options.distance_clustering_threshold = 15.0 * options.environment_resolution;
        options.feasibility_alpha = 0.75;
        options.variance_alpha = 0.75;
        options.actuator_error = M_PI * 0.02;
        options.sensor_error = 0.0;
        options.action_attempt_count = 50u;
        options.num_particles = 24u;
        options.use_contact = true;
        options.use_reverse = true;
        options.use_spur_actions = true;
        options.exec_step_limit = 1000u;
        options.num_policy_simulations = 1000u;
        options.num_policy_executions = 100u;
        options.enable_contact_manifold_target_adjustment = false;
        options.planner_log_file = "/tmp/linked_planner_log.txt";
        options.policy_log_file = "/tmp/linked_policy_log.txt";
        options.planned_policy_file = "/tmp/linked_planned_policy.policy";
        options.executed_policy_file = "/tmp/linked_executed_policy.policy";
        return options;
    }

#ifdef USE_ROS
    inline common_config::OPTIONS GetOptions(const common_config::OPTIONS::TYPE& type)
    {
        common_config::OPTIONS options = GetDefaultOptions();
        // Get options via ROS params
        ros::NodeHandle nhp("~");
        if (type == common_config::OPTIONS::PLANNING)
        {
            options.planner_time_limit = nhp.param(std::string("planning_time"), options.planner_time_limit);
            options.num_particles = (uint32_t)nhp.param(std::string("num_particles"), (int)options.num_particles);
            options.actuator_error = nhp.param(std::string("actuator_error"), options.actuator_error);
            options.sensor_error = nhp.param(std::string("sensor_error"), options.sensor_error);
            options.planner_log_file = nhp.param(std::string("planner_log_file"), options.planner_log_file);
            options.planned_policy_file = nhp.param(std::string("planned_policy_file"), options.planned_policy_file);
        }
        else if (type == common_config::OPTIONS::EXECUTION)
        {
            options.num_policy_simulations = (uint32_t)nhp.param(std::string("num_policy_simulations"), (int)options.num_policy_simulations);
            options.num_policy_executions = (uint32_t)nhp.param(std::string("num_policy_executions"), (int)options.num_policy_executions);
            options.actuator_error = nhp.param(std::string("actuator_error"), options.actuator_error);
            options.sensor_error = nhp.param(std::string("sensor_error"), options.sensor_error);
            options.policy_log_file = nhp.param(std::string("policy_log_file"), options.policy_log_file);
            options.planned_policy_file = nhp.param(std::string("planned_policy_file"), options.planned_policy_file);
            options.executed_policy_file = nhp.param(std::string("executed_policy_file"), options.executed_policy_file);
        }
        else
        {
            throw std::invalid_argument("Unsupported options type");
        }
        return options;
    }
#else
    inline common_config::OPTIONS GetOptions(int argc, char** argv, const common_config::OPTIONS::TYPE& type)
    {
        common_config::OPTIONS options = GetDefaultOptions();
        // Get options via arguments
        if (type == common_config::OPTIONS::PLANNING)
        {
            if (argc >= 2)
            {
                options.planner_time_limit = atof(argv[1]);
            }
            if (argc >= 3)
            {
                options.num_particles = (uint32_t)atoi(argv[2]);
            }
            if (argc >= 4)
            {
                options.actuator_error = atof(argv[3]);
            }
            if (argc >= 5)
            {
                options.sensor_error = atof(argv[4]);
            }
            if (argc >= 6)
            {
                options.planner_log_file = std::string(argv[5]);
            }
            if (argc >= 7)
            {
                options.planned_policy_file = std::string(argv[6]);
            }
        }
        else if (type == common_config::OPTIONS::EXECUTION)
        {
            if (argc >= 2)
            {
                options.num_policy_simulations = (uint32_t)atoi(argv[1]);
            }
            if (argc >= 3)
            {
                options.num_policy_executions = (uint32_t)atoi(argv[2]);
            }
            if (argc >= 4)
            {
                options.actuator_error = atof(argv[3]);
            }
            if (argc >= 5)
            {
                options.sensor_error = atof(argv[4]);
            }
            if (argc >= 6)
            {
                options.policy_log_file = std::string(argv[5]);
            }
            if (argc >= 7)
            {
                options.planned_policy_file = std::string(argv[6]);
            }
            if (argc >= 7)
            {
                options.executed_policy_file = std::string(argv[6]);
            }
        }
        else
        {
            throw std::invalid_argument("Unsupported options type");
        }
        return options;
    }
#endif

    inline simplelinked_robot_helpers::ROBOT_CONFIG GetDefaultRobotConfig(const common_config::OPTIONS& options)
    {
        const double env_resolution = options.environment_resolution;
        const double kp = 0.1;
        const double ki = 0.0;
        const double kd = 0.01;
        const double i_clamp = 0.0;
        const double velocity_limit = env_resolution * 2.0;
        const double max_sensor_noise = options.sensor_error;
        const double max_actuator_noise = options.actuator_error;
        const simplelinked_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
        return robot_config;
    }

    inline Eigen::Affine3d GetBaseTransform()
    {
        const Eigen::Affine3d base_transform = Eigen::Translation3d(5.0, 5.0, 3.0) * Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
        return base_transform;
    }

    inline std::pair<SLC, SLC> GetStartAndGoal()
    {
        // Define the goals of the plan
        const SLC start = {SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), -M_PI_2, SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), M_PI_2, SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), M_PI_2, SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE)};
        const SLC goal = {SJM(std::pair<double, double>(-M_PI, M_PI), M_PI_4, SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), -(M_PI_4 * 1.2), SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), (M_PI_2 * 1.2), SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), (M_PI_4 * 0.8), SJM::REVOLUTE),
                            SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE)};
        return std::make_pair(start, goal);
    }

    inline SLC GetReferenceConfiguration()
    {
        const SLC reference_configuration = {SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE),
                                            SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE),
                                            SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE),
                                            SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE),
                                            SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE),
                                            SJM(std::pair<double, double>(-M_PI, M_PI), 0.0, SJM::REVOLUTE)};
        return reference_configuration;
    }

    inline simplelinked_robot_helpers::SimpleLinkedRobot GetRobot(const Eigen::Affine3d& base_transform, const simplelinked_robot_helpers::ROBOT_CONFIG& joint_config)
    {
        // Make the reference configuration
        const SLC reference_configuration = GetReferenceConfiguration();
        // Make the robot model
        simplelinked_robot_helpers::RobotLink link1;
        link1.link_name = "link1";
        link1.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link2;
        link2.link_name = "link2";
        link2.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link2.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.125));
        simplelinked_robot_helpers::RobotLink link3;
        link3.link_name = "link3";
        link3.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(1.125, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(1.25, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(1.375, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(1.5, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(1.625, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(1.75, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(1.875, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(2.0, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(2.125, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(2.25, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(2.375, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(2.5, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(2.625, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(2.75, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(2.875, 0.0, 0.0));
        link3.link_points.push_back(Eigen::Vector3d(3.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link4;
        link4.link_name = "link4";
        link4.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.125, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.25, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.375, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.5, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.625, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.75, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(1.875, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(2.0, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(2.125, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(2.25, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(2.375, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(2.5, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(2.625, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(2.75, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(2.875, 0.0, 0.0));
        link4.link_points.push_back(Eigen::Vector3d(3.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link5;
        link5.link_name = "link5";
        link5.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link5.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link6;
        link6.link_name = "link6";
        link6.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink link7;
        link7.link_name = "link7";
        link7.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.25, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.375, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.625, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.75, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(0.875, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(1.0, 0.125, 0.0));
        link7.link_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.125));
        std::vector<simplelinked_robot_helpers::RobotLink> links = {link1, link2, link3, link4, link5, link6, link7};
        std::vector<std::pair<size_t, size_t>> allowed_self_collisions = {std::pair<size_t, size_t>(0, 1), std::pair<size_t, size_t>(1, 2), std::pair<size_t, size_t>(2, 3), std::pair<size_t, size_t>(3, 4), std::pair<size_t, size_t>(4, 5), std::pair<size_t, size_t>(5, 6), std::pair<size_t, size_t>(3, 5), std::pair<size_t, size_t>(3, 6), std::pair<size_t, size_t>(4, 6)};
        // Base yaw
        simplelinked_robot_helpers::RobotJoint joint1;
        joint1.parent_link_index = 0;
        joint1.child_link_index = 1;
        joint1.joint_axis = Eigen::Vector3d::UnitZ();
        joint1.joint_transform = Eigen::Translation3d(0.0, 0.0, 0.125) * Eigen::Quaterniond::Identity();
        joint1.joint_model = reference_configuration[0];
        joint1.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        // Base pitch
        simplelinked_robot_helpers::RobotJoint joint2;
        joint2.parent_link_index = 1;
        joint2.child_link_index = 2;
        joint2.joint_axis = Eigen::Vector3d::UnitY();
        joint2.joint_transform = Eigen::Translation3d(0.0, 0.125, 0.125) * Eigen::Quaterniond::Identity();
        joint2.joint_model = reference_configuration[1];
        joint2.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        // Elbow pitch
        simplelinked_robot_helpers::RobotJoint joint3;
        joint3.parent_link_index = 2;
        joint3.child_link_index = 3;
        joint3.joint_axis = Eigen::Vector3d::UnitY();
        joint3.joint_transform = Eigen::Translation3d(3.0, -0.125, 0.0) * Eigen::Quaterniond::Identity();
        joint3.joint_model = reference_configuration[2];
        joint3.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        // Elbow roll
        simplelinked_robot_helpers::RobotJoint joint4;
        joint4.parent_link_index = 3;
        joint4.child_link_index = 4;
        joint4.joint_axis = Eigen::Vector3d::UnitX();
        joint4.joint_transform = Eigen::Translation3d(3.0, 0.0, 0.0) * Eigen::Quaterniond::Identity();
        joint4.joint_model = reference_configuration[3];
        joint4.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        // Wrist pitch
        simplelinked_robot_helpers::RobotJoint joint5;
        joint5.parent_link_index = 4;
        joint5.child_link_index = 5;
        joint5.joint_axis = Eigen::Vector3d::UnitY();
        joint5.joint_transform = Eigen::Translation3d(0.125, 0.0, 0.0) * Eigen::Quaterniond::Identity();
        joint5.joint_model = reference_configuration[4];
        joint5.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        // Wrist roll
        simplelinked_robot_helpers::RobotJoint joint6;
        joint6.parent_link_index = 5;
        joint6.child_link_index = 6;
        joint6.joint_axis = Eigen::Vector3d::UnitX();
        joint6.joint_transform = Eigen::Translation3d(0.0, 0.0, 0.0) * Eigen::Quaterniond::Identity();
        joint6.joint_model = reference_configuration[5];
        joint6.joint_controller = simplelinked_robot_helpers::JointControllerGroup(joint_config);
        std::vector<simplelinked_robot_helpers::RobotJoint>joints = {joint1, joint2, joint3, joint4, joint5, joint6};
        const simplelinked_robot_helpers::SimpleLinkedRobot robot(base_transform, links, joints, allowed_self_collisions, reference_configuration);
        return robot;
    }

    inline simplelinked_robot_helpers::SimpleLinkedBaseSampler GetSampler()
    {
        // Make the sampler
        const SLC reference_configuration = GetReferenceConfiguration();
        const simplelinked_robot_helpers::SimpleLinkedBaseSampler sampler(reference_configuration);
        return sampler;
    }
}
