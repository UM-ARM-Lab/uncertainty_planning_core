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
#include "common_config.hpp"

#ifndef SE3_COMMON_CONFIG_HPP
#define SE3_COMMON_CONFIG_HPP

namespace se3_common_config
{
    inline common_config::OPTIONS GetDefaultOptions()
    {
        common_config::OPTIONS options;
        options.clustering_type = nomdp_contact_planning::CONVEX_REGION_SIGNATURE;
        options.environment_resolution = 0.125;
        options.planner_time_limit = 300.0;
        options.goal_bias = 0.1;
        options.step_size = 24.0 * options.environment_resolution;
        options.goal_probability_threshold = 0.51;
        options.goal_distance_threshold = 5.0 * options.environment_resolution;
        options.signature_matching_threshold = 0.75;
        options.distance_clustering_threshold = 15.0 * options.environment_resolution;
        options.feasibility_alpha = 0.75;
        options.variance_alpha = 0.75;
        options.actuator_error = options.environment_resolution * 1.0;
        options.sensor_error = 0.0;
        options.edge_attempt_count = 50u;
        options.num_particles = 24u;
        options.use_contact = true;
        options.use_reverse = true;
        options.use_spur_actions = true;
        options.max_exec_actions = 1000u;
        options.max_policy_exec_time = 300.0;
        options.num_policy_simulations = 1u;
        options.num_policy_executions = 1u;
        options.policy_action_attempt_count = 100u;
        options.enable_contact_manifold_target_adjustment = false;
        options.planner_log_file = "/tmp/se3_planner_log.txt";
        options.policy_log_file = "/tmp/se3_policy_log.txt";
        options.planned_policy_file = "/tmp/se3_planned_policy.policy";
        options.executed_policy_file = "/dev/null";
        return options;
    }

#ifdef USE_ROS
    inline common_config::OPTIONS GetOptions(const common_config::OPTIONS::TYPE& type)
    {
        return common_config::GetOptions(GetDefaultOptions(), type);
    }
#else
    inline common_config::OPTIONS GetOptions(int argc, char** argv, const common_config::OPTIONS::TYPE& type)
    {
        return common_config::GetOptions(GetDefaultOptions(), argc, argv, type);
    }
#endif

    inline simplese3_robot_helpers::ROBOT_CONFIG GetDefaultRobotConfig(const common_config::OPTIONS& options)
    {
        const double env_resolution = options.environment_resolution;
        const double kp = 0.1;
        const double ki = 0.0;
        const double kd = 0.01;
        const double i_clamp = 0.0;
        const double velocity_limit = env_resolution * 2.0;
        const double max_sensor_noise = options.sensor_error;
        const double max_actuator_noise = options.actuator_error;
        //const simplese3_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, (velocity_limit * 0.25), (max_sensor_noise * 0.25), (max_actuator_noise * 0.25));
        const simplese3_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, (velocity_limit * 0.25), (max_sensor_noise * 0.25), (max_actuator_noise * 0.25));
        return robot_config;
    }

    inline std::pair<Eigen::Affine3d, Eigen::Affine3d> GetStartAndGoal()
    {
        // Define the goals of the plan
        const Eigen::Affine3d start = Eigen::Translation3d(9.0, 9.0, 9.0) * Eigen::Quaterniond::Identity();
        const Eigen::Affine3d goal = Eigen::Translation3d(2.25, 2.25, 0.625) * Eigen::Quaterniond::Identity();
        return std::make_pair(start, goal);
    }

    inline EigenHelpers::VectorVector3d GetRobotPoints()
    {
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
        return robot_points;
    }

    inline simplese3_robot_helpers::SimpleSE3Robot GetRobot(const simplese3_robot_helpers::ROBOT_CONFIG& robot_config)
    {
        // Make the actual robot
        const EigenHelpers::VectorVector3d robot_points = GetRobotPoints();
        const Eigen::Affine3d initial_config = Eigen::Affine3d::Identity();
        simplese3_robot_helpers::SimpleSE3Robot robot(robot_points, initial_config, robot_config);
        return robot;
    }

    inline simplese3_robot_helpers::SimpleSE3BaseSampler GetSampler()
    {
        const double env_resolution = 0.125;
        const double env_min_x = 0.0 + (env_resolution);
        const double env_max_x = 10.0 - (env_resolution);
        const double env_min_y = 0.0 + (env_resolution);
        const double env_max_y = 10.0 - (env_resolution);
        const double env_min_z = 0.0 + (env_resolution);
        const double env_max_z = 10.0 - (env_resolution);
        // Make the sampler
        const simplese3_robot_helpers::SimpleSE3BaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y), std::pair<double, double>(env_min_z, env_max_z));
        return sampler;
    }
}

#endif // SE3_COMMON_CONFIG_HPP
