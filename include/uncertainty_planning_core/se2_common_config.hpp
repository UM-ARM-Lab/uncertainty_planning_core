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
#include "uncertainty_planning_core/simple_pid_controller.hpp"
#include "uncertainty_planning_core/simple_uncertainty_models.hpp"
#include "uncertainty_planning_core/uncertainty_contact_planning.hpp"
#include "uncertainty_planning_core/simplese2_robot_helpers.hpp"
#include "uncertainty_planning_core/uncertainty_planning_core.hpp"

#ifndef SE2_COMMON_CONFIG_HPP
#define SE2_COMMON_CONFIG_HPP

namespace se2_common_config
{
    inline uncertainty_planning_core::OPTIONS GetDefaultOptions()
    {
        uncertainty_planning_core::OPTIONS options;
        options.clustering_type = uncertainty_contact_planning::CONVEX_REGION_SIGNATURE;
        options.environment_name = "se2_maze";
        options.environment_resolution = 0.125;
        options.planner_time_limit = 120.0;
        options.goal_bias = 0.1;
        options.step_size = 15.0 * options.environment_resolution;
        options.step_duration = 10.0;
        options.goal_probability_threshold = 0.51;
        options.goal_distance_threshold = 2.0 * options.environment_resolution;
        options.connect_after_first_solution = 0.0;
        options.signature_matching_threshold = 0.75;
        options.distance_clustering_threshold = 15.0 * options.environment_resolution;
        options.feasibility_alpha = 0.75;
        options.variance_alpha = 0.75;
        options.actuator_error = options.environment_resolution * 1.0;
        options.sensor_error = 0.0;
        options.simulation_controller_frequency = 10.0;
        options.edge_attempt_count = 50u;
        options.num_particles = 24u;
        options.use_contact = true;
        options.use_reverse = true;
        options.use_spur_actions = true;
        options.max_exec_actions = 1000u;
        options.max_policy_exec_time = 300.0;
        options.num_policy_simulations = 10u;
        options.num_policy_executions = 0u;
        options.policy_action_attempt_count = 100u;
        options.debug_level = 0;
        options.planner_log_file = "/tmp/se2_planner_log.txt";
        options.policy_log_file = "/tmp/se2_policy_log.txt";
        options.planned_policy_file = "/tmp/se2_planned_policy.policy";
        options.executed_policy_file = "/dev/null";
        return options;
    }

    inline uncertainty_planning_core::OPTIONS GetOptions()
    {
        return uncertainty_planning_core::GetOptions(GetDefaultOptions());
    }

    inline simplese2_robot_helpers::ROBOT_CONFIG GetDefaultRobotConfig(const uncertainty_planning_core::OPTIONS& options)
    {
        const double kp = 0.1;
        const double ki = 0.0;
        const double kd = 0.01;
        const double i_clamp = 0.0;
        const double velocity_limit = 1.0;
        const double angular_velocity_limit = velocity_limit * 0.125;
        const double max_sensor_noise = options.sensor_error;
        const double max_angular_sensor_noise = max_sensor_noise * 0.125;
        const double max_actuator_noise = options.actuator_error;
        const double max_angular_actuator_noise = max_actuator_noise * 0.125;
        const simplese2_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise, kp, ki, kd, i_clamp, angular_velocity_limit, max_angular_sensor_noise, max_angular_actuator_noise);
        return robot_config;
    }

    inline std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1>> GetStartAndGoal()
    {
        // Define the goals of the plan
        Eigen::Matrix<double, 3, 1> start;
        start << 7.75, 7.75, 0.0;
        Eigen::Matrix<double, 3, 1> goal;
        goal << 0.75, 0.75, 0.0;
        return std::make_pair(start, goal);
    }

    inline std::shared_ptr<EigenHelpers::VectorVector3d> GetRobotPoints()
    {
        std::shared_ptr<EigenHelpers::VectorVector3d> robot_points(new EigenHelpers::VectorVector3d());
        const std::vector<double> x_pos = {-0.1875, -0.0625, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.0625, 1.1875, 1.3125, 1.4375};
        const std::vector<double> y_pos = {-0.1875, -0.0625, 0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.0625, 1.1875, 1.3125, 1.4375};
        const std::vector<double> z_pos = {-0.4375, -0.3125, -0.1875, -0.0625, 0.0625, 0.1875, 0.3125, 0.4375};
        for (size_t xpdx = 0; xpdx < x_pos.size(); xpdx++)
        {
            for (size_t ypdx = 0; ypdx < y_pos.size(); ypdx++)
            {
                if (xpdx <= 3 || ypdx <= 3)
                {
                    for (size_t zpdx = 0; zpdx < z_pos.size(); zpdx++)
                    {
                        robot_points->push_back(Eigen::Vector3d(x_pos[xpdx], y_pos[ypdx], z_pos[zpdx]));
                    }
                }
            }
        }
        return robot_points;
    }

    inline simplese2_robot_helpers::SimpleSE2Robot GetRobot(const simplese2_robot_helpers::ROBOT_CONFIG& robot_config)
    {
        // Make the actual robot
        const Eigen::Matrix<double, 3, 1> initial_config = Eigen::Matrix<double, 3, 1>::Zero();
        const simplese2_robot_helpers::SimpleSE2Robot robot(GetRobotPoints(), initial_config, robot_config);
        return robot;
    }

    inline simplese2_robot_helpers::SimpleSE2BaseSampler GetSampler()
    {
        const double env_resolution = 0.125;
        const double env_min_x = 0.0 + (env_resolution);
        const double env_max_x = 10.0 - (env_resolution);
        const double env_min_y = 0.0 + (env_resolution);
        const double env_max_y = 10.0 - (env_resolution);
        // Make the sampler
        const simplese2_robot_helpers::SimpleSE2BaseSampler sampler(std::pair<double, double>(env_min_x, env_max_x), std::pair<double, double>(env_min_y, env_max_y));
        return sampler;
    }
}

#endif // SE2_COMMON_CONFIG_HPP
