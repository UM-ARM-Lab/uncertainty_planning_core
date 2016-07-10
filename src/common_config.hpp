#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <nomdp_planning/nomdp_contact_planning.hpp>

#ifdef USE_ROS
#include <ros/ros.h>
#endif

#ifndef COMMON_CONFIG_HPP
#define COMMON_CONFIG_HPP

namespace common_config
{
    struct OPTIONS
    {
        enum TYPE {PLANNING, EXECUTION};

        nomdp_contact_planning::SPATIAL_FEATURE_CLUSTERING_TYPE clustering_type;
        double environment_resolution;
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
        // Uncertainty
        double actuator_error;
        double sensor_error;
        // Reverse/repeat params
        uint32_t edge_attempt_count;
        // Particle/execution limits
        uint32_t num_particles;
        // Execution limits
        uint32_t num_policy_simulations;
        uint32_t num_policy_executions;
        // How many attempts does a policy action count for?
        uint32_t policy_action_attempt_count;
        // Execution limits
        uint32_t max_exec_actions;
        // Control flags
        bool use_contact;
        bool use_reverse;
        bool use_spur_actions;
        bool enable_contact_manifold_target_adjustment;
        // Log & data files
        std::string planner_log_file;
        std::string policy_log_file;
        std::string planned_policy_file;
        std::string executed_policy_file;
    };

#ifdef USE_ROS
    inline OPTIONS GetOptions(const OPTIONS& initial_options, const common_config::OPTIONS::TYPE& type)
    {
        OPTIONS options = initial_options;
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
            options.clustering_type = nomdp_contact_planning::ParseSpatialFeatureClusteringType(nhp.param(std::string("clustering_type"), nomdp_contact_planning::PrintSpatialFeatureClusteringType(options.clustering_type)));
            options.signature_matching_threshold = nhp.param(std::string("signature_matching_threshold"), options.signature_matching_threshold);
            options.policy_action_attempt_count = (uint32_t)nhp.param(std::string("policy_action_attempt_count"), (int)options.policy_action_attempt_count);
            options.use_contact = nhp.param(std::string("use_contact"), options.use_contact);
            options.use_reverse = nhp.param(std::string("use_reverse"), options.use_reverse);
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
            options.clustering_type = nomdp_contact_planning::ParseSpatialFeatureClusteringType(nhp.param(std::string("clustering_type"), nomdp_contact_planning::PrintSpatialFeatureClusteringType(options.clustering_type)));
            options.signature_matching_threshold = nhp.param(std::string("signature_matching_threshold"), options.signature_matching_threshold);
            options.max_exec_actions = (uint32_t)nhp.param(std::string("max_exec_actions"), (int)options.max_exec_actions);
            options.policy_action_attempt_count = (uint32_t)nhp.param(std::string("policy_action_attempt_count"), (int)options.policy_action_attempt_count);
        }
        else
        {
            throw std::invalid_argument("Unsupported options type");
        }
        return options;
    }
#else
    inline OPTIONS GetOptions(const OPTIONS& initial_options, int argc, char** argv, const common_config::OPTIONS::TYPE& type)
    {
        OPTIONS options = initial_options;
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
            if (argc >= 8)
            {
                options.clustering_type = nomdp_contact_planning::ParseSpatialFeatureClusteringType(std::string(argv[7]));;
            }
            if (argc >= 9)
            {
                options.signature_matching_threshold = atof(argv[8]);
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
            if (argc >= 8)
            {
                options.clustering_type = nomdp_contact_planning::ParseSpatialFeatureClusteringType(std::string(argv[7]));;
            }
            if (argc >= 9)
            {
                options.signature_matching_threshold = atof(argv[8]);
            }
            if (argc >= 10)
            {
                options.max_exec_actions = (uint32_t)atoi(argv[9]);
            }
            if (argc >= 11)
            {
                options.policy_action_attempt_count = (uint32_t)atoi(argv[10]);
            }
        }
        else
        {
            throw std::invalid_argument("Unsupported options type");
        }
        return options;
    }
#endif
}

std::ostream& operator<<(std::ostream& strm, const common_config::OPTIONS& options)
{
    strm << "OPTIONS:";
    strm << "\nclustering_type: " << nomdp_contact_planning::PrintSpatialFeatureClusteringType(options.clustering_type);
    strm << "\nenvironment_resolution: " << options.environment_resolution;
    strm << "\nplanner_time_limit: " << options.planner_time_limit;
    strm << "\ngoal_bias: " << options.goal_bias;
    strm << "\nstep_size: " << options.step_size;
    strm << "\ngoal_probability_threshold: " << options.goal_probability_threshold;
    strm << "\ngoal_distance_threshold: " << options.goal_distance_threshold;
    strm << "\nsignature_matching_threshold: " << options.signature_matching_threshold;
    strm << "\ndistance_clustering_threshold: " << options.distance_clustering_threshold;
    strm << "\nfeasibility_alpha: " << options.feasibility_alpha;
    strm << "\nvariance_alpha: " << options.variance_alpha;
    strm << "\nactuator_error: " << options.actuator_error;
    strm << "\nsensor_error: " << options.sensor_error;
    strm << "\nedge_attempt_count: " << options.edge_attempt_count;
    strm << "\npolicy_action_attempt_count: " << options.policy_action_attempt_count;
    strm << "\nnum_particles: " << options.num_particles;
    strm << "\nnum_policy_simulations: " << options.num_policy_simulations;
    strm << "\nnum_policy_executions: " << options.num_policy_executions;
    strm << "\nmax_exec_actions: " << options.max_exec_actions;
    strm << "\nuse_contact: " << options.use_contact;
    strm << "\nuse_reverse: " << options.use_reverse;
    strm << "\nuse_spur_actions: " << options.use_spur_actions;
    strm << "\nenable_contact_manifold_target_adjustment: " << options.enable_contact_manifold_target_adjustment;
    strm << "\nplanner_log_file: " << options.planner_log_file;
    strm << "\npolicy_log_file: " << options.policy_log_file;
    strm << "\nplanned_policy_file: " << options.planned_policy_file;
    strm << "\nexecuted_policy_file: " << options.executed_policy_file;
    return strm;
}

#endif // COMMON_CONFIG_HPP
