#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#ifndef COMMON_CONFIG_HPP
#define COMMON_CONFIG_HPP

namespace common_config
{
    struct OPTIONS
    {
        enum TYPE {PLANNING, EXECUTION};

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
        uint32_t action_attempt_count;
        // Particle/execution limits
        uint32_t num_particles;
        // Execution limits
        uint32_t num_policy_simulations;
        uint32_t num_policy_executions;
        // Execution limits
        uint32_t exec_step_limit;
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
}

std::ostream& operator<<(std::ostream& strm, const common_config::OPTIONS& options)
{
    strm << "OPTIONS:";
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
    strm << "\naction_attempt_count: " << options.action_attempt_count;
    strm << "\nnum_particles: " << options.num_particles;
    strm << "\nnum_policy_simulations: " << options.num_policy_simulations;
    strm << "\nnum_policy_executions: " << options.num_policy_executions;
    strm << "\nexec_step_limit: " << options.exec_step_limit;
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
