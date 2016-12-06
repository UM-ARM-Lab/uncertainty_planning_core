#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <random>
#include <mutex>
#include <thread>
#include <atomic>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/simple_hierarchical_clustering.hpp>
#include <arc_utilities/simple_hausdorff_distance.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>
#include <sdf_tools/tagged_object_collision_map.hpp>
#include <sdf_tools/sdf.hpp>
#include <uncertainty_planning_core/simple_pid_controller.hpp>
#include <uncertainty_planning_core/simple_uncertainty_models.hpp>
#include <uncertainty_planning_core/uncertainty_planner_state.hpp>
#include <uncertainty_planning_core/simple_simulator_interface.hpp>
#include <uncertainty_planning_core/execution_policy.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <uncertainty_planning_core/uncertainty_contact_planning.hpp>
#include <uncertainty_planning_core/simple_robot_models.hpp>
#include <uncertainty_planning_core/simple_samplers.hpp>
#include <uncertainty_planning_core/uncertainty_planning_core.hpp>

using namespace uncertainty_planning_core;

bool uncertainty_planning_core::SaveSE2Policy(const SE2Policy& policy, const std::string& filename)
{
    return SE2PlanningSpace::SavePolicy(policy, filename);
}

SE2Policy uncertainty_planning_core::LoadSE2Policy(const std::string& filename)
{
    return SE2PlanningSpace::LoadPolicy(filename);
}

bool uncertainty_planning_core::SaveSE3Policy(const SE3Policy& policy, const std::string& filename)
{
    return SE3PlanningSpace::SavePolicy(policy, filename);
}

SE3Policy uncertainty_planning_core::LoadSE3Policy(const std::string& filename)
{
    return SE3PlanningSpace::LoadPolicy(filename);
}

bool uncertainty_planning_core::SaveLinkedPolicy(const LinkedPolicy& policy, const std::string& filename)
{
    return LinkedPlanningSpace::SavePolicy(policy, filename);
}

LinkedPolicy uncertainty_planning_core::LoadLinkedPolicy(const std::string& filename)
{
    return LinkedPlanningSpace::LoadPolicy(filename);
}

// SE2 Interface

std::vector<SE2Config, SE2ConfigAlloc> uncertainty_planning_core::DemonstrateSE2Simulator(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE2Robot& robot, const SE2SimulatorPtr& simulator, const SE2SamplerPtr& sampler, const SE2Config& start, const SE2Config& goal, ros::Publisher& display_debug_publisher)
{
    SE2PlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    const simple_simulator_interface::ForwardSimulationStepTrace<SE2Config, SE2ConfigAlloc> trace = planning_space.DemonstrateSimulator(start, goal, display_debug_publisher);
    return simple_simulator_interface::ExtractTrajectoryFromTrace(trace);
}

std::pair<SE2Policy, std::map<std::string, double>> uncertainty_planning_core::PlanSE2Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE2Robot& robot, const SE2SimulatorPtr& simulator, const SE2SamplerPtr& sampler, const SE2Config& start, const SE2Config& goal, ros::Publisher& display_debug_publisher)
{
    SE2PlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.Plan(start, goal, options.goal_bias, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, display_debug_publisher);
}

std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> uncertainty_planning_core::SimulateSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE2Robot& robot, const SE2SimulatorPtr& simulator, const SE2SamplerPtr& sampler, const SE2Policy& policy, const SE2Config& start, const SE2Config& goal, ros::Publisher& display_debug_publisher)
{
    SE2Policy working_policy = policy;
    SE2PlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, start, goal, options.num_policy_simulations, options.max_exec_actions, display_debug_publisher, true, 0.001);
}

std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> uncertainty_planning_core::ExecuteSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE2Robot& robot, const SE2SimulatorPtr& simulator, const SE2SamplerPtr& sampler, const SE2Policy& policy, const SE2Config& start, const SE2Config& goal, const std::function<std::vector<SE2Config, SE2ConfigAlloc>(const SE2Config&,  const SE2Config&, const double, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher)
{
    SE2Policy working_policy = policy;
    SE2PlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, start, goal, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_debug_publisher, false, 0.001);
}

// SE3 Interface

std::vector<SE3Config, SE3ConfigAlloc> uncertainty_planning_core::DemonstrateSE3Simulator(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE3Robot& robot, const SE3SimulatorPtr& simulator, const SE3SamplerPtr& sampler, const SE3Config& start, const SE3Config& goal, ros::Publisher& display_debug_publisher)
{
    SE3PlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    const simple_simulator_interface::ForwardSimulationStepTrace<SE3Config, SE3ConfigAlloc> trace = planning_space.DemonstrateSimulator(start, goal, display_debug_publisher);
    return simple_simulator_interface::ExtractTrajectoryFromTrace(trace);
}

std::pair<SE3Policy, std::map<std::string, double>> uncertainty_planning_core::PlanSE3Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE3Robot& robot, const SE3SimulatorPtr& simulator, const SE3SamplerPtr& sampler, const SE3Config& start, const SE3Config& goal, ros::Publisher& display_debug_publisher)
{
    SE3PlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.Plan(start, goal, options.goal_bias, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, display_debug_publisher);
}

std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> uncertainty_planning_core::SimulateSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE3Robot& robot, const SE3SimulatorPtr& simulator, const SE3SamplerPtr& sampler, const SE3Policy& policy, const SE3Config& start, const SE3Config& goal, ros::Publisher& display_debug_publisher)
{
    SE3Policy working_policy = policy;
    SE3PlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, start, goal, options.num_policy_simulations, options.max_exec_actions, display_debug_publisher, true, 0.001);
}

std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> uncertainty_planning_core::ExecuteSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE3Robot& robot, const SE3SimulatorPtr& simulator, const SE3SamplerPtr& sampler, const SE3Policy& policy, const SE3Config& start, const SE3Config& goal, const std::function<std::vector<SE3Config, SE3ConfigAlloc>(const SE3Config&,  const SE3Config&, const double, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher)
{
    SE3Policy working_policy = policy;
    SE3PlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, start, goal, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_debug_publisher, false, 0.001);
}

// Linked Interface

std::vector<LinkedConfig, LinkedConfigAlloc> uncertainty_planning_core::DemonstrateLinkedSimulator(const PLANNING_AND_EXECUTION_OPTIONS& options, const LinkedRobot& robot, const LinkedSimulatorPtr& simulator, const LinkedSamplerPtr& sampler, const LinkedConfig& start, const LinkedConfig& goal, ros::Publisher& display_debug_publisher)
{
    LinkedPlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    const simple_simulator_interface::ForwardSimulationStepTrace<LinkedConfig, LinkedConfigAlloc> trace = planning_space.DemonstrateSimulator(start, goal, display_debug_publisher);
    return simple_simulator_interface::ExtractTrajectoryFromTrace(trace);
}

std::pair<LinkedPolicy, std::map<std::string, double>> uncertainty_planning_core::PlanLinkedUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options, const LinkedRobot& robot, const LinkedSimulatorPtr& simulator, const LinkedSamplerPtr& sampler, const LinkedConfig& start, const LinkedConfig& goal, ros::Publisher& display_debug_publisher)
{
    LinkedPlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.Plan(start, goal, options.goal_bias, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, display_debug_publisher);
}

std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> uncertainty_planning_core::SimulateLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const LinkedRobot& robot, const LinkedSimulatorPtr& simulator, const LinkedSamplerPtr& sampler, const LinkedPolicy& policy, const LinkedConfig& start, const LinkedConfig& goal, ros::Publisher& display_debug_publisher)
{
    LinkedPolicy working_policy = policy;
    LinkedPlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, start, goal, options.num_policy_simulations, options.max_exec_actions, display_debug_publisher, true, 0.001);
}

std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> uncertainty_planning_core::ExecuteLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const LinkedRobot& robot, const LinkedSimulatorPtr& simulator, const LinkedSamplerPtr& sampler, const LinkedPolicy& policy, const LinkedConfig& start, const LinkedConfig& goal, const std::function<std::vector<LinkedConfig, LinkedConfigAlloc>(const LinkedConfig&,  const LinkedConfig&, const double, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher)
{
    LinkedPolicy working_policy = policy;
    LinkedPlanningSpace planning_space(options.clustering_type, false, options.debug_level, options.num_particles, options.step_size, options.step_duration, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, options.prng_seed);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, start, goal, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_debug_publisher, false, 0.001);
}
