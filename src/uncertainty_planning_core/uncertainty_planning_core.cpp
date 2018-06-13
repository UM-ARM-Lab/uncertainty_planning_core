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
#include <arc_utilities/eigen_helpers.hpp>
#include <uncertainty_planning_core/execution_policy.hpp>
#include <uncertainty_planning_core/simple_simulator_interface.hpp>
#include <uncertainty_planning_core/uncertainty_contact_planning.hpp>
#include <uncertainty_planning_core/uncertainty_planner_state.hpp>
#include <uncertainty_planning_core/uncertainty_planning_core.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

using namespace uncertainty_planning_core;

bool uncertainty_planning_core::SaveSE2Policy(const SE2Policy& policy, const std::string& filename)
{
    return SavePolicy<SE2Config, SE2ConfigSerializer, SE2ConfigAlloc>(policy, filename);
}

SE2Policy uncertainty_planning_core::LoadSE2Policy(const std::string& filename)
{
    return LoadPolicy<SE2Config, SE2ConfigSerializer, SE2ConfigAlloc>(filename);
}

bool uncertainty_planning_core::SaveSE3Policy(const SE3Policy& policy, const std::string& filename)
{
    return SavePolicy<SE3Config, SE3ConfigSerializer, SE3ConfigAlloc>(policy, filename);
}

SE3Policy uncertainty_planning_core::LoadSE3Policy(const std::string& filename)
{
    return LoadPolicy<SE3Config, SE3ConfigSerializer, SE3ConfigAlloc>(filename);
}

bool uncertainty_planning_core::SaveLinkedPolicy(const LinkedPolicy& policy, const std::string& filename)
{
    return SavePolicy<LinkedConfig, LinkedConfigSerializer, LinkedConfigAlloc>(policy, filename);
}

LinkedPolicy uncertainty_planning_core::LoadLinkedPolicy(const std::string& filename)
{
    return LoadPolicy<LinkedConfig, LinkedConfigSerializer, LinkedConfigAlloc>(filename);
}

bool uncertainty_planning_core::SaveVectorXdPolicy(const VectorXdPolicy& policy, const std::string& filename)
{
    return SavePolicy<VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>(policy, filename);
}

VectorXdPolicy uncertainty_planning_core::LoadVectorXdPolicy(const std::string& filename)
{
    return LoadPolicy<VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>(filename);
}

inline double uncertainty_planning_core::SE2UserGoalCheckWrapperFn(const UncertaintyPlanningState<SE2Config, SE2ConfigSerializer, SE2ConfigAlloc>& state,
                                                                   const SE2UserGoalConfigCheckFn& user_goal_config_check_fn)
{
    return UserGoalCheckWrapperFn(state, user_goal_config_check_fn);
}

inline double uncertainty_planning_core::SE3UserGoalCheckWrapperFn(const UncertaintyPlanningState<SE3Config, SE3ConfigSerializer, SE3ConfigAlloc>& state,
                                                                   const SE3UserGoalConfigCheckFn& user_goal_config_check_fn)
{
    return UserGoalCheckWrapperFn(state, user_goal_config_check_fn);
}

inline double uncertainty_planning_core::LinkedUserGoalCheckWrapperFn(const UncertaintyPlanningState<LinkedConfig, LinkedConfigSerializer, LinkedConfigAlloc>& state,
                                                                      const LinkedUserGoalConfigCheckFn& user_goal_config_check_fn)
{
    return UserGoalCheckWrapperFn(state, user_goal_config_check_fn);
}

inline double uncertainty_planning_core::VectorXdUserGoalCheckWrapperFn(const UncertaintyPlanningState<VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>& state,
                                                                        const VectorXdUserGoalConfigCheckFn& user_goal_config_check_fn)
{
    return UserGoalCheckWrapperFn(state, user_goal_config_check_fn);
}

// SE2 Interface

std::vector<SE2Config, SE2ConfigAlloc>
uncertainty_planning_core::DemonstrateSE2Simulator(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                   const SE2RobotPtr& robot,
                                                   const SE2SimulatorPtr& simulator,
                                                   const SE2SamplerPtr& sampler,
                                                   const SE2ClusteringPtr& clustering,
                                                   const SE2Config& start,
                                                   const SE2Config& goal,
                                                   const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                   const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE2PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const simple_simulator_interface::ForwardSimulationStepTrace<SE2Config, SE2ConfigAlloc> trace = planning_space.DemonstrateSimulator(start, goal, display_fn);
    return simple_simulator_interface::ExtractTrajectoryFromTrace(trace);
}

std::pair<SE2Policy, std::map<std::string, double>>
uncertainty_planning_core::PlanSE2Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                              const SE2RobotPtr& robot,
                                              const SE2SimulatorPtr& simulator,
                                              const SE2SamplerPtr& sampler,
                                              const SE2ClusteringPtr& clustering,
                                              const SE2Config& start,
                                              const SE2Config& goal,
                                              const double policy_marker_size,
                                              const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                              const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE2PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.PlanGoalState(start, goal, options.goal_bias, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, policy_marker_size, options.p_goal_reached_termination_threshold, display_fn);
}

std::pair<SE2Policy, std::map<std::string, double>>
uncertainty_planning_core::PlanSE2Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                              const SE2RobotPtr& robot,
                                              const SE2SimulatorPtr& simulator,
                                              const SE2SamplerPtr& sampler,
                                              const SE2ClusteringPtr& clustering,
                                              const SE2Config& start,
                                              const SE2UserGoalStateCheckFn& user_goal_check_fn,
                                              const double policy_marker_size,
                                              const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                              const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE2PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.PlanGoalSampling(start, options.goal_bias, user_goal_check_fn, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, policy_marker_size, options.p_goal_reached_termination_threshold, display_fn);
}

std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::SimulateSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                        const SE2RobotPtr& robot,
                                                        const SE2SimulatorPtr& simulator,
                                                        const SE2SamplerPtr& sampler,
                                                        const SE2ClusteringPtr& clustering,
                                                        const SE2Policy& policy,
                                                        const bool allow_branch_jumping,
                                                        const bool link_runtime_states_to_planned_parent,
                                                        const SE2Config& start,
                                                        const SE2Config& goal,
                                                        const double policy_marker_size,
                                                        const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                        const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE2Policy working_policy = policy;
    SE2PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, goal, options.num_policy_simulations, options.max_exec_actions, display_fn, policy_marker_size, true, 0.001);
}

std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::ExecuteSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                       const SE2RobotPtr& robot,
                                                       const SE2SimulatorPtr& simulator,
                                                       const SE2SamplerPtr& sampler,
                                                       const SE2ClusteringPtr& clustering,
                                                       const SE2Policy& policy,
                                                       const bool allow_branch_jumping,
                                                       const bool link_runtime_states_to_planned_parent,
                                                       const SE2Config& start,
                                                       const SE2Config& goal,
                                                       const double policy_marker_size,
                                                       const std::function<std::vector<SE2Config, SE2ConfigAlloc>(const SE2Config&,
                                                                                                                  const SE2Config&,
                                                                                                                  const SE2Config&,
                                                                                                                  const bool,
                                                                                                                  const bool)>& robot_execution_fn,
                                                       const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                       const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE2Policy working_policy = policy;
    SE2PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, goal, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_fn, policy_marker_size, false, 0.001);
}

std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::SimulateSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                        const SE2RobotPtr& robot,
                                                        const SE2SimulatorPtr& simulator,
                                                        const SE2SamplerPtr& sampler,
                                                        const SE2ClusteringPtr& clustering,
                                                        const SE2Policy& policy,
                                                        const bool allow_branch_jumping,
                                                        const bool link_runtime_states_to_planned_parent,
                                                        const SE2Config& start,
                                                        const SE2UserGoalConfigCheckFn& user_goal_check_fn,
                                                        const double policy_marker_size,
                                                        const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                        const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE2Policy working_policy = policy;
    SE2PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, user_goal_check_fn, options.num_policy_simulations, options.max_exec_actions, display_fn, policy_marker_size, true, 0.001);
}

std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::ExecuteSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                       const SE2RobotPtr& robot,
                                                       const SE2SimulatorPtr& simulator,
                                                       const SE2SamplerPtr& sampler,
                                                       const SE2ClusteringPtr& clustering,
                                                       const SE2Policy& policy,
                                                       const bool allow_branch_jumping,
                                                       const bool link_runtime_states_to_planned_parent,
                                                       const SE2Config& start,
                                                       const SE2UserGoalConfigCheckFn& user_goal_check_fn,
                                                       const double policy_marker_size,
                                                       const std::function<std::vector<SE2Config, SE2ConfigAlloc>(const SE2Config&,
                                                                                                                  const SE2Config&,
                                                                                                                  const SE2Config&,
                                                                                                                  const bool,
                                                                                                                  const bool)>& robot_execution_fn,
                                                       const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                       const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE2Policy working_policy = policy;
    SE2PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, user_goal_check_fn, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_fn, policy_marker_size, false, 0.001);
}

// SE3 Interface

std::vector<SE3Config, SE3ConfigAlloc>
uncertainty_planning_core::DemonstrateSE3Simulator(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                   const SE3RobotPtr& robot,
                                                   const SE3SimulatorPtr& simulator,
                                                   const SE3SamplerPtr& sampler,
                                                   const SE3ClusteringPtr& clustering,
                                                   const SE3Config& start,
                                                   const SE3Config& goal,
                                                   const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                   const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE3PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const simple_simulator_interface::ForwardSimulationStepTrace<SE3Config, SE3ConfigAlloc> trace = planning_space.DemonstrateSimulator(start, goal, display_fn);
    return simple_simulator_interface::ExtractTrajectoryFromTrace(trace);
}

std::pair<SE3Policy, std::map<std::string, double>>
uncertainty_planning_core::PlanSE3Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                              const SE3RobotPtr& robot,
                                              const SE3SimulatorPtr& simulator,
                                              const SE3SamplerPtr& sampler,
                                              const SE3ClusteringPtr& clustering,
                                              const SE3Config& start,
                                              const SE3Config& goal,
                                              const double policy_marker_size,
                                              const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                              const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE3PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.PlanGoalState(start, goal, options.goal_bias, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, policy_marker_size, options.p_goal_reached_termination_threshold, display_fn);
}

std::pair<SE3Policy, std::map<std::string, double>>
uncertainty_planning_core::PlanSE3Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                              const SE3RobotPtr& robot,
                                              const SE3SimulatorPtr& simulator,
                                              const SE3SamplerPtr& sampler,
                                              const SE3ClusteringPtr& clustering,
                                              const SE3Config& start,
                                              const SE3UserGoalStateCheckFn& user_goal_check_fn,
                                              const double policy_marker_size,
                                              const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                              const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE3PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.PlanGoalSampling(start, options.goal_bias, user_goal_check_fn, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, policy_marker_size, options.p_goal_reached_termination_threshold, display_fn);
}

std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::SimulateSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                        const SE3RobotPtr& robot,
                                                        const SE3SimulatorPtr& simulator,
                                                        const SE3SamplerPtr& sampler,
                                                        const SE3ClusteringPtr& clustering,
                                                        const SE3Policy& policy,
                                                        const bool allow_branch_jumping,
                                                        const bool link_runtime_states_to_planned_parent,
                                                        const SE3Config& start,
                                                        const SE3Config& goal,
                                                        const double policy_marker_size,
                                                        const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                        const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE3Policy working_policy = policy;
    SE3PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, goal, options.num_policy_simulations, options.max_exec_actions, display_fn, policy_marker_size, true, 0.001);
}

std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::ExecuteSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                       const SE3RobotPtr& robot,
                                                       const SE3SimulatorPtr& simulator,
                                                       const SE3SamplerPtr& sampler,
                                                       const SE3ClusteringPtr& clustering,
                                                       const SE3Policy& policy,
                                                       const bool allow_branch_jumping,
                                                       const bool link_runtime_states_to_planned_parent,
                                                       const SE3Config& start,
                                                       const SE3Config& goal,
                                                       const double policy_marker_size,
                                                       const std::function<std::vector<SE3Config, SE3ConfigAlloc>(const SE3Config&,
                                                                                                                  const SE3Config&,
                                                                                                                  const SE3Config&,
                                                                                                                  const bool,
                                                                                                                  const bool)>& robot_execution_fn,
                                                       const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                       const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE3Policy working_policy = policy;
    SE3PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, goal, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_fn, policy_marker_size, false, 0.001);
}

std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::SimulateSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                        const SE3RobotPtr& robot,
                                                        const SE3SimulatorPtr& simulator,
                                                        const SE3SamplerPtr& sampler,
                                                        const SE3ClusteringPtr& clustering,
                                                        const SE3Policy& policy,
                                                        const bool allow_branch_jumping,
                                                        const bool link_runtime_states_to_planned_parent,
                                                        const SE3Config& start,
                                                        const SE3UserGoalConfigCheckFn& user_goal_check_fn,
                                                        const double policy_marker_size,
                                                        const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                        const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE3Policy working_policy = policy;
    SE3PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, user_goal_check_fn, options.num_policy_simulations, options.max_exec_actions, display_fn, policy_marker_size, true, 0.001);
}

std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::ExecuteSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                       const SE3RobotPtr& robot,
                                                       const SE3SimulatorPtr& simulator,
                                                       const SE3SamplerPtr& sampler,
                                                       const SE3ClusteringPtr& clustering,
                                                       const SE3Policy& policy,
                                                       const bool allow_branch_jumping,
                                                       const bool link_runtime_states_to_planned_parent,
                                                       const SE3Config& start,
                                                       const SE3UserGoalConfigCheckFn& user_goal_check_fn,
                                                       const double policy_marker_size,
                                                       const std::function<std::vector<SE3Config, SE3ConfigAlloc>(const SE3Config&,
                                                                                                                  const SE3Config&,
                                                                                                                  const SE3Config&,
                                                                                                                  const bool,
                                                                                                                  const bool)>& robot_execution_fn,
                                                       const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                       const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    SE3Policy working_policy = policy;
    SE3PlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, user_goal_check_fn, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_fn, policy_marker_size, false, 0.001);
}

// Linked Interface

std::vector<LinkedConfig, LinkedConfigAlloc>
uncertainty_planning_core::DemonstrateLinkedSimulator(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                      const LinkedRobotPtr& robot,
                                                      const LinkedSimulatorPtr& simulator,
                                                      const LinkedSamplerPtr& sampler,
                                                      const LinkedClusteringPtr& clustering,
                                                      const LinkedConfig& start,
                                                      const LinkedConfig& goal,
                                                      const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                      const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    LinkedPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const simple_simulator_interface::ForwardSimulationStepTrace<LinkedConfig, LinkedConfigAlloc> trace = planning_space.DemonstrateSimulator(start, goal, display_fn);
    return simple_simulator_interface::ExtractTrajectoryFromTrace(trace);
}

std::pair<LinkedPolicy, std::map<std::string, double>>
uncertainty_planning_core::PlanLinkedUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                 const LinkedRobotPtr& robot,
                                                 const LinkedSimulatorPtr& simulator,
                                                 const LinkedSamplerPtr& sampler,
                                                 const LinkedClusteringPtr& clustering,
                                                 const LinkedConfig& start,
                                                 const LinkedConfig& goal,
                                                 const double policy_marker_size,
                                                 const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                 const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    LinkedPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.PlanGoalState(start, goal, options.goal_bias, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, policy_marker_size, options.p_goal_reached_termination_threshold, display_fn);
}

std::pair<LinkedPolicy, std::map<std::string, double>>
uncertainty_planning_core::PlanLinkedUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                 const LinkedRobotPtr& robot,
                                                 const LinkedSimulatorPtr& simulator,
                                                 const LinkedSamplerPtr& sampler,
                                                 const LinkedClusteringPtr& clustering,
                                                 const LinkedConfig& start,
                                                 const LinkedUserGoalStateCheckFn& user_goal_check_fn,
                                                 const double policy_marker_size,
                                                 const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                 const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    LinkedPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.PlanGoalSampling(start, options.goal_bias, user_goal_check_fn, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, policy_marker_size, options.p_goal_reached_termination_threshold, display_fn);
}

std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::SimulateLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                           const LinkedRobotPtr& robot,
                                                           const LinkedSimulatorPtr& simulator,
                                                           const LinkedSamplerPtr& sampler,
                                                           const LinkedClusteringPtr& clustering,
                                                           const LinkedPolicy& policy,
                                                           const bool allow_branch_jumping,
                                                           const bool link_runtime_states_to_planned_parent,
                                                           const LinkedConfig& start,
                                                           const LinkedConfig& goal,
                                                           const double policy_marker_size,
                                                           const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                           const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    LinkedPolicy working_policy = policy;
    LinkedPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, goal, options.num_policy_simulations, options.max_exec_actions, display_fn, policy_marker_size, true, 0.001);
}

std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::ExecuteLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                          const LinkedRobotPtr& robot,
                                                          const LinkedSimulatorPtr& simulator,
                                                          const LinkedSamplerPtr& sampler,
                                                          const LinkedClusteringPtr& clustering,
                                                          const LinkedPolicy& policy,
                                                          const bool allow_branch_jumping,
                                                          const bool link_runtime_states_to_planned_parent,
                                                          const LinkedConfig& start,
                                                          const LinkedConfig& goal,
                                                          const double policy_marker_size,
                                                          const std::function<std::vector<LinkedConfig, LinkedConfigAlloc>(const LinkedConfig&,
                                                                                                                           const LinkedConfig&,
                                                                                                                           const LinkedConfig&,
                                                                                                                           const bool,
                                                                                                                           const bool)>& robot_execution_fn,
                                                          const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                          const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    LinkedPolicy working_policy = policy;
    LinkedPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, goal, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_fn, policy_marker_size, false, 0.001);
}

std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::SimulateLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                           const LinkedRobotPtr& robot,
                                                           const LinkedSimulatorPtr& simulator,
                                                           const LinkedSamplerPtr& sampler,
                                                           const LinkedClusteringPtr& clustering,
                                                           const LinkedPolicy& policy,
                                                           const bool allow_branch_jumping,
                                                           const bool link_runtime_states_to_planned_parent,
                                                           const LinkedConfig& start,
                                                           const LinkedUserGoalConfigCheckFn& user_goal_check_fn,
                                                           const double policy_marker_size,
                                                           const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                           const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    LinkedPolicy working_policy = policy;
    LinkedPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, user_goal_check_fn, options.num_policy_simulations, options.max_exec_actions, display_fn, policy_marker_size, true, 0.001);
}

std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::ExecuteLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                          const LinkedRobotPtr& robot,
                                                          const LinkedSimulatorPtr& simulator,
                                                          const LinkedSamplerPtr& sampler,
                                                          const LinkedClusteringPtr& clustering,
                                                          const LinkedPolicy& policy,
                                                          const bool allow_branch_jumping,
                                                          const bool link_runtime_states_to_planned_parent,
                                                          const LinkedConfig& start,
                                                          const LinkedUserGoalConfigCheckFn& user_goal_check_fn,
                                                          const double policy_marker_size,
                                                          const std::function<std::vector<LinkedConfig, LinkedConfigAlloc>(const LinkedConfig&,
                                                                                                                           const LinkedConfig&,
                                                                                                                           const LinkedConfig&,
                                                                                                                           const bool,
                                                                                                                           const bool)>& robot_execution_fn,
                                                          const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                          const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    LinkedPolicy working_policy = policy;
    LinkedPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, user_goal_check_fn, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_fn, policy_marker_size, false, 0.001);
}

// VectorXd Interface

std::vector<VectorXdConfig, VectorXdConfigAlloc>
uncertainty_planning_core::DemonstrateVectorXdSimulator(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                        const VectorXdRobotPtr& robot,
                                                        const VectorXdSimulatorPtr& simulator,
                                                        const VectorXdSamplerPtr& sampler,
                                                        const VectorXdClusteringPtr& clustering,
                                                        const VectorXdConfig& start,
                                                        const VectorXdConfig& goal,
                                                        const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                        const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    VectorXdPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const simple_simulator_interface::ForwardSimulationStepTrace<VectorXdConfig, VectorXdConfigAlloc> trace = planning_space.DemonstrateSimulator(start, goal, display_fn);
    return simple_simulator_interface::ExtractTrajectoryFromTrace(trace);
}

std::pair<VectorXdPolicy, std::map<std::string, double>>
uncertainty_planning_core::PlanVectorXdUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                   const VectorXdRobotPtr& robot,
                                                   const VectorXdSimulatorPtr& simulator,
                                                   const VectorXdSamplerPtr& sampler,
                                                   const VectorXdClusteringPtr& clustering,
                                                   const VectorXdConfig& start,
                                                   const VectorXdConfig& goal,
                                                   const double policy_marker_size,
                                                   const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                   const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    VectorXdPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.PlanGoalState(start, goal, options.goal_bias, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, policy_marker_size, options.p_goal_reached_termination_threshold, display_fn);
}

std::pair<VectorXdPolicy, std::map<std::string, double>>
uncertainty_planning_core::PlanVectorXdUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                   const VectorXdRobotPtr& robot,
                                                   const VectorXdSimulatorPtr& simulator,
                                                   const VectorXdSamplerPtr& sampler,
                                                   const VectorXdClusteringPtr& clustering,
                                                   const VectorXdConfig& start,
                                                   const VectorXdUserGoalStateCheckFn& user_goal_check_fn,
                                                   const double policy_marker_size,
                                                   const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                   const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    VectorXdPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(options.planner_time_limit);
    return planning_space.PlanGoalSampling(start, options.goal_bias, user_goal_check_fn, planner_time_limit, options.edge_attempt_count, options.policy_action_attempt_count, options.use_contact, options.use_reverse, options.use_spur_actions, policy_marker_size, options.p_goal_reached_termination_threshold, display_fn);
}

std::pair<VectorXdPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::SimulateVectorXdUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                             const VectorXdRobotPtr& robot,
                                                             const VectorXdSimulatorPtr& simulator,
                                                             const VectorXdSamplerPtr& sampler,
                                                             const VectorXdClusteringPtr& clustering,
                                                             const VectorXdPolicy& policy,
                                                             const bool allow_branch_jumping,
                                                             const bool link_runtime_states_to_planned_parent,
                                                             const VectorXdConfig& start,
                                                             const VectorXdConfig& goal,
                                                             const double policy_marker_size,
                                                             const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                             const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    VectorXdPolicy working_policy = policy;
    VectorXdPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, goal, options.num_policy_simulations, options.max_exec_actions, display_fn, policy_marker_size, true, 0.001);
}

std::pair<VectorXdPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::ExecuteVectorXdUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                            const VectorXdRobotPtr& robot,
                                                            const VectorXdSimulatorPtr& simulator,
                                                            const VectorXdSamplerPtr& sampler,
                                                            const VectorXdClusteringPtr& clustering,
                                                            const VectorXdPolicy& policy,
                                                            const bool allow_branch_jumping,
                                                            const bool link_runtime_states_to_planned_parent,
                                                            const VectorXdConfig& start,
                                                            const VectorXdConfig& goal,
                                                            const double policy_marker_size,
                                                            const std::function<std::vector<VectorXdConfig, VectorXdConfigAlloc>(const VectorXdConfig&,
                                                                                                                                 const VectorXdConfig&,
                                                                                                                                 const VectorXdConfig&,
                                                                                                                                 const bool,
                                                                                                                                 const bool)>& robot_execution_fn,
                                                            const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                            const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    VectorXdPolicy working_policy = policy;
    VectorXdPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, goal, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_fn, policy_marker_size, false, 0.001);
}

std::pair<VectorXdPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::SimulateVectorXdUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                             const VectorXdRobotPtr& robot,
                                                             const VectorXdSimulatorPtr& simulator,
                                                             const VectorXdSamplerPtr& sampler,
                                                             const VectorXdClusteringPtr& clustering,
                                                             const VectorXdPolicy& policy,
                                                             const bool allow_branch_jumping,
                                                             const bool link_runtime_states_to_planned_parent,
                                                             const VectorXdConfig& start,
                                                             const VectorXdUserGoalConfigCheckFn& user_goal_check_fn,
                                                             const double policy_marker_size,
                                                             const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                             const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    VectorXdPolicy working_policy = policy;
    VectorXdPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, user_goal_check_fn, options.num_policy_simulations, options.max_exec_actions, display_fn, policy_marker_size, true, 0.001);
}

std::pair<VectorXdPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
uncertainty_planning_core::ExecuteVectorXdUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                                            const VectorXdRobotPtr& robot,
                                                            const VectorXdSimulatorPtr& simulator,
                                                            const VectorXdSamplerPtr& sampler,
                                                            const VectorXdClusteringPtr& clustering,
                                                            const VectorXdPolicy& policy,
                                                            const bool allow_branch_jumping,
                                                            const bool link_runtime_states_to_planned_parent,
                                                            const VectorXdConfig& start,
                                                            const VectorXdUserGoalConfigCheckFn& user_goal_check_fn,
                                                            const double policy_marker_size,
                                                            const std::function<std::vector<VectorXdConfig, VectorXdConfigAlloc>(const VectorXdConfig&,
                                                                                                                                 const VectorXdConfig&,
                                                                                                                                 const VectorXdConfig&,
                                                                                                                                 const bool,
                                                                                                                                 const bool)>& robot_execution_fn,
                                                            const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                                            const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn)
{
    VectorXdPolicy working_policy = policy;
    VectorXdPlanningSpace planning_space(options.debug_level, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.feasibility_alpha, options.variance_alpha, options.connect_after_first_solution, robot, sampler, simulator, clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(working_policy, allow_branch_jumping, link_runtime_states_to_planned_parent, start, user_goal_check_fn, robot_execution_fn, options.num_policy_executions, options.max_policy_exec_time, display_fn, policy_marker_size, false, 0.001);
}
