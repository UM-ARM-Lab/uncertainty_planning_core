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

#ifndef UNCERTAINTY_PLANNING_CORE_HPP
#define UNCERTAINTY_PLANNING_CORE_HPP

#ifndef PRNG_TYPE
    #define PRNG_TYPE std::mt19937_64
#endif

namespace uncertainty_planning_core
{
    struct PLANNING_AND_EXECUTION_OPTIONS
    {
        uncertainty_contact_planning::SPATIAL_FEATURE_CLUSTERING_TYPE clustering_type;
        uint64_t prng_seed;
        // Time limits
        double planner_time_limit;
        // Standard planner control params
        double goal_bias;
        double step_size;
        double step_duration;
        double goal_probability_threshold;
        double goal_distance_threshold;
        double connect_after_first_solution;
        // Distance function control params/weights
        double signature_matching_threshold;
        double distance_clustering_threshold;
        double feasibility_alpha;
        double variance_alpha;
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
        double max_policy_exec_time;
        // Control flags
        int32_t debug_level;
        bool use_contact;
        bool use_reverse;
        bool use_spur_actions;
        // Log & data files
        std::string planner_log_file;
        std::string policy_log_file;
        std::string planned_policy_file;
        std::string executed_policy_file;
    };

    inline PLANNING_AND_EXECUTION_OPTIONS GetOptions(const PLANNING_AND_EXECUTION_OPTIONS& initial_options)
    {
        PLANNING_AND_EXECUTION_OPTIONS options = initial_options;
        // Get options via ROS params
        ros::NodeHandle nhp("~");
        const int32_t prng_seed_init = (int32_t)nhp.param(std::string("prng_seed_init"), -1);
        if (prng_seed_init >= 0)
        {
            arc_helpers::SplitMix64PRNG seed_gen((uint64_t)prng_seed_init);
            options.prng_seed = seed_gen();
        }
        else
        {
            options.prng_seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        }
        options.planner_time_limit = nhp.param(std::string("planner_time_limit"), options.planner_time_limit);
        options.goal_bias = nhp.param(std::string("goal_bias"), options.goal_bias);
        options.step_size = nhp.param(std::string("step_size"), options.step_size);
        options.step_duration = nhp.param(std::string("step_duration"), options.step_duration);
        options.goal_probability_threshold = nhp.param(std::string("goal_probability_threshold"), options.goal_probability_threshold);
        options.goal_distance_threshold = nhp.param(std::string("goal_distance_threshold"), options.goal_distance_threshold);
        options.connect_after_first_solution = nhp.param(std::string("connect_after_first_solution"), options.connect_after_first_solution);
        options.feasibility_alpha = nhp.param(std::string("feasibility_alpha"), options.feasibility_alpha);
        options.variance_alpha = nhp.param(std::string("variance_alpha"), options.variance_alpha);
        options.num_particles = (uint32_t)nhp.param(std::string("num_particles"), (int)options.num_particles);
        options.planner_log_file = nhp.param(std::string("planner_log_file"), options.planner_log_file);
        options.planned_policy_file = nhp.param(std::string("planned_policy_file"), options.planned_policy_file);
        options.clustering_type = uncertainty_contact_planning::ParseSpatialFeatureClusteringType(nhp.param(std::string("clustering_type"), uncertainty_contact_planning::PrintSpatialFeatureClusteringType(options.clustering_type)));
        options.signature_matching_threshold = nhp.param(std::string("signature_matching_threshold"), options.signature_matching_threshold);
        options.distance_clustering_threshold = nhp.param(std::string("distance_clustering_threshold"), options.distance_clustering_threshold);
        options.policy_action_attempt_count = (uint32_t)nhp.param(std::string("policy_action_attempt_count"), (int)options.policy_action_attempt_count);
        options.debug_level = nhp.param(std::string("debug_level"), options.debug_level);
        options.use_contact = nhp.param(std::string("use_contact"), options.use_contact);
        options.use_reverse = nhp.param(std::string("use_reverse"), options.use_reverse);
        options.num_policy_simulations = (uint32_t)nhp.param(std::string("num_policy_simulations"), (int)options.num_policy_simulations);
        options.num_policy_executions = (uint32_t)nhp.param(std::string("num_policy_executions"), (int)options.num_policy_executions);
        options.policy_log_file = nhp.param(std::string("policy_log_file"), options.policy_log_file);
        options.executed_policy_file = nhp.param(std::string("executed_policy_file"), options.executed_policy_file);
        options.max_exec_actions = (uint32_t)nhp.param(std::string("max_exec_actions"), (int)options.max_exec_actions);
        options.max_policy_exec_time = nhp.param(std::string("max_policy_exec_time"), options.max_policy_exec_time);
        options.policy_action_attempt_count = (uint32_t)nhp.param(std::string("policy_action_attempt_count"), (int)options.policy_action_attempt_count);
        return options;
    }

    // Typedefs to make things easier to read
    typedef PRNG_TYPE PRNG;

    typedef Eigen::Matrix<double, 3, 1> SE2Config;
    typedef std::allocator<Eigen::Matrix<double, 3, 1>> SE2ConfigAlloc;
    typedef simple_robot_models::EigenMatrixD31Serializer SE2ConfigSerializer;
    typedef execution_policy::ExecutionPolicy<SE2Config, SE2ConfigSerializer, SE2ConfigAlloc> SE2Policy;
    typedef simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator SE2ActuatorModel;
    typedef simple_robot_models::SimpleSE2Robot SE2Robot;
    typedef simple_simulator_interface::SimulatorInterface<SE2Robot, SE2Config, PRNG, SE2ConfigAlloc> SE2Simulator;
    typedef std::shared_ptr<SE2Simulator> SE2SimulatorPtr;
    typedef simple_samplers::SimpleBaseSampler<SE2Config, PRNG> SE2Sampler;
    typedef std::shared_ptr<SE2Sampler> SE2SamplerPtr;
    typedef uncertainty_contact_planning::UncertaintyPlanningSpace<SE2Robot, SE2Config, SE2ConfigSerializer, SE2ConfigAlloc, PRNG> SE2PlanningSpace;

    typedef Eigen::Isometry3d SE3Config;
    typedef Eigen::aligned_allocator<Eigen::Isometry3d> SE3ConfigAlloc;
    typedef simple_robot_models::EigenIsometry3dSerializer SE3ConfigSerializer;
    typedef execution_policy::ExecutionPolicy<SE3Config, SE3ConfigSerializer, SE3ConfigAlloc> SE3Policy;
    typedef simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator SE3ActuatorModel;
    typedef simple_robot_models::SimpleSE3Robot SE3Robot;
    typedef simple_simulator_interface::SimulatorInterface<SE3Robot, SE3Config, PRNG, SE3ConfigAlloc> SE3Simulator;
    typedef std::shared_ptr<SE3Simulator> SE3SimulatorPtr;
    typedef simple_samplers::SimpleBaseSampler<SE3Config, PRNG> SE3Sampler;
    typedef std::shared_ptr<SE3Sampler> SE3SamplerPtr;
    typedef uncertainty_contact_planning::UncertaintyPlanningSpace<SE3Robot, SE3Config, SE3ConfigSerializer, SE3ConfigAlloc, PRNG> SE3PlanningSpace;

    typedef simple_robot_models::SimpleLinkedConfiguration LinkedConfig;
    typedef std::allocator<LinkedConfig> LinkedConfigAlloc;
    typedef simple_robot_models::SimpleLinkedConfigurationSerializer LinkedConfigSerializer;
    typedef execution_policy::ExecutionPolicy<LinkedConfig, LinkedConfigSerializer, LinkedConfigAlloc> LinkedPolicy;
    typedef simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator LinkedActuatorModel;
    typedef simple_robot_models::SimpleLinkedRobot<LinkedActuatorModel> LinkedRobot;
    typedef simple_simulator_interface::SimulatorInterface<LinkedRobot, LinkedConfig, PRNG, LinkedConfigAlloc> LinkedSimulator;
    typedef std::shared_ptr<LinkedSimulator> LinkedSimulatorPtr;
    typedef simple_samplers::SimpleBaseSampler<LinkedConfig, PRNG> LinkedSampler;
    typedef std::shared_ptr<LinkedSampler> LinkedSamplerPtr;
    typedef uncertainty_contact_planning::UncertaintyPlanningSpace<LinkedRobot, LinkedConfig, LinkedConfigSerializer, LinkedConfigAlloc, PRNG> LinkedPlanningSpace;

    // Policy saving and loading

    bool SaveSE2Policy(const SE2Policy& policy, const std::string& filename);

    SE2Policy LoadSE2Policy(const std::string& filename);

    bool SaveSE3Policy(const SE3Policy& policy, const std::string& filename);

    SE3Policy LoadSE3Policy(const std::string& filename);

    bool SaveLinkedPolicy(const LinkedPolicy& policy, const std::string& filename);

    LinkedPolicy LoadLinkedPolicy(const std::string& filename);

    // SE2 Interface

    std::vector<SE2Config, SE2ConfigAlloc> DemonstrateSE2Simulator(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE2Robot& robot, const SE2SimulatorPtr& simulator, const SE2SamplerPtr& sampler, const SE2Config& start, const SE2Config& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE2Policy, std::map<std::string, double>> PlanSE2Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE2Robot& robot, const SE2SimulatorPtr& simulator, const SE2SamplerPtr& sampler, const SE2Config& start, const SE2Config& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> SimulateSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE2Robot& robot, const SE2SimulatorPtr& simulator, const SE2SamplerPtr& sampler, const SE2Policy& policy, const SE2Config& start, const SE2Config& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> ExecuteSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE2Robot& robot, const SE2SimulatorPtr& simulator, const SE2SamplerPtr& sampler, const SE2Policy& policy, const SE2Config& start, const SE2Config& goal, const std::function<std::vector<SE2Config, SE2ConfigAlloc>(const SE2Config&,  const SE2Config&, const double, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher);

    // SE3 Interface

    std::vector<SE3Config, SE3ConfigAlloc> DemonstrateSE3Simulator(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE3Robot& robot, const SE3SimulatorPtr& simulator, const SE3SamplerPtr& sampler, const SE3Config& start, const SE3Config& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE3Policy, std::map<std::string, double>> PlanSE3Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE3Robot& robot, const SE3SimulatorPtr& simulator, const SE3SamplerPtr& sampler, const SE3Config& start, const SE3Config& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> SimulateSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE3Robot& robot, const SE3SimulatorPtr& simulator, const SE3SamplerPtr& sampler, const SE3Policy& policy, const SE3Config& start, const SE3Config& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> ExecuteSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const SE3Robot& robot, const SE3SimulatorPtr& simulator, const SE3SamplerPtr& sampler, const SE3Policy& policy, const SE3Config& start, const SE3Config& goal, const std::function<std::vector<SE3Config, SE3ConfigAlloc>(const SE3Config&,  const SE3Config&, const double, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher);

    // Linked Interface

    std::vector<LinkedConfig, LinkedConfigAlloc> DemonstrateLinkedSimulator(const PLANNING_AND_EXECUTION_OPTIONS& options, const LinkedRobot& robot, const LinkedSimulatorPtr& simulator, const LinkedSamplerPtr& sampler, const LinkedConfig& start, const LinkedConfig& goal, ros::Publisher& display_debug_publisher);

    std::pair<LinkedPolicy, std::map<std::string, double>> PlanLinkedUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options, const LinkedRobot& robot, const LinkedSimulatorPtr& simulator, const LinkedSamplerPtr& sampler, const LinkedConfig& start, const LinkedConfig& goal, ros::Publisher& display_debug_publisher);

    std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> SimulateLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const LinkedRobot& robot, const LinkedSimulatorPtr& simulator, const LinkedSamplerPtr& sampler, const LinkedPolicy& policy, const LinkedConfig& start, const LinkedConfig& goal, ros::Publisher& display_debug_publisher);

    std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> ExecuteLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options, const LinkedRobot& robot, const LinkedSimulatorPtr& simulator, const LinkedSamplerPtr& sampler, const LinkedPolicy& policy, const LinkedConfig& start, const LinkedConfig& goal, const std::function<std::vector<LinkedConfig, LinkedConfigAlloc>(const LinkedConfig&,  const LinkedConfig&, const double, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher);

    inline std::ostream& operator<<(std::ostream& strm, const PLANNING_AND_EXECUTION_OPTIONS& options)
    {
        strm << "OPTIONS:";
        strm << "\nclustering_type: " << uncertainty_contact_planning::PrintSpatialFeatureClusteringType(options.clustering_type);
        strm << "\nprng_seed: " << options.prng_seed;
        strm << "\nplanner_time_limit: " << options.planner_time_limit;
        strm << "\ngoal_bias: " << options.goal_bias;
        strm << "\nstep_size: " << options.step_size;
        strm << "\nstep_duration: " << options.step_duration;
        strm << "\ngoal_probability_threshold: " << options.goal_probability_threshold;
        strm << "\ngoal_distance_threshold: " << options.goal_distance_threshold;
        strm << "\nconnect_after_first_solution: " << options.connect_after_first_solution;
        strm << "\nsignature_matching_threshold: " << options.signature_matching_threshold;
        strm << "\ndistance_clustering_threshold: " << options.distance_clustering_threshold;
        strm << "\nfeasibility_alpha: " << options.feasibility_alpha;
        strm << "\nvariance_alpha: " << options.variance_alpha;
        strm << "\nedge_attempt_count: " << options.edge_attempt_count;
        strm << "\npolicy_action_attempt_count: " << options.policy_action_attempt_count;
        strm << "\nnum_particles: " << options.num_particles;
        strm << "\nnum_policy_simulations: " << options.num_policy_simulations;
        strm << "\nnum_policy_executions: " << options.num_policy_executions;
        strm << "\nmax_exec_actions: " << options.max_exec_actions;
        strm << "\nmax_policy_exec_time: " << options.max_policy_exec_time;
        strm << "\ndebug_level: " << options.debug_level;
        strm << "\nuse_contact: " << options.use_contact;
        strm << "\nuse_reverse: " << options.use_reverse;
        strm << "\nuse_spur_actions: " << options.use_spur_actions;
        strm << "\nplanner_log_file: " << options.planner_log_file;
        strm << "\npolicy_log_file: " << options.policy_log_file;
        strm << "\nplanned_policy_file: " << options.planned_policy_file;
        strm << "\nexecuted_policy_file: " << options.executed_policy_file;
        return strm;
    }
}

#endif // UNCERTAINTY_PLANNING_CORE_HPP
