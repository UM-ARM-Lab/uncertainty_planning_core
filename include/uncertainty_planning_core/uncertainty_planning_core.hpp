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
#include <uncertainty_planning_core/simple_particle_contact_simulator.hpp>
#include <uncertainty_planning_core/execution_policy.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <uncertainty_planning_core/uncertainty_contact_planning.hpp>
#include <uncertainty_planning_core/simplese2_robot_helpers.hpp>
#include <uncertainty_planning_core/simplese3_robot_helpers.hpp>
#include <uncertainty_planning_core/simplelinked_robot_helpers.hpp>
#include <uncertainty_planning_core/baxter_actuator_helpers.hpp>
#include <uncertainty_planning_core/ur5_actuator_helpers.hpp>

#ifndef UNCERTAINTY_PLANNING_CORE_HPP
#define UNCERTAINTY_PLANNING_CORE_HPP

namespace uncertainty_planning_core
{
    struct OPTIONS
    {
        uncertainty_contact_planning::SPATIAL_FEATURE_CLUSTERING_TYPE clustering_type;
        uint64_t prng_seed;
        double environment_resolution;
        std::string environment_name;
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
        double max_policy_exec_time;
        // Control flags
        int32_t debug_level;
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

    inline OPTIONS GetOptions(const OPTIONS& initial_options)
    {
        OPTIONS options = initial_options;
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
        options.environment_resolution = nhp.param(std::string("environment_resolution"), options.environment_resolution);
        options.environment_name = nhp.param(std::string("environment_name"), options.environment_name);
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
        options.actuator_error = nhp.param(std::string("actuator_error"), options.actuator_error);
        options.sensor_error = nhp.param(std::string("sensor_error"), options.sensor_error);
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

    typedef execution_policy::ExecutionPolicy<Eigen::Matrix<double, 3, 1>, simplese2_robot_helpers::EigenMatrixD31Serializer, simplese2_robot_helpers::SimpleSE2Averager, simplese2_robot_helpers::SimpleSE2Distancer, simplese2_robot_helpers::SimpleSE2DimDistancer, std::allocator<Eigen::Matrix<double, 3, 1>>> SE2Policy;

    bool SaveSE2Policy(const SE2Policy& policy, const std::string& filename);

    SE2Policy LoadSE2Policy(const std::string& filename);

    std::vector<Eigen::Matrix<double, 3, 1>, std::allocator<Eigen::Matrix<double, 3, 1>>> DemonstrateSE2Simulator(const OPTIONS& options, const simplese2_robot_helpers::SimpleSE2Robot& robot, const simplese2_robot_helpers::SimpleSE2BaseSampler& sampler, const Eigen::Matrix<double, 3, 1>& start, const Eigen::Matrix<double, 3, 1>& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE2Policy, std::map<std::string, double>> PlanSE2Uncertainty(const OPTIONS& options, const simplese2_robot_helpers::SimpleSE2Robot& robot, const simplese2_robot_helpers::SimpleSE2BaseSampler& sampler, const Eigen::Matrix<double, 3, 1>& start, const Eigen::Matrix<double, 3, 1>& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> SimulateSE2UncertaintyPolicy(const OPTIONS& options, const simplese2_robot_helpers::SimpleSE2Robot& robot, const simplese2_robot_helpers::SimpleSE2BaseSampler& sampler, SE2Policy policy, const Eigen::Matrix<double, 3, 1>& start, const Eigen::Matrix<double, 3, 1>& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> ExecuteSE2UncertaintyPolicy(const OPTIONS& options, const simplese2_robot_helpers::SimpleSE2Robot& robot, const simplese2_robot_helpers::SimpleSE2BaseSampler& sampler, SE2Policy policy, const Eigen::Matrix<double, 3, 1>& start, const Eigen::Matrix<double, 3, 1>& goal, const std::function<std::vector<Eigen::Matrix<double, 3, 1>, std::allocator<Eigen::Matrix<double, 3, 1>>>(const Eigen::Matrix<double, 3, 1>&,  const Eigen::Matrix<double, 3, 1>&, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher);

    typedef execution_policy::ExecutionPolicy<Eigen::Affine3d, simplese3_robot_helpers::EigenAffine3dSerializer, simplese3_robot_helpers::SimpleSE3Averager, simplese3_robot_helpers::SimpleSE3Distancer, simplese3_robot_helpers::SimpleSE3DimDistancer, Eigen::aligned_allocator<Eigen::Affine3d>> SE3Policy;

    bool SaveSE3Policy(const SE3Policy& policy, const std::string& filename);

    SE3Policy LoadSE3Policy(const std::string& filename);

    EigenHelpers::VectorAffine3d DemonstrateSE3Simulator(const OPTIONS& options, const simplese3_robot_helpers::SimpleSE3Robot& robot, const simplese3_robot_helpers::SimpleSE3BaseSampler& sampler, const Eigen::Affine3d& start, const Eigen::Affine3d& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE3Policy, std::map<std::string, double>> PlanSE3Uncertainty(const OPTIONS& options, const simplese3_robot_helpers::SimpleSE3Robot& robot, const simplese3_robot_helpers::SimpleSE3BaseSampler& sampler, const Eigen::Affine3d& start, const Eigen::Affine3d& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> SimulateSE3UncertaintyPolicy(const OPTIONS& options, const simplese3_robot_helpers::SimpleSE3Robot& robot, const simplese3_robot_helpers::SimpleSE3BaseSampler& sampler, SE3Policy policy, const Eigen::Affine3d& start, const Eigen::Affine3d& goal, ros::Publisher& display_debug_publisher);

    std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> ExecuteSE3UncertaintyPolicy(const OPTIONS& options, const simplese3_robot_helpers::SimpleSE3Robot& robot, const simplese3_robot_helpers::SimpleSE3BaseSampler& sampler, SE3Policy policy, const Eigen::Affine3d& start, const Eigen::Affine3d& goal, const std::function<EigenHelpers::VectorAffine3d(const Eigen::Affine3d&, const Eigen::Affine3d&, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher);

    typedef execution_policy::ExecutionPolicy<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfigurationSerializer, simplelinked_robot_helpers::SimpleLinkedAverager, baxter_actuator_helpers::SimpleLinkedDistancer, baxter_actuator_helpers::SimpleLinkedDimDistancer> BaxterPolicy;

    bool SaveBaxterPolicy(const BaxterPolicy& policy, const std::string& filename);

    BaxterPolicy LoadBaxterPolicy(const std::string& filename);

    std::vector<simplelinked_robot_helpers::SimpleLinkedConfiguration, std::allocator<simplelinked_robot_helpers::SimpleLinkedConfiguration>> DemonstrateBaxterSimulator(const OPTIONS& options, const simplelinked_robot_helpers::SimpleLinkedRobot<baxter_actuator_helpers::BaxterJointActuatorModel>& robot, const simplelinked_robot_helpers::SimpleLinkedBaseSampler& sampler, const simplelinked_robot_helpers::SimpleLinkedConfiguration& start, const simplelinked_robot_helpers::SimpleLinkedConfiguration& goal, ros::Publisher& display_debug_publisher);

    std::pair<BaxterPolicy, std::map<std::string, double>> PlanBaxterUncertainty(const OPTIONS& options, const simplelinked_robot_helpers::SimpleLinkedRobot<baxter_actuator_helpers::BaxterJointActuatorModel>& robot, const simplelinked_robot_helpers::SimpleLinkedBaseSampler& sampler, const simplelinked_robot_helpers::SimpleLinkedConfiguration& start, const simplelinked_robot_helpers::SimpleLinkedConfiguration& goal, ros::Publisher& display_debug_publisher);

    std::pair<BaxterPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> SimulateBaxterUncertaintyPolicy(const OPTIONS& options, const simplelinked_robot_helpers::SimpleLinkedRobot<baxter_actuator_helpers::BaxterJointActuatorModel>& robot, const simplelinked_robot_helpers::SimpleLinkedBaseSampler& sampler, BaxterPolicy policy, const simplelinked_robot_helpers::SimpleLinkedConfiguration& start, const simplelinked_robot_helpers::SimpleLinkedConfiguration& goal, ros::Publisher& display_debug_publisher);

    std::pair<BaxterPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> ExecuteBaxterUncertaintyPolicy(const OPTIONS& options, const simplelinked_robot_helpers::SimpleLinkedRobot<baxter_actuator_helpers::BaxterJointActuatorModel>& robot, const simplelinked_robot_helpers::SimpleLinkedBaseSampler& sampler, BaxterPolicy policy, const simplelinked_robot_helpers::SimpleLinkedConfiguration& start, const simplelinked_robot_helpers::SimpleLinkedConfiguration& goal, const std::function<std::vector<simplelinked_robot_helpers::SimpleLinkedConfiguration, std::allocator<simplelinked_robot_helpers::SimpleLinkedConfiguration>>(const simplelinked_robot_helpers::SimpleLinkedConfiguration&, const simplelinked_robot_helpers::SimpleLinkedConfiguration&, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher);

    typedef execution_policy::ExecutionPolicy<simplelinked_robot_helpers::SimpleLinkedConfiguration, simplelinked_robot_helpers::SimpleLinkedConfigurationSerializer, simplelinked_robot_helpers::SimpleLinkedAverager, ur5_actuator_helpers::SimpleLinkedDistancer, ur5_actuator_helpers::SimpleLinkedDimDistancer> UR5Policy;

    bool SaveUR5Policy(const UR5Policy& policy, const std::string& filename);

    UR5Policy LoadUR5Policy(const std::string& filename);

    std::vector<simplelinked_robot_helpers::SimpleLinkedConfiguration, std::allocator<simplelinked_robot_helpers::SimpleLinkedConfiguration>> DemonstrateUR5Simulator(const OPTIONS& options, const simplelinked_robot_helpers::SimpleLinkedRobot<ur5_actuator_helpers::UR5JointActuatorModel>& robot, const simplelinked_robot_helpers::SimpleLinkedBaseSampler& sampler, const simplelinked_robot_helpers::SimpleLinkedConfiguration& start, const simplelinked_robot_helpers::SimpleLinkedConfiguration& goal, ros::Publisher& display_debug_publisher);

    std::pair<UR5Policy, std::map<std::string, double>> PlanUR5Uncertainty(const OPTIONS& options, const simplelinked_robot_helpers::SimpleLinkedRobot<ur5_actuator_helpers::UR5JointActuatorModel>& robot, const simplelinked_robot_helpers::SimpleLinkedBaseSampler& sampler, const simplelinked_robot_helpers::SimpleLinkedConfiguration& start, const simplelinked_robot_helpers::SimpleLinkedConfiguration& goal, ros::Publisher& display_debug_publisher);

    std::pair<UR5Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> SimulateUR5UncertaintyPolicy(const OPTIONS& options, const simplelinked_robot_helpers::SimpleLinkedRobot<ur5_actuator_helpers::UR5JointActuatorModel>& robot, const simplelinked_robot_helpers::SimpleLinkedBaseSampler& sampler, UR5Policy policy, const simplelinked_robot_helpers::SimpleLinkedConfiguration& start, const simplelinked_robot_helpers::SimpleLinkedConfiguration& goal, ros::Publisher& display_debug_publisher);

    std::pair<UR5Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> ExecuteUR5UncertaintyPolicy(const OPTIONS& options, const simplelinked_robot_helpers::SimpleLinkedRobot<ur5_actuator_helpers::UR5JointActuatorModel>& robot, const simplelinked_robot_helpers::SimpleLinkedBaseSampler& sampler, UR5Policy policy, const simplelinked_robot_helpers::SimpleLinkedConfiguration& start, const simplelinked_robot_helpers::SimpleLinkedConfiguration& goal, const std::function<std::vector<simplelinked_robot_helpers::SimpleLinkedConfiguration, std::allocator<simplelinked_robot_helpers::SimpleLinkedConfiguration>>(const simplelinked_robot_helpers::SimpleLinkedConfiguration&, const simplelinked_robot_helpers::SimpleLinkedConfiguration&, const double, const bool)>& robot_execution_fn, ros::Publisher& display_debug_publisher);

    std::ostream& operator<<(std::ostream& strm, const OPTIONS& options)
    {
        strm << "OPTIONS:";
        strm << "\nclustering_type: " << uncertainty_contact_planning::PrintSpatialFeatureClusteringType(options.clustering_type);
        strm << "\nprng_seed: " << options.prng_seed;
        strm << "\nenvironment_resolution: " << options.environment_resolution;
        strm << "\nenvironment_name: " << options.environment_name;
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
        strm << "\nactuator_error: " << options.actuator_error;
        strm << "\nsensor_error: " << options.sensor_error;
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
        strm << "\nenable_contact_manifold_target_adjustment: " << options.enable_contact_manifold_target_adjustment;
        strm << "\nplanner_log_file: " << options.planner_log_file;
        strm << "\npolicy_log_file: " << options.policy_log_file;
        strm << "\nplanned_policy_file: " << options.planned_policy_file;
        strm << "\nexecuted_policy_file: " << options.executed_policy_file;
        return strm;
    }
}

#endif // UNCERTAINTY_PLANNING_CORE_HPP
