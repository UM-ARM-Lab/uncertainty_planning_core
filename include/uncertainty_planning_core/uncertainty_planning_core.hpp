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
#include <arc_utilities/simple_robot_models.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <uncertainty_planning_core/execution_policy.hpp>
#include <uncertainty_planning_core/simple_simulator_interface.hpp>
#include <uncertainty_planning_core/uncertainty_planner_state.hpp>
#include <uncertainty_planning_core/uncertainty_contact_planning.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#ifndef UNCERTAINTY_PLANNING_CORE_HPP
#define UNCERTAINTY_PLANNING_CORE_HPP

#ifndef PRNG_TYPE
    #define PRNG_TYPE std::mt19937_64
#endif

namespace uncertainty_planning_core
{
    struct PLANNING_AND_EXECUTION_OPTIONS
    {
        // Time limits
        double planner_time_limit;
        // P(goal reached) termination threshold
        double p_goal_reached_termination_threshold;
        // Standard planner control params
        double goal_bias;
        double step_size;
        double goal_probability_threshold;
        double goal_distance_threshold;
        double connect_after_first_solution;
        // Distance function control params/weights
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
        options.planner_time_limit = nhp.param(std::string("planner_time_limit"), options.planner_time_limit);
        options.p_goal_reached_termination_threshold = nhp.param(std::string("p_goal_reached_termination_threshold"), options.p_goal_reached_termination_threshold);
        options.goal_bias = nhp.param(std::string("goal_bias"), options.goal_bias);
        options.step_size = nhp.param(std::string("step_size"), options.step_size);
        options.goal_probability_threshold = nhp.param(std::string("goal_probability_threshold"), options.goal_probability_threshold);
        options.goal_distance_threshold = nhp.param(std::string("goal_distance_threshold"), options.goal_distance_threshold);
        options.connect_after_first_solution = nhp.param(std::string("connect_after_first_solution"), options.connect_after_first_solution);
        options.feasibility_alpha = nhp.param(std::string("feasibility_alpha"), options.feasibility_alpha);
        options.variance_alpha = nhp.param(std::string("variance_alpha"), options.variance_alpha);
        options.num_particles = (uint32_t)nhp.param(std::string("num_particles"), (int)options.num_particles);
        options.planner_log_file = nhp.param(std::string("planner_log_file"), options.planner_log_file);
        options.planned_policy_file = nhp.param(std::string("planned_policy_file"), options.planned_policy_file);
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

    // SE(2)

    using SE2Config = simple_se2_robot_model::SimpleSE2Configuration;
    using SE2ConfigAlloc = simple_se2_robot_model::SimpleSE2ConfigAlloc;
    using SE2ConfigSerializer = simple_se2_robot_model::SimpleSE2ConfigSerializer;
    using SE2Policy = execution_policy::ExecutionPolicy<SE2Config, SE2ConfigSerializer, SE2ConfigAlloc>;
    using SE2Sampler = simple_sampler_interface::SamplerInterface<SE2Config, PRNG>;
    using SE2SamplerPtr = std::shared_ptr<SE2Sampler>;
    using SE2Simulator = simple_simulator_interface::SimulatorInterface<SE2Config, PRNG, SE2ConfigAlloc>;
    using SE2SimulatorPtr = std::shared_ptr<SE2Simulator>;
    using SE2Clustering = simple_outcome_clustering_interface::OutcomeClusteringInterface<SE2Config, SE2ConfigAlloc>;
    using SE2ClusteringPtr = std::shared_ptr<SE2Clustering>;
    using SE2PlanningSpace = uncertainty_contact_planning::UncertaintyPlanningSpace<SE2Config, SE2ConfigSerializer, SE2ConfigAlloc, PRNG>;
    using SE2Robot = simple_robot_model_interface::SimpleRobotModelInterface<SE2Config, SE2ConfigAlloc>;
    using SE2RobotPtr = std::shared_ptr<SE2Robot>;

    // SE(3)

    using SE3Config = simple_se3_robot_model::SimpleSE3Configuration;
    using SE3ConfigAlloc = simple_se3_robot_model::SimpleSE3ConfigAlloc;
    using SE3ConfigSerializer = simple_se3_robot_model::SimpleSE3ConfigSerializer;
    using SE3Policy = execution_policy::ExecutionPolicy<SE3Config, SE3ConfigSerializer, SE3ConfigAlloc>;
    using SE3Sampler = simple_sampler_interface::SamplerInterface<SE3Config, PRNG>;
    using SE3SamplerPtr = std::shared_ptr<SE3Sampler>;
    using SE3Robot = simple_robot_model_interface::SimpleRobotModelInterface<SE3Config, SE3ConfigAlloc>;
    using SE3RobotPtr = std::shared_ptr<SE3Robot>;
    using SE3Simulator = simple_simulator_interface::SimulatorInterface<SE3Config, PRNG, SE3ConfigAlloc>;
    using SE3SimulatorPtr = std::shared_ptr<SE3Simulator>;
    using SE3Clustering = simple_outcome_clustering_interface::OutcomeClusteringInterface<SE3Config, SE3ConfigAlloc>;
    using SE3ClusteringPtr = std::shared_ptr<SE3Clustering>;
    using SE3PlanningSpace = uncertainty_contact_planning::UncertaintyPlanningSpace<SE3Config, SE3ConfigSerializer, SE3ConfigAlloc, PRNG>;

    // Linked

    using LinkedConfig = simple_linked_robot_model::SimpleLinkedConfiguration;
    using LinkedConfigAlloc = simple_linked_robot_model::SimpleLinkedConfigAlloc;
    using LinkedConfigSerializer = simple_linked_robot_model::SimpleLinkedConfigSerializer;
    using LinkedPolicy = execution_policy::ExecutionPolicy<LinkedConfig, LinkedConfigSerializer, LinkedConfigAlloc>;
    using LinkedSampler = simple_sampler_interface::SamplerInterface<LinkedConfig, PRNG>;
    using LinkedSamplerPtr = std::shared_ptr<LinkedSampler>;
    using LinkedRobot = simple_robot_model_interface::SimpleRobotModelInterface<LinkedConfig, LinkedConfigAlloc>;
    using LinkedRobotPtr = std::shared_ptr<LinkedRobot>;
    using LinkedSimulator = simple_simulator_interface::SimulatorInterface<LinkedConfig, PRNG, LinkedConfigAlloc>;
    using LinkedSimulatorPtr = std::shared_ptr<LinkedSimulator>;
    using LinkedClustering = simple_outcome_clustering_interface::OutcomeClusteringInterface<LinkedConfig, LinkedConfigAlloc>;
    using LinkedClusteringPtr = std::shared_ptr<LinkedClustering>;
    using LinkedPlanningSpace = uncertainty_contact_planning::UncertaintyPlanningSpace<LinkedConfig, LinkedConfigSerializer, LinkedConfigAlloc, PRNG>;

    // VectorXd

    class VectorXdConfigSerializer
    {
    public:

        static inline std::string TypeName()
        {
            return std::string("EigenVectorXdSerializer");
        }

        static inline uint64_t Serialize(const Eigen::VectorXd& value, std::vector<uint8_t>& buffer)
        {
            return EigenHelpers::Serialize(value, buffer);
        }

        static inline std::pair<Eigen::VectorXd, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            return EigenHelpers::Deserialize<Eigen::VectorXd>(buffer, current);
        }
    };

    using VectorXdConfig = Eigen::VectorXd;
    using VectorXdConfigAlloc = std::allocator<Eigen::VectorXd>;
    using VectorXdPolicy = execution_policy::ExecutionPolicy<VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>;
    using VectorXdSampler = simple_sampler_interface::SamplerInterface<VectorXdConfig, PRNG>;
    using VectorXdSamplerPtr = std::shared_ptr<VectorXdSampler>;
    using VectorXdRobot = simple_robot_model_interface::SimpleRobotModelInterface<VectorXdConfig, VectorXdConfigAlloc>;
    using VectorXdRobotPtr = std::shared_ptr<VectorXdRobot>;
    using VectorXdSimulator = simple_simulator_interface::SimulatorInterface<VectorXdConfig, PRNG, VectorXdConfigAlloc>;
    using VectorXdSimulatorPtr = std::shared_ptr<VectorXdSimulator>;
    using VectorXdClustering = simple_outcome_clustering_interface::OutcomeClusteringInterface<VectorXdConfig, VectorXdConfigAlloc>;
    using VectorXdClusteringPtr = std::shared_ptr<VectorXdClustering>;
    using VectorXdPlanningSpace = uncertainty_contact_planning::UncertaintyPlanningSpace<VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc, PRNG>;

    // Policy and tree type definitions

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    using UncertaintyPlanningState = uncertainty_planning_tools::UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc>;

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    using UncertaintyPlanningPolicy = execution_policy::ExecutionPolicy<Configuration, ConfigSerializer, ConfigAlloc>;

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    using UncertaintyPlanningTreeState = simple_rrt_planner::SimpleRRTPlannerState<UncertaintyPlanningState<Configuration, ConfigSerializer, ConfigAlloc>, std::allocator<UncertaintyPlanningState<Configuration, ConfigSerializer, ConfigAlloc>>>;

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    using UncertaintyPlanningTree = std::vector<UncertaintyPlanningTreeState<Configuration, ConfigSerializer, ConfigAlloc>>;

    // Typedefs for user-provided goal check functions

    using SE2UserGoalStateCheckFn = std::function<double(const UncertaintyPlanningState<SE2Config, SE2ConfigSerializer, SE2ConfigAlloc>&)>;

    using SE3UserGoalStateCheckFn = std::function<double(const UncertaintyPlanningState<SE3Config, SE3ConfigSerializer, SE3ConfigAlloc>&)>;

    using LinkedUserGoalStateCheckFn = std::function<double(const UncertaintyPlanningState<LinkedConfig, LinkedConfigSerializer, LinkedConfigAlloc>&)>;

    using VectorXdUserGoalStateCheckFn = std::function<double(const UncertaintyPlanningState<VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>&)>;

    using SE2UserGoalConfigCheckFn = std::function<bool(const SE2Config&)>;

    using SE3UserGoalConfigCheckFn = std::function<bool(const SE3Config&)>;

    using LinkedUserGoalConfigCheckFn = std::function<bool(const LinkedConfig&)>;

    using VectorXdUserGoalConfigCheckFn = std::function<bool(const VectorXdConfig&)>;

    // Implementations of basic user goal config check -> user goal state check functions

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    inline double UserGoalCheckWrapperFn(const UncertaintyPlanningState<Configuration, ConfigSerializer, ConfigAlloc>& state,
                                         const std::function<bool(const Configuration&)>& user_goal_config_check_fn)
    {
        if (state.HasParticles())
        {
            const std::vector<Configuration, ConfigAlloc>& particle_positions = state.GetParticlePositionsImmutable().first;
            const size_t num_particles = state.GetNumParticles();
            if (num_particles > 0)
            {
                size_t reached_goal = 0;
                for (size_t idx = 0; idx < num_particles; idx++)
                {
                    const bool particle_reached_goal = user_goal_config_check_fn(particle_positions[idx]);
                    if (particle_reached_goal)
                    {
                        reached_goal++;
                    }
                }
                const double p_goal_reached = (double)reached_goal / (double)num_particles;
                return p_goal_reached;
            }
            else
            {
                return 0.0;
            }
        }
        else
        {
            if (user_goal_config_check_fn(state.GetExpectation()))
            {
                return 1.0;
            }
            else
            {
                return 0.0;
            }
        }
    }

    inline double SE2UserGoalCheckWrapperFn(const UncertaintyPlanningState<SE2Config, SE2ConfigSerializer, SE2ConfigAlloc>& state,
                                            const SE2UserGoalConfigCheckFn& user_goal_config_check_fn);

    inline double SE3UserGoalCheckWrapperFn(const UncertaintyPlanningState<SE3Config, SE3ConfigSerializer, SE3ConfigAlloc>& state,
                                            const SE3UserGoalConfigCheckFn& user_goal_config_check_fn);

    inline double LinkedUserGoalCheckWrapperFn(const UncertaintyPlanningState<LinkedConfig, LinkedConfigSerializer, LinkedConfigAlloc>& state,
                                               const LinkedUserGoalConfigCheckFn& user_goal_config_check_fn);

    inline double VectorXdUserGoalCheckWrapperFn(const UncertaintyPlanningState<VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>& state,
                                                 const VectorXdUserGoalConfigCheckFn& user_goal_config_check_fn);

    // Policy saving and loading

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    inline uint64_t SerializePlannerTree(const UncertaintyPlanningTree<Configuration, ConfigSerializer, ConfigAlloc>& planner_tree, std::vector<uint8_t>& buffer)
    {
        std::cout << "Serializing planner tree..." << std::endl;
        std::function<uint64_t(const UncertaintyPlanningTreeState<Configuration, ConfigSerializer, ConfigAlloc>&, std::vector<uint8_t>&)> planning_tree_state_serializer_fn
                = [] (const UncertaintyPlanningTreeState<Configuration, ConfigSerializer, ConfigAlloc>& state, std::vector<uint8_t>& ser_buffer)
        { return UncertaintyPlanningTreeState<Configuration, ConfigSerializer, ConfigAlloc>::Serialize(state,
                                                                                                       ser_buffer,
                                                                                                       UncertaintyPlanningState<Configuration, ConfigSerializer, ConfigAlloc>::Serialize); };
        arc_helpers::SerializeVector(planner_tree, buffer, planning_tree_state_serializer_fn);
        const uint64_t size = arc_helpers::SerializeVector(planner_tree, buffer, planning_tree_state_serializer_fn);
        std::cout << "...planner tree of " << planner_tree.size() << " states serialized into " << buffer.size() << " bytes" << std::endl;
        return size;
    }

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    inline std::pair<UncertaintyPlanningTree<Configuration, ConfigSerializer, ConfigAlloc>, uint64_t> DeserializePlannerTree(const std::vector<uint8_t>& buffer, const uint64_t current)
    {
        std::cout << "Deserializing planner tree..." << std::endl;
        std::function<std::pair<UncertaintyPlanningTreeState<Configuration, ConfigSerializer, ConfigAlloc>, uint64_t>(const std::vector<uint8_t>&, const uint64_t)> planning_tree_state_deserializer_fn
                = [] (const std::vector<uint8_t>& deser_buffer, const uint64_t deser_current)
        { return UncertaintyPlanningTreeState<Configuration, ConfigSerializer, ConfigAlloc>::Deserialize(deser_buffer,
                                                                                                         deser_current,
                                                                                                         UncertaintyPlanningState<Configuration, ConfigSerializer, ConfigAlloc>::Deserialize); };
        const std::pair<UncertaintyPlanningTree<Configuration, ConfigSerializer, ConfigAlloc>, uint64_t> deserialized_tree
                = arc_helpers::DeserializeVector<UncertaintyPlanningTreeState<Configuration, ConfigSerializer, ConfigAlloc>>(buffer, current, planning_tree_state_deserializer_fn);
        std::cout << "...planner tree of " << deserialized_tree.first.size() << " states deserialized from " << deserialized_tree.second << " bytes" << std::endl;
        return deserialized_tree;
    }

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    inline bool SavePlannerTree(const UncertaintyPlanningTree<Configuration, ConfigSerializer, ConfigAlloc>& planner_tree, const std::string& filepath)
    {
        try
        {
            std::cout << "Attempting to serialize tree..." << std::endl;
            std::vector<uint8_t> buffer;
            SerializePlannerTree<Configuration, ConfigSerializer, ConfigAlloc>(planner_tree, buffer);
            // Write a header to detect if compression is enabled (someday)
            std::cout << "Compressing for storage..." << std::endl;
            const std::vector<uint8_t> compressed_serialized_tree = ZlibHelpers::CompressBytes(buffer);
            std::cout << "Attempting to save to file..." << std::endl;
            std::ofstream output_file(filepath, std::ios::out|std::ios::binary);
            const size_t serialized_size = compressed_serialized_tree.size();
            output_file.write(reinterpret_cast<const char*>(compressed_serialized_tree.data()), (std::streamsize)serialized_size);
            output_file.close();
            return true;
        }
        catch (...)
        {
            std::cerr << "Saving planner tree failed" << std::endl;
            return false;
        }
    }

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    inline UncertaintyPlanningTree<Configuration, ConfigSerializer, ConfigAlloc> LoadPlannerTree(const std::string& filepath)
    {
        std::cout << "Attempting to load from file..." << std::endl;
        std::ifstream input_file(filepath, std::ios::in|std::ios::binary);
        if (input_file.good() == false)
        {
            throw std::invalid_argument("Planner tree file does not exist");
        }
        input_file.seekg(0, std::ios::end);
        std::streampos end = input_file.tellg();
        input_file.seekg(0, std::ios::beg);
        std::streampos begin = input_file.tellg();
        const std::streamsize serialized_size = end - begin;
        std::vector<uint8_t> file_buffer((size_t)serialized_size, 0x00);
        input_file.read(reinterpret_cast<char*>(file_buffer.data()), serialized_size);
        // Write a header to detect if compression is enabled (someday)
        std::cout << "Decompressing from storage..." << std::endl;
        const std::vector<uint8_t> decompressed_serialized_tree = ZlibHelpers::DecompressBytes(file_buffer);
        std::cout << "Attempting to deserialize tree..." << std::endl;
        return DeserializePlannerTree<Configuration, ConfigSerializer, ConfigAlloc>(decompressed_serialized_tree, 0u).first;
    }

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    inline bool SavePolicy(const UncertaintyPlanningPolicy<Configuration, ConfigSerializer, ConfigAlloc>& policy, const std::string& filepath)
    {
        try
        {
            std::cout << "Attempting to serialize policy..." << std::endl;
            std::vector<uint8_t> buffer;
            UncertaintyPlanningPolicy<Configuration, ConfigSerializer, ConfigAlloc>::Serialize(policy, buffer);
            // Write a header to detect if compression is enabled (someday)
            //std::cout << "Compressing for storage..." << std::endl;
            //const std::vector<uint8_t> compressed_serialized_policy = ZlibHelpers::CompressBytes(buffer);
            std::cout << "Compression disabled (no Zlib available)..." << std::endl;
            const std::vector<uint8_t> compressed_serialized_policy = buffer;
            std::cout << "Attempting to save to file..." << std::endl;
            std::ofstream output_file(filepath, std::ios::out|std::ios::binary);
            const size_t serialized_size = compressed_serialized_policy.size();
            output_file.write(reinterpret_cast<const char*>(compressed_serialized_policy.data()), (std::streamsize)serialized_size);
            output_file.close();
            return true;
        }
        catch (...)
        {
            std::cerr << "Saving policy failed" << std::endl;
            return false;
        }
    }

    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc>
    inline UncertaintyPlanningPolicy<Configuration, ConfigSerializer, ConfigAlloc> LoadPolicy(const std::string& filepath)
    {
        std::cout << "Attempting to load from file..." << std::endl;
        std::ifstream input_file(filepath, std::ios::in|std::ios::binary);
        if (input_file.good() == false)
        {
            throw std::invalid_argument("Policy file does not exist");
        }
        input_file.seekg(0, std::ios::end);
        std::streampos end = input_file.tellg();
        input_file.seekg(0, std::ios::beg);
        std::streampos begin = input_file.tellg();
        const std::streamsize serialized_size = end - begin;
        std::vector<uint8_t> file_buffer((size_t)serialized_size, 0x00);
        input_file.read(reinterpret_cast<char*>(file_buffer.data()), serialized_size);
        // Write a header to detect if compression is enabled (someday)
        //std::cout << "Decompressing from storage..." << std::endl;
        //const std::vector<uint8_t> decompressed_serialized_policy = ZlibHelpers::DecompressBytes(file_buffer);
        std::cout << "Decompression disabled (no Zlib available)..." << std::endl;
        const std::vector<uint8_t> decompressed_serialized_policy = file_buffer;
        std::cout << "Attempting to deserialize policy..." << std::endl;
        return UncertaintyPlanningPolicy<Configuration, ConfigSerializer, ConfigAlloc>::Deserialize(decompressed_serialized_policy, 0u).first;
    }

    // Policy saving and loading concrete implementations

    bool SaveSE2Policy(const SE2Policy& policy, const std::string& filename);

    SE2Policy LoadSE2Policy(const std::string& filename);

    bool SaveSE3Policy(const SE3Policy& policy, const std::string& filename);

    SE3Policy LoadSE3Policy(const std::string& filename);

    bool SaveLinkedPolicy(const LinkedPolicy& policy, const std::string& filename);

    LinkedPolicy LoadLinkedPolicy(const std::string& filename);

    bool SaveVectorXdPolicy(const VectorXdPolicy& policy, const std::string& filename);

    VectorXdPolicy LoadVectorXdPolicy(const std::string& filename);

    // SE2 Interface

    std::vector<SE2Config, SE2ConfigAlloc>
    DemonstrateSE2Simulator(const PLANNING_AND_EXECUTION_OPTIONS& options,
                            const SE2RobotPtr& robot,
                            const SE2SimulatorPtr& simulator,
                            const SE2SamplerPtr& sampler,
                            const SE2ClusteringPtr& clustering,
                            const SE2Config& start,
                            const SE2Config& goal,
                            const std::function<void(const std::string&, const int32_t)>& logging_fn,
                            const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE2Policy, std::map<std::string, double>>
    PlanSE2Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                       const SE2RobotPtr& robot,
                       const SE2SimulatorPtr& simulator,
                       const SE2SamplerPtr& sampler,
                       const SE2ClusteringPtr& clustering,
                       const SE2Config& start,
                       const SE2Config& goal,
                       const double policy_marker_size,
                       const std::function<void(const std::string&, const int32_t)>& logging_fn,
                       const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE2Policy, std::map<std::string, double>>
    PlanSE2Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                       const SE2RobotPtr& robot,
                       const SE2SimulatorPtr& simulator,
                       const SE2SamplerPtr& sampler,
                       const SE2ClusteringPtr& clustering,
                       const SE2Config& start,
                       const SE2UserGoalStateCheckFn& user_goal_check_fn,
                       const double policy_marker_size,
                       const std::function<void(const std::string&, const int32_t)>& logging_fn,
                       const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    SimulateSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                 const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    ExecuteSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    SimulateSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                 const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE2Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    ExecuteSE2UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    // SE3 Interface

    std::vector<SE3Config, SE3ConfigAlloc>
    DemonstrateSE3Simulator(const PLANNING_AND_EXECUTION_OPTIONS& options,
                            const SE3RobotPtr& robot,
                            const SE3SimulatorPtr& simulator,
                            const SE3SamplerPtr& sampler,
                            const SE3ClusteringPtr& clustering,
                            const SE3Config& start,
                            const SE3Config& goal,
                            const std::function<void(const std::string&, const int32_t)>& logging_fn,
                            const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE3Policy, std::map<std::string, double>>
    PlanSE3Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                       const SE3RobotPtr& robot,
                       const SE3SimulatorPtr& simulator,
                       const SE3SamplerPtr& sampler,
                       const SE3ClusteringPtr& clustering,
                       const SE3Config& start,
                       const SE3Config& goal,
                       const double policy_marker_size,
                       const std::function<void(const std::string&, const int32_t)>& logging_fn,
                       const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE3Policy, std::map<std::string, double>>
    PlanSE3Uncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                       const SE3RobotPtr& robot,
                       const SE3SimulatorPtr& simulator,
                       const SE3SamplerPtr& sampler,
                       const SE3ClusteringPtr& clustering,
                       const SE3Config& start,
                       const SE3UserGoalStateCheckFn& user_goal_check_fn,
                       const double policy_marker_size,
                       const std::function<void(const std::string&, const int32_t)>& logging_fn,
                       const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    SimulateSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                 const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    ExecuteSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    SimulateSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                 const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<SE3Policy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    ExecuteSE3UncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    // Linked Interface

    std::vector<LinkedConfig, LinkedConfigAlloc>
    DemonstrateLinkedSimulator(const PLANNING_AND_EXECUTION_OPTIONS& options,
                               const LinkedRobotPtr& robot,
                               const LinkedSimulatorPtr& simulator,
                               const LinkedSamplerPtr& sampler,
                               const LinkedClusteringPtr& clustering,
                               const LinkedConfig& start,
                               const LinkedConfig& goal,
                               const std::function<void(const std::string&, const int32_t)>& logging_fn,
                               const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<LinkedPolicy, std::map<std::string, double>>
    PlanLinkedUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                          const LinkedRobotPtr& robot,
                          const LinkedSimulatorPtr& simulator,
                          const LinkedSamplerPtr& sampler,
                          const LinkedClusteringPtr& clustering,
                          const LinkedConfig& start,
                          const LinkedConfig& goal,
                          const double policy_marker_size,
                          const std::function<void(const std::string&, const int32_t)>& logging_fn,
                          const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<LinkedPolicy, std::map<std::string, double>>
    PlanLinkedUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                          const LinkedRobotPtr& robot,
                          const LinkedSimulatorPtr& simulator,
                          const LinkedSamplerPtr& sampler,
                          const LinkedClusteringPtr& clustering,
                          const LinkedConfig& start,
                          const LinkedUserGoalStateCheckFn& user_goal_check_fn,
                          const double policy_marker_size,
                          const std::function<void(const std::string&, const int32_t)>& logging_fn,
                          const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    SimulateLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                    const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    ExecuteLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                   const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    SimulateLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                    const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<LinkedPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    ExecuteLinkedUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                   const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    // VectorXd Interface

    std::vector<VectorXdConfig, VectorXdConfigAlloc>
    DemonstrateVectorXdSimulator(const PLANNING_AND_EXECUTION_OPTIONS& options,
                                 const VectorXdRobotPtr& robot,
                                 const VectorXdSimulatorPtr& simulator,
                                 const VectorXdSamplerPtr& sampler,
                                 const VectorXdClusteringPtr& clustering,
                                 const VectorXdConfig& start,
                                 const VectorXdConfig& goal,
                                 const std::function<void(const std::string&, const int32_t)>& logging_fn,
                                 const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<VectorXdPolicy, std::map<std::string, double>>
    PlanVectorXdUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                            const VectorXdRobotPtr& robot,
                            const VectorXdSimulatorPtr& simulator,
                            const VectorXdSamplerPtr& sampler,
                            const VectorXdClusteringPtr& clustering,
                            const VectorXdConfig& start,
                            const VectorXdConfig& goal,
                            const double policy_marker_size,
                            const std::function<void(const std::string&, const int32_t)>& logging_fn,
                            const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<VectorXdPolicy, std::map<std::string, double>>
    PlanVectorXdUncertainty(const PLANNING_AND_EXECUTION_OPTIONS& options,
                            const VectorXdRobotPtr& robot,
                            const VectorXdSimulatorPtr& simulator,
                            const VectorXdSamplerPtr& sampler,
                            const VectorXdClusteringPtr& clustering,
                            const VectorXdConfig& start,
                            const VectorXdUserGoalStateCheckFn& user_goal_check_fn,
                            const double policy_marker_size,
                            const std::function<void(const std::string&, const int32_t)>& logging_fn,
                            const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<VectorXdPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    SimulateVectorXdUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                      const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<VectorXdPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    ExecuteVectorXdUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                     const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<VectorXdPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    SimulateVectorXdUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                      const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    std::pair<VectorXdPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>>
    ExecuteVectorXdUncertaintyPolicy(const PLANNING_AND_EXECUTION_OPTIONS& options,
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
                                     const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn);

    inline std::ostream& operator<<(std::ostream& strm, const PLANNING_AND_EXECUTION_OPTIONS& options)
    {
        strm << "OPTIONS:";
        strm << "\nplanner_time_limit: " << options.planner_time_limit;
        strm << "\np_goal_reached_termination_threshold: " << options.p_goal_reached_termination_threshold;
        strm << "\ngoal_bias: " << options.goal_bias;
        strm << "\nstep_size: " << options.step_size;
        strm << "\ngoal_probability_threshold: " << options.goal_probability_threshold;
        strm << "\ngoal_distance_threshold: " << options.goal_distance_threshold;
        strm << "\nconnect_after_first_solution: " << options.connect_after_first_solution;
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
