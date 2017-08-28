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
#include <uncertainty_planning_core/simple_samplers.hpp>
#include <uncertainty_planning_core/uncertainty_planner_state.hpp>
#include <uncertainty_planning_core/simple_simulator_interface.hpp>
#include <uncertainty_planning_core/execution_policy.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <omp.h>

#ifndef UNCERTAINTY_CONTACT_PLANNING_HPP
#define UNCERTAINTY_CONTACT_PLANNING_HPP

inline uint32_t get_num_omp_threads(void)
{
#if defined(_OPENMP)
    uint32_t num_threads = 0;
    #pragma omp parallel
    {
        num_threads = (uint32_t)omp_get_num_threads();
    }
    return num_threads;
#else
    return 1u;
#endif
}

namespace uncertainty_contact_planning
{
    enum SPATIAL_FEATURE_CLUSTERING_TYPE {CONVEX_REGION_SIGNATURE, ACTUATION_CENTER_CONNECTIVITY, POINT_TO_POINT_MOVEMENT, COMPARE};

    inline SPATIAL_FEATURE_CLUSTERING_TYPE ParseSpatialFeatureClusteringType(const std::string& typestr)
    {
        if (typestr == "CRS" || typestr == "crs")
        {
            return CONVEX_REGION_SIGNATURE;
        }
        else if (typestr == "AC" || typestr == "ac")
        {
            return ACTUATION_CENTER_CONNECTIVITY;
        }
        else if (typestr == "PTPM" || typestr == "ptpm")
        {
            return POINT_TO_POINT_MOVEMENT;
        }
        else if (typestr == "COMPARE" || typestr == "compare")
        {
            return COMPARE;
        }
        else
        {
            assert(false);
        }
    }

    inline std::string PrintSpatialFeatureClusteringType(const SPATIAL_FEATURE_CLUSTERING_TYPE& clustering_type)
    {
        if (clustering_type == CONVEX_REGION_SIGNATURE)
        {
            return std::string("CRS");
        }
        else if (clustering_type == ACTUATION_CENTER_CONNECTIVITY)
        {
            return std::string("AC");
        }
        else if (clustering_type == POINT_TO_POINT_MOVEMENT)
        {
            return std::string("PTPM");
        }
        else if (clustering_type == COMPARE)
        {
            return std::string("COMPARE");
        }
        else
        {
            assert(false);
        }
    }

    template<typename Robot, typename Configuration, typename ConfigSerializer, typename ConfigAlloc=std::allocator<Configuration>, typename PRNG=std::mt19937_64>
    class UncertaintyPlanningSpace
    {
    protected:

        struct SpatialClusteringPerformance
        {
            std::vector<uint32_t> clustering_splits;
            std::vector<double> crs_similarities;
            std::vector<double> ac_similarities;

            SpatialClusteringPerformance()
            {
                clustering_splits.clear();;
                crs_similarities.clear();
                ac_similarities.clear();
            }

            void ExportResults() const
            {
                const std::string log_file_name("/tmp/spatial_clustering_performance.csv");
                std::ofstream log_file(log_file_name, std::ios_base::out);
                if (!log_file.is_open())
                {
                    std::cerr << "\x1b[31;1m Unable to create folder/file to log to: " << log_file_name << "\x1b[0m \n";
                    throw std::invalid_argument("Log filename must be write-openable");
                }
                log_file << PrettyPrint::PrettyPrint(clustering_splits, false, ",") << std::endl;
                log_file << PrettyPrint::PrettyPrint(crs_similarities, false, ",") << std::endl;
                log_file << PrettyPrint::PrettyPrint(ac_similarities, false, ",") << std::endl;
                log_file.close();
            }
        };

        // Typedef so we don't hate ourselves
        typedef uncertainty_planning_tools::UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc> UncertaintyPlanningState;
        typedef execution_policy::ExecutionPolicy<Configuration, ConfigSerializer, ConfigAlloc> UncertaintyPlanningPolicy;
        typedef simple_rrt_planner::SimpleRRTPlannerState<UncertaintyPlanningState, std::allocator<UncertaintyPlanningState>> UncertaintyPlanningTreeState;
        typedef std::vector<UncertaintyPlanningTreeState> UncertaintyPlanningTree;
        typedef arc_dijkstras::Graph<UncertaintyPlanningState, std::allocator<UncertaintyPlanningState>> ExecutionPolicyGraph;

        bool simulate_with_individual_jacobians_;
        SPATIAL_FEATURE_CLUSTERING_TYPE spatial_feature_clustering_type_;
        size_t num_particles_;
        double step_size_;
        double step_duration_;
        double goal_distance_threshold_;
        double goal_probability_threshold_;
        double signature_matching_threshold_;
        double distance_clustering_threshold_;
        double feasibility_alpha_;
        double variance_alpha_;
        double connect_after_first_solution_;
        int32_t debug_level_;
        Robot robot_;
        std::shared_ptr<simple_samplers::SimpleBaseSampler<Configuration, PRNG>> sampler_ptr_;
        std::shared_ptr<simple_simulator_interface::SimulatorInterface<Robot, Configuration, PRNG, ConfigAlloc>> simulator_ptr_;
        mutable PRNG rng_;
        mutable std::vector<PRNG> rngs_;
        mutable uint64_t state_counter_;
        mutable uint64_t transition_id_;
        mutable uint64_t split_id_;
        mutable uint64_t cluster_calls_;
        mutable uint64_t cluster_fallback_calls_;
        mutable uint64_t particles_stored_;
        mutable uint64_t particles_simulated_;
        mutable uint64_t goal_candidates_evaluated_;
        mutable uint64_t goal_reaching_performed_;
        mutable uint64_t goal_reaching_successful_;
        mutable double total_goal_reached_probability_;
        mutable double time_to_first_solution_;
        mutable double elapsed_clustering_time_;
        mutable double elapsed_simulation_time_;
        mutable UncertaintyPlanningTree nearest_neighbors_storage_;
        mutable SpatialClusteringPerformance clustering_performance_;

        /*
         * Private helper function - needs well-formed inputs, so it isn't safe to expose to external code
         */
        inline static void ExtractChildStates(const UncertaintyPlanningTree& raw_planner_tree, const int64_t raw_parent_index, const int64_t pruned_parent_index, UncertaintyPlanningTree& pruned_planner_tree)
        {
            assert((raw_parent_index >= 0) && (raw_parent_index < (int64_t)raw_planner_tree.size()));
            assert((pruned_parent_index >= 0) && (pruned_parent_index < (int64_t)pruned_planner_tree.size()));
            assert(raw_planner_tree[(size_t)raw_parent_index].IsInitialized());
            assert(pruned_planner_tree[(size_t)pruned_parent_index].IsInitialized());
            // Clear the child indices, so we can update them with new values later
            pruned_planner_tree[(size_t)pruned_parent_index].ClearChildIndicies();
            const std::vector<int64_t>& current_child_indices = raw_planner_tree[(size_t)raw_parent_index].GetChildIndices();
            for (size_t idx = 0; idx < current_child_indices.size(); idx++)
            {
                const int64_t raw_child_index = current_child_indices[idx];
                assert((raw_child_index > 0) && (raw_child_index < (int64_t)raw_planner_tree.size()));
                const UncertaintyPlanningTreeState& current_child_state = raw_planner_tree[(size_t)raw_child_index];
                if (current_child_state.GetParentIndex() >= 0)
                {
                    // Get the new child index
                    const int64_t pruned_child_index = (int64_t)pruned_planner_tree.size();
                    // Add to the pruned tree
                    pruned_planner_tree.push_back(current_child_state);
                    // Update parent indices
                    pruned_planner_tree[(size_t)pruned_child_index].SetParentIndex(pruned_parent_index);
                    // Update the parent
                    pruned_planner_tree[(size_t)pruned_parent_index].AddChildIndex(pruned_child_index);
                    // Recursive call
                    ExtractChildStates(raw_planner_tree, raw_child_index, pruned_child_index, pruned_planner_tree);
                }
            }
        }

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /*
         * Serialization/deserialization helpers
         */
        static inline uint64_t SerializePlannerTree(const UncertaintyPlanningTree& planner_tree, std::vector<uint8_t>& buffer)
        {
            std::cout << "Serializing planner tree..." << std::endl;
            std::function<uint64_t(const UncertaintyPlanningTreeState&, std::vector<uint8_t>&)> planning_tree_state_serializer_fn = [] (const UncertaintyPlanningTreeState& state, std::vector<uint8_t>& buffer) { return UncertaintyPlanningTreeState::Serialize(state, buffer, UncertaintyPlanningState::Serialize); };
            arc_helpers::SerializeVector(planner_tree, buffer, planning_tree_state_serializer_fn);
            const uint64_t size = arc_helpers::SerializeVector(planner_tree, buffer, planning_tree_state_serializer_fn);
            std::cout << "...planner tree of " << planner_tree.size() << " states serialized into " << buffer.size() << " bytes" << std::endl;
            return size;
        }

        static inline std::pair<UncertaintyPlanningTree, uint64_t> DeserializePlannerTree(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            std::cout << "Deserializing planner tree..." << std::endl;
            std::function<std::pair<UncertaintyPlanningTreeState, uint64_t>(const std::vector<uint8_t>&, const uint64_t)> planning_tree_state_deserializer_fn = [] (const std::vector<uint8_t>& buffer, const uint64_t current) { return UncertaintyPlanningTreeState::Deserialize(buffer, current, UncertaintyPlanningState::Deserialize); };
            const std::pair<UncertaintyPlanningTree, uint64_t> deserialized_tree = arc_helpers::DeserializeVector<UncertaintyPlanningTreeState>(buffer, current, planning_tree_state_deserializer_fn);
            std::cout << "...planner tree of " << deserialized_tree.first.size() << " states deserialized from " << deserialized_tree.second << " bytes" << std::endl;
            return deserialized_tree;
        }

        static inline bool SavePlannerTree(const UncertaintyPlanningTree& planner_tree, const std::string& filepath)
        {
            try
            {
                std::cout << "Attempting to serialize tree..." << std::endl;
                std::vector<uint8_t> buffer;
                SerializePlannerTree(planner_tree, buffer);
                // Write a header to detect if compression is enabled (someday)
                //std::cout << "Compressing for storage..." << std::endl;
                //const std::vector<uint8_t> compressed_serialized_tree = ZlibHelpers::CompressBytes(buffer);
                std::cout << " Compression disabled (no Zlib available)..." << std::endl;
                const std::vector<uint8_t> compressed_serialized_tree = buffer;
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

        static inline UncertaintyPlanningTree LoadPlannerTree(const std::string& filepath)
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
            //std::cout << "Decompressing from storage..." << std::endl;
            //const std::vector<uint8_t> decompressed_serialized_tree = ZlibHelpers::DecompressBytes(file_buffer);
            std::cout << "Decompression disabled (no Zlib available)..." << std::endl;
            const std::vector<uint8_t> decompressed_serialized_tree = file_buffer;
            std::cout << "Attempting to deserialize tree..." << std::endl;
            return DeserializePlannerTree(decompressed_serialized_tree, 0u).first;
        }

        static inline bool SavePolicy(const UncertaintyPlanningPolicy& policy, const std::string& filepath)
        {
            try
            {
                std::cout << "Attempting to serialize policy..." << std::endl;
                std::vector<uint8_t> buffer;
                UncertaintyPlanningPolicy::Serialize(policy, buffer);
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

        static inline UncertaintyPlanningPolicy LoadPolicy(const std::string& filepath)
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
            return UncertaintyPlanningPolicy::Deserialize(decompressed_serialized_policy, 0u).first;
        }

        /*
         * Constructor
         */
        inline UncertaintyPlanningSpace(const SPATIAL_FEATURE_CLUSTERING_TYPE& spatial_feature_clustering_type, const bool simulate_with_individual_jacobians, const int32_t debug_level, const size_t num_particles, const double step_size, const double simulation_step_duration, const double goal_distance_threshold, const double goal_probability_threshold, const double signature_matching_threshold, const double distance_clustering_threshold, const double feasibility_alpha, const double variance_alpha, const double connect_after_first_solution, const Robot& robot, const std::shared_ptr<simple_samplers::SimpleBaseSampler<Configuration, PRNG>>& sampler_ptr, const std::shared_ptr<simple_simulator_interface::SimulatorInterface<Robot, Configuration, PRNG, ConfigAlloc>>& simulator_ptr, const uint64_t prng_seed) : robot_(robot), sampler_ptr_(sampler_ptr), simulator_ptr_(simulator_ptr)
        {
            debug_level_ = debug_level;
            // Prepare the default RNG
            PRNG prng(prng_seed);
            rng_ = prng;
            // Temp seed distribution
            std::uniform_int_distribution<uint64_t> seed_dist(0, std::numeric_limits<uint64_t>::max());
            // Get the number of threads we're using
            const uint32_t num_threads = get_num_omp_threads();
            assert(num_threads >= 1);
            // Prepare a number of PRNGs for each thread
            rngs_.clear();
            for (uint32_t tidx = 0; tidx < num_threads; tidx++)
            {
                rngs_.push_back(PRNG(seed_dist(rng_)));
            }
            spatial_feature_clustering_type_ = spatial_feature_clustering_type;
            simulate_with_individual_jacobians_ = simulate_with_individual_jacobians;
            num_particles_ = num_particles;
            step_size_ = step_size;
            step_duration_ = simulation_step_duration;
            goal_distance_threshold_ = goal_distance_threshold;
            goal_probability_threshold_ = goal_probability_threshold;
            signature_matching_threshold_ = signature_matching_threshold;
            distance_clustering_threshold_ = distance_clustering_threshold;
            feasibility_alpha_ = feasibility_alpha;
            variance_alpha_ = variance_alpha;
            connect_after_first_solution_ = connect_after_first_solution;
            Reset();
        }

        inline void Reset()
        {
            state_counter_ = 0;
            transition_id_ = 0;
            split_id_ = 0;
            cluster_calls_ = 0;
            cluster_fallback_calls_ = 0;
            elapsed_clustering_time_ = 0.0;
            elapsed_simulation_time_ = 0.0;
            particles_stored_ = 0;
            particles_simulated_ = 0;
            goal_candidates_evaluated_ = 0;
            goal_reaching_performed_ = 0;
            goal_reaching_successful_ = 0;
            nearest_neighbors_storage_.clear();
        }

        inline void Reset(const uint64_t prng_seed)
        {
            // Prepare the default RNG
            PRNG prng(prng_seed);
            rng_ = prng;
            // Temp seed distribution
            std::uniform_int_distribution<uint64_t> seed_dist(0, std::numeric_limits<uint64_t>::max());
            // Get the number of threads we're using
            const uint32_t num_threads = get_num_omp_threads();
            assert(num_threads >= 1);
            // Prepare a number of PRNGs for each thread
            rngs_.clear();
            for (uint32_t tidx = 0; tidx < num_threads; tidx++)
            {
                rngs_.push_back(PRNG(seed_dist(rng_)));
            }
            Reset();
        }

        /*
         * Test example to show the behavior of the lightweight simulator
         */
        inline simple_simulator_interface::ForwardSimulationStepTrace<Configuration, ConfigAlloc> DemonstrateSimulator(const Configuration& start, const Configuration& goal, ros::Publisher& display_pub) const
        {
            // Draw the simulation environment
            display_pub.publish(simulator_ptr_->ExportAllForDisplay());
            // Draw the start and goal
            std_msgs::ColorRGBA start_color;
            start_color.r = 1.0;
            start_color.g = 0.5;
            start_color.b = 0.0;
            start_color.a = 1.0;
            std_msgs::ColorRGBA goal_color;
            goal_color.r = 1.0;
            goal_color.g = 0.0;
            goal_color.b = 1.0;
            goal_color.a = 1.0;
            visualization_msgs::Marker start_marker = DrawRobotConfiguration(robot_, start, start_color);
            start_marker.ns = "start_state";
            start_marker.id = 1;
            visualization_msgs::Marker goal_marker = DrawRobotConfiguration(robot_, goal, goal_color);
            goal_marker.ns = "goal_state";
            goal_marker.id = 1;
            visualization_msgs::MarkerArray simulator_step_display_rep;
            simulator_step_display_rep.markers.push_back(start_marker);
            simulator_step_display_rep.markers.push_back(goal_marker);
            display_pub.publish(simulator_step_display_rep);
            // Wait for input
            std::cout << "Press ENTER to solve..." << std::endl;
            std::cin.get();
            simple_simulator_interface::ForwardSimulationStepTrace<Configuration, ConfigAlloc> trace;
            simulator_ptr_->ForwardSimulateRobot(robot_, start, goal, rng_, step_duration_, (goal_distance_threshold_ * 0.5), simulate_with_individual_jacobians_, true, trace, true, display_pub);
            // Wait for input
            std::cout << "Press ENTER to draw..." << std::endl;
            std::cin.get();
            if (debug_level_ >= 20)
            {
                // Draw the action
                std_msgs::ColorRGBA free_color;
                free_color.r = 0.0;
                free_color.g = 1.0;
                free_color.b = 0.0;
                free_color.a = 1.0;
                std_msgs::ColorRGBA colliding_color;
                colliding_color.r = 1.0;
                colliding_color.g = 0.0;
                colliding_color.b = 0.0;
                colliding_color.a = 1.0;
                std_msgs::ColorRGBA control_input_color;
                control_input_color.r = 1.0;
                control_input_color.g = 1.0;
                control_input_color.b = 0.0;
                control_input_color.a = 1.0;
                std_msgs::ColorRGBA control_step_color;
                control_step_color.r = 0.0;
                control_step_color.g = 1.0;
                control_step_color.b = 1.0;
                control_step_color.a = 1.0;
                // Keep track of previous position
                Configuration previous_config = start;
                for (size_t step_idx = 0; step_idx < trace.resolver_steps.size(); step_idx++)
                {
                    const simple_simulator_interface::ForwardSimulationResolverTrace<Configuration, ConfigAlloc>& step_trace = trace.resolver_steps[step_idx];
                    const Eigen::VectorXd& control_input_step = step_trace.control_input_step;
                    // Draw the control input for the entire trace segment
                    const Eigen::VectorXd& control_input = step_trace.control_input;
                    visualization_msgs::Marker control_step_marker = DrawRobotControlInput(robot_, previous_config, control_input, control_input_color);
                    control_step_marker.ns = "control_input_state";
                    control_step_marker.id = 1;
                    visualization_msgs::MarkerArray control_display_rep;
                    control_display_rep.markers.push_back(control_step_marker);
                    display_pub.publish(control_display_rep);
                    for (size_t resolver_step_idx = 0; resolver_step_idx < step_trace.contact_resolver_steps.size(); resolver_step_idx++)
                    {
                        // Get the current trace segment
                        const simple_simulator_interface::ForwardSimulationContactResolverStepTrace<Configuration, ConfigAlloc>& contact_resolution_trace = step_trace.contact_resolver_steps[resolver_step_idx];
                        for (size_t contact_resolution_step_idx = 0; contact_resolution_step_idx < contact_resolution_trace.contact_resolution_steps.size(); contact_resolution_step_idx++)
                        {
                            const Configuration& current_config = contact_resolution_trace.contact_resolution_steps[contact_resolution_step_idx];
                            previous_config = current_config;
                            const std_msgs::ColorRGBA& current_color = (contact_resolution_step_idx == (contact_resolution_trace.contact_resolution_steps.size() - 1)) ? free_color : colliding_color;
                            visualization_msgs::Marker step_marker = DrawRobotConfiguration(robot_, current_config, current_color);
                            step_marker.ns = "step_state";
                            step_marker.id = 1;
                            visualization_msgs::Marker control_step_marker = DrawRobotControlInput(robot_, current_config, -control_input_step, control_step_color);
                            control_step_marker.ns = "control_step_state";
                            control_step_marker.id = 1;
                            visualization_msgs::MarkerArray simulator_step_display_rep;
                            simulator_step_display_rep.markers.push_back(step_marker);
                            simulator_step_display_rep.markers.push_back(control_step_marker);
                            display_pub.publish(simulator_step_display_rep);
                            // Wait for input
                            //std::cout << "Press ENTER to continue..." << std::endl;
                            ros::Duration(0.05).sleep();
                            //std::cin.get();
                        }
                    }
                }
            }
            else
            {
                const std::vector<Configuration, ConfigAlloc> trajectory = simple_simulator_interface::ExtractTrajectoryFromTrace(trace);
                const double time_interval = 1.0 / 25.0;
                const uint32_t rand_suffix = std::uniform_int_distribution<uint32_t>(1, 1000000)(rng_);
                const std::string ns = "simulator_test_" + std::to_string(rand_suffix);
                DrawParticlePolicyExecution(ns, trajectory, display_pub, time_interval, MakeColor(0.0f, 0.25f, 0.5f, 1.0f));
            }
            return trace;
        }

        /*
         * Planning function
         */
        inline std::pair<UncertaintyPlanningPolicy, std::map<std::string, double>> Plan(const Configuration& start, const Configuration& goal, const double goal_bias, const std::chrono::duration<double>& time_limit, const uint32_t edge_attempt_count, const uint32_t policy_action_attempt_count, const bool allow_contacts, const bool include_reverse_actions, const bool include_spur_actions, ros::Publisher& display_pub)
        {
            // Draw the simulation environment
            display_pub.publish(simulator_ptr_->ExportAllForDisplay());
            // Wait for input
            if (debug_level_ >= 10)
            {
                std::cout << "Press ENTER to draw start and goal states..." << std::endl;
                std::cin.get();
            }
            // Draw the start and goal
            std_msgs::ColorRGBA start_color;
            start_color.r = 1.0;
            start_color.g = 0.0;
            start_color.b = 0.0;
            start_color.a = 1.0;
            visualization_msgs::Marker start_marker = DrawRobotConfiguration(robot_, start, start_color);
            start_marker.ns = "start_state";
            start_marker.id = 1;
            std_msgs::ColorRGBA goal_color;
            goal_color.r = 0.0;
            goal_color.g = 1.0;
            goal_color.b = 0.0;
            goal_color.a = 1.0;
            visualization_msgs::Marker goal_marker = DrawRobotConfiguration(robot_, goal, goal_color);
            goal_marker.ns = "goal_state";
            goal_marker.id = 1;
            visualization_msgs::MarkerArray problem_display_rep;
            problem_display_rep.markers.push_back(start_marker);
            problem_display_rep.markers.push_back(goal_marker);
            display_pub.publish(problem_display_rep);
            // Wait for input
            if (debug_level_ >= 10)
            {
                std::cout << "Press ENTER to start planning..." << std::endl;
                std::cin.get();
            }
            UncertaintyPlanningState start_state(start);
            UncertaintyPlanningState goal_state(goal);
            // Bind the helper functions
            const std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
            std::function<int64_t(const UncertaintyPlanningTree&, const UncertaintyPlanningState&)> nearest_neighbor_fn = [&] (const UncertaintyPlanningTree& tree, const UncertaintyPlanningState& new_state) { return GetNearestNeighbor(tree, new_state); };
            std::function<bool(const UncertaintyPlanningState&)> goal_reached_fn = [&] (const UncertaintyPlanningState& goal_candidate) { return GoalReached(goal_candidate, goal_state, edge_attempt_count, allow_contacts); };
            std::function<void(UncertaintyPlanningTreeState&)> goal_reached_callback = [&] (UncertaintyPlanningTreeState& new_goal_state) { return GoalReachedCallback(new_goal_state, edge_attempt_count, start_time); };
            std::function<UncertaintyPlanningState(void)> state_sampling_fn = [&] (void) { return SampleRandomTargetState(); };
            std::uniform_real_distribution<double> goal_bias_distribution(0.0, 1.0);
            std::function<UncertaintyPlanningState(void)> complete_sampling_fn = [&](void) { if (goal_bias_distribution(rng_) > goal_bias) { auto state = state_sampling_fn(); arc_helpers::ConditionalPrint("Sampled state", 2, debug_level_); return state; } else { arc_helpers::ConditionalPrint("Sampled goal state", 2, debug_level_); return goal_state; } };
            std::function<std::vector<std::pair<UncertaintyPlanningState, int64_t>>(const UncertaintyPlanningState&, const UncertaintyPlanningState&)> forward_propagation_fn = [&] (const UncertaintyPlanningState& nearest, const UncertaintyPlanningState& target) { return PropagateForwardsAndDraw(nearest, target, edge_attempt_count, allow_contacts, include_reverse_actions, display_pub); };
            std::function<bool(void)> termination_check_fn = [&] (void) { return PlannerTerminationCheck(start_time, time_limit); };
            // Call the planner
            total_goal_reached_probability_ = 0.0;
            time_to_first_solution_ = 0.0;
            simulator_ptr_->ResetStatistics();
            std::pair<std::vector<std::vector<UncertaintyPlanningState>>, std::map<std::string, double>> planning_results = simple_rrt_planner::SimpleHybridRRTPlanner::PlanMultiPath(nearest_neighbors_storage_, start_state, nearest_neighbor_fn, goal_reached_fn, goal_reached_callback, complete_sampling_fn, forward_propagation_fn, termination_check_fn);
            // Make sure we got somewhere
            std::cout << "Planner terminated with goal reached probability: " << total_goal_reached_probability_ << std::endl;
            planning_results.second["P(goal reached)"] = total_goal_reached_probability_;
            planning_results.second["Time to first solution"] = time_to_first_solution_;
            clustering_performance_.ExportResults();
            std::cout << "Planner performed " << cluster_calls_ << " clustering calls, which took " << elapsed_clustering_time_ << " seconds of total runtime of which " << cluster_fallback_calls_ << " required hierarchical distance clustering" << std::endl;
            std::cout << "Planner statistics: " << PrettyPrint::PrettyPrint(planning_results.second) << std::endl;
            const std::map<std::string, double> simulator_resolve_statistics = simulator_ptr_->GetStatistics();
            planning_results.second.insert(simulator_resolve_statistics.begin(), simulator_resolve_statistics.end());
            planning_results.second["elapsed_clustering_time"] = elapsed_clustering_time_;
            planning_results.second["elapsed_simulation_time"] = elapsed_simulation_time_;
            planning_results.second["Particles stored"] = (double)particles_stored_;
            planning_results.second["Particles simulated"] = (double)particles_simulated_;
            planning_results.second["Goal candidates evaluated"] = (double)goal_candidates_evaluated_;
            planning_results.second["Goal reaching performed"] = (double)goal_reaching_performed_;
            planning_results.second["Goal reaching successful"] = (double)goal_reaching_successful_;
            if (total_goal_reached_probability_ >= goal_probability_threshold_)
            {
                const UncertaintyPlanningTree postprocessed_tree = PostProcessTree(nearest_neighbors_storage_);
                const UncertaintyPlanningTree pruned_tree = PruneTree(postprocessed_tree, include_spur_actions);
                const UncertaintyPlanningPolicy policy = ExtractPolicy(pruned_tree, goal, edge_attempt_count, policy_action_attempt_count);
                planning_results.second["Extracted policy size"] = (double)policy.GetRawPolicy().GetNodesImmutable().size();
                if (debug_level_ >= 1)
                {
                    std::cout << "Press ENTER to draw planned paths..." << std::endl;
                    std::cin.get();
                }
                // Draw the final path(s)
                for (size_t pidx = 0; pidx < planning_results.first.size(); pidx++)
                {
                    const std::vector<UncertaintyPlanningState>& planned_path = planning_results.first[pidx];
                    if (planned_path.size() >= 2)
                    {
                        double goal_reached_probability = planned_path[planned_path.size() - 1].GetGoalPfeasibility() * planned_path[planned_path.size() - 1].GetMotionPfeasibility();
                        visualization_msgs::MarkerArray path_display_rep;
                        for (size_t idx = 0; idx < planned_path.size(); idx++)
                        {
                            const UncertaintyPlanningState& current_state = planned_path[idx];
                            const Configuration current_configuration = current_state.GetExpectation();
                            std_msgs::ColorRGBA forward_color;
                            forward_color.r = (float)(1.0 - goal_reached_probability);
                            forward_color.g = 0.0f;
                            forward_color.b = 0.0f;
                            forward_color.a = (float)current_state.GetMotionPfeasibility();
                            visualization_msgs::Marker forward_expectation_marker = DrawRobotConfiguration(robot_, current_configuration, forward_color);
                            forward_expectation_marker.id = (int)idx;
                            forward_expectation_marker.ns = "final_path_" + std::to_string(pidx + 1);
                            // Make the display color
                            std_msgs::ColorRGBA reverse_color;
                            reverse_color.r = (float)(1.0 - goal_reached_probability);
                            reverse_color.g = 0.0f;
                            reverse_color.b = 0.0f;
                            reverse_color.a = (float)current_state.GetReverseEdgePfeasibility();
                            visualization_msgs::Marker reverse_expectation_marker = DrawRobotConfiguration(robot_, current_configuration, reverse_color);
                            reverse_expectation_marker.id = (int)idx;
                            reverse_expectation_marker.ns = "final_path_reversible_" + std::to_string(pidx + 1);;
                            // Add the markers
                            path_display_rep.markers.push_back(forward_expectation_marker);
                            path_display_rep.markers.push_back(reverse_expectation_marker);
                        }
                        display_pub.publish(path_display_rep);
                    }
                }
                DrawPolicy(policy, display_pub);
                // Wait for input
                if (debug_level_ >= 1)
                {
                    std::cout << "Press ENTER to export policy and print statistics..." << std::endl;
                    std::cin.get();
                }
                return std::pair<UncertaintyPlanningPolicy, std::map<std::string, double>>(policy, planning_results.second);
            }
            else
            {
                const UncertaintyPlanningPolicy policy;
                planning_results.second["Extracted policy size"] = 0.0;
                // Wait for input
                if (debug_level_ >= 1)
                {
                    std::cout << "Press ENTER to export policy and print statistics..." << std::endl;
                    std::cin.get();
                }
                return std::pair<UncertaintyPlanningPolicy, std::map<std::string, double>>(policy, planning_results.second);
            }
        }

    protected:

        /*
         * Solution tree post-processing functions
         */
        inline UncertaintyPlanningTree PostProcessTree(const UncertaintyPlanningTree& planner_tree) const
        {
            std::cout << "Postprocessing planner tree in preparation for policy extraction..." << std::endl;
            std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
            // Let's do some post-processing to the planner tree - we don't want to mess with the original tree, so we copy it
            UncertaintyPlanningTree postprocessed_planner_tree = planner_tree;
            // We have already computed reversibility for all edges, however, we now need to update the P(goal reached) for reversible children
            // We start with a naive implementation of this - this works because given the process that the tree is generated, children *MUST* have higher indices than their parents, so we can depend on the parents
            // having been updated first by the time we get to an index. To make this parallelizable, we'll need to switch to an explicitly branch-based approach
            // Go through each state in the tree - we skip the initial state, since it has no transition
            for (size_t sdx = 1; sdx < postprocessed_planner_tree.size(); sdx++)
            {
                // Get the current state
                UncertaintyPlanningTreeState& current_state = postprocessed_planner_tree[sdx];
                const int64_t parent_index = current_state.GetParentIndex();
                // Get the parent state
                const UncertaintyPlanningTreeState& parent_state = postprocessed_planner_tree[(size_t)parent_index];
                // If the current state is on a goal branch
                if (current_state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
                {
                    // Reversibility has already been computed
                    continue;
                }
                // If we are a non-goal child of a goal branch state
                else if (parent_state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
                {
                    // Make sure we're a child of a split where at least one child reaches the goal
                    const uint64_t transition_id = current_state.GetValueImmutable().GetTransitionId();
                    const uint64_t state_id = current_state.GetValueImmutable().GetStateId();
                    bool result_of_goal_reaching_split = false;
                    const std::vector<int64_t>& other_children = parent_state.GetChildIndices();
                    for (size_t idx = 0; idx < other_children.size(); idx++)
                    {
                        const int64_t other_child_index = other_children[idx];
                        const UncertaintyPlanningTreeState& other_child_state = postprocessed_planner_tree[(size_t)other_child_index];
                        const uint64_t other_child_transition_id = other_child_state.GetValueImmutable().GetTransitionId();
                        const uint64_t other_child_state_id = other_child_state.GetValueImmutable().GetStateId();
                        // If it's a child of the same split that produced us
                        if ((state_id != other_child_state_id) && (transition_id == other_child_transition_id))
                        {
                            const double other_child_goal_probability = other_child_state.GetValueImmutable().GetGoalPfeasibility();
                            if (other_child_goal_probability > 0.0)
                            {
                                result_of_goal_reaching_split = true;
                                break;
                            }
                        }
                    }
                    if (result_of_goal_reaching_split)
                    {
                        // Update P(goal reached) based on our ability to reverse to the goal branch
                        const double parent_pgoalreached = parent_state.GetValueImmutable().GetGoalPfeasibility();
                        const double new_pgoalreached = -(parent_pgoalreached * current_state.GetValueImmutable().GetReverseEdgePfeasibility()); // We use negative goal reached probabilities to signal probability due to reversing
                        current_state.GetValueMutable().SetGoalPfeasibility(new_pgoalreached);
                    }
                }
            }
            std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> postprocessing_time(end_time - start_time);
            std::cout << "...postprocessing complete, took " << postprocessing_time.count() << " seconds" << std::endl;
            return postprocessed_planner_tree;
        }

        inline UncertaintyPlanningTree PruneTree(const UncertaintyPlanningTree& planner_tree, const bool include_spur_actions) const
        {
            if (planner_tree.size() <= 1)
            {
                return planner_tree;
            }
            // Test to make sure the tree linkage is intact
            assert(simple_rrt_planner::SimpleHybridRRTPlanner::CheckTreeLinkage(planner_tree));
            std::cout << "Pruning planner tree in preparation for policy extraction..." << std::endl;
            std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
            // Let's do some post-processing to the planner tree - we don't want to mess with the original tree, so we copy it
            UncertaintyPlanningTree intermediate_planner_tree = planner_tree;
            // Loop through the tree and prune unproductive nodes+edges
            for (size_t idx = 0; idx < intermediate_planner_tree.size(); idx++)
            {
                UncertaintyPlanningTreeState& current_state = intermediate_planner_tree[idx];
                assert(current_state.IsInitialized());
                // If we're on a path to the goal, we always keep it
                if (current_state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
                {
                    continue;
                }
                // If the current node can reverse to reach the goal
                else if (current_state.GetValueImmutable().GetGoalPfeasibility() < -0.0)
                {
                    // If we allow spur nodes, we keep it
                    if (include_spur_actions)
                    {
                        continue;
                    }
                    // If not, prune the node
                    else
                    {
                        current_state.SetParentIndex(-1);
                        current_state.ClearChildIndicies();
                    }
                }
                // We always prune nodes that can't reach the goal
                else
                {
                    current_state.SetParentIndex(-1);
                    current_state.ClearChildIndicies();
                }
            }
            // Now, extract the pruned tree
            UncertaintyPlanningTree pruned_planner_tree;
            // Add root state
            UncertaintyPlanningTreeState root_state = intermediate_planner_tree[0];
            assert(root_state.IsInitialized());
            pruned_planner_tree.push_back(root_state);
            // Recursive call to extract live branches
            ExtractChildStates(intermediate_planner_tree, 0, 0, pruned_planner_tree);
            // Test to make sure the tree linkage is intact
            assert(simple_rrt_planner::SimpleHybridRRTPlanner::CheckTreeLinkage(pruned_planner_tree));
            std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> pruning_time(end_time - start_time);
            std::cout << "...pruning complete, pruned to " << pruned_planner_tree.size() << " states, took " << pruning_time.count() << " seconds" << std::endl;
            return pruned_planner_tree;
        }

        /*
         * Policy generation wrapper function
         */
        inline UncertaintyPlanningPolicy ExtractPolicy(const UncertaintyPlanningTree& planner_tree, const Configuration& goal, const uint32_t planner_action_try_attempts, const uint32_t policy_action_attempt_count) const
        {
            const double marginal_edge_weight = 0.05;
            const UncertaintyPlanningPolicy policy(planner_tree, goal, marginal_edge_weight, goal_probability_threshold_, planner_action_try_attempts, policy_action_attempt_count);
            return policy;
        }

        inline void LogParticleTrajectories(const std::vector<std::vector<Configuration, ConfigAlloc>>& particle_executions, const std::string& filename) const
        {
            std::ofstream log_file(filename, std::ios_base::out);
            if (!log_file.is_open())
            {
                std::cerr << "\x1b[31;1m Unable to create folder/file to log to: " << filename << "\x1b[0m \n";
                throw std::invalid_argument("Log filename must be write-openable");
            }
            for (size_t idx = 0; idx < particle_executions.size(); idx++)
            {
                const std::vector<Configuration, ConfigAlloc>& particle_trajectory = particle_executions[idx];
                log_file << "Particle trajectory " << (idx + 1) << std::endl;
                for (size_t sdx = 0; sdx < particle_trajectory.size(); sdx++)
                {
                    const Configuration& config = particle_trajectory[sdx];
                    log_file << PrettyPrint::PrettyPrint(config) << std::endl;
                }
            }
            log_file.close();
        }

    public:

        /*
         * Policy simulation and execution functions
         */
        inline std::pair<UncertaintyPlanningPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> SimulateExectionPolicy(UncertaintyPlanningPolicy policy, const Configuration& start, const Configuration& goal, const uint32_t num_executions, const uint32_t exec_step_limit, ros::Publisher& display_pub, const bool wait_for_user, const double draw_wait) const
        {
            simulator_ptr_->ResetStatistics();
            std::vector<std::vector<Configuration, ConfigAlloc>> particle_executions(num_executions);
            std::vector<int64_t> policy_execution_step_counts(num_executions, 0u);
            std::vector<double> policy_execution_times(num_executions, -0.0);
            uint32_t reached_goal = 0;
            for (size_t idx = 0; idx < num_executions; idx++)
            {
                const std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
                const std::pair<std::vector<Configuration, ConfigAlloc>, std::pair<UncertaintyPlanningPolicy, int64_t>> particle_execution = SimulateSinglePolicyExecution(policy, start, goal, exec_step_limit, rng_, display_pub, wait_for_user);
                const std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double> execution_time(end_time - start_time);
                const double execution_seconds = execution_time.count();
                policy_execution_times[idx] = execution_seconds;
                particle_executions[idx] = particle_execution.first;
                policy = particle_execution.second.first;
                const int64_t policy_execution_step_count = particle_execution.second.second;
                policy_execution_step_counts[idx] = policy_execution_step_count;
                if (policy_execution_step_count >= 0)
                {
                    reached_goal++;
                    std::cout << "...finished policy execution " << idx + 1 << " of " << num_executions << " successfully, " << reached_goal << " successful so far" << std::endl;
                }
                else
                {
                    std::cout << "...finished policy execution " << idx + 1 << " of " << num_executions << " unsuccessfully, " << reached_goal << " successful so far" << std::endl;
                }
            }
            // Draw the trajectory in a pretty way
            if (wait_for_user)
            {
                // Wait for input
                std::cout << "Press ENTER to draw pretty simulation tracks..." << std::endl;
                std::cin.get();
            }
            for (size_t idx = 0; idx < num_executions; idx++)
            {
                const std::string ns = "policy_simulation_" + std::to_string(idx + 1);
                DrawParticlePolicyExecution(ns, particle_executions[idx], display_pub, draw_wait, MakeColor(0.0f, 0.0f, 0.8f, 0.25f));
            }
            const double policy_success = (double)reached_goal / (double)num_executions;
            std::map<std::string, double> policy_statistics;
            policy_statistics["(Simulation) Policy success"] = policy_success;
            const std::map<std::string, double> simulator_resolve_statistics = simulator_ptr_->GetStatistics();
            policy_statistics.insert(simulator_resolve_statistics.begin(), simulator_resolve_statistics.end());
            if (debug_level_ >= 15)
            {
                LogParticleTrajectories(particle_executions, "/tmp/policy_simulation_trajectories.csv");
            }
            return std::make_pair(policy, std::make_pair(policy_statistics, std::make_pair(policy_execution_step_counts, policy_execution_times)));
        }

        inline std::pair<UncertaintyPlanningPolicy, std::pair<std::map<std::string, double>, std::pair<std::vector<int64_t>, std::vector<double>>>> ExecuteExectionPolicy(UncertaintyPlanningPolicy policy, const Configuration& start, const Configuration& goal, const std::function<std::vector<Configuration, ConfigAlloc>(const Configuration&, const Configuration&, const double, const double, const bool)>& move_fn, const uint32_t num_executions, const double exec_time_limit, ros::Publisher& display_pub, const bool wait_for_user, const double draw_wait) const
        {
            // Buffer for a teensy bit of time
            for (size_t iter = 0; iter < 100; iter++)
            {
                ros::spinOnce();
                ros::Duration(0.005).sleep();
            }
            std::vector<std::vector<Configuration, ConfigAlloc>> particle_executions(num_executions);
            std::vector<int64_t> policy_execution_step_counts(num_executions, 0u);
            std::vector<double> policy_execution_times(num_executions, -0.0);
            uint32_t reached_goal = 0;
            for (size_t idx = 0; idx < num_executions; idx++)
            {
                std::cout << "Starting policy execution " << idx << "..." << std::endl;
                const double start_time = ros::Time::now().toSec();
                std::pair<std::vector<Configuration, ConfigAlloc>, std::pair<UncertaintyPlanningPolicy, int64_t>> particle_execution = ExecuteSinglePolicyExecution(policy, start, goal, move_fn, exec_time_limit, display_pub, wait_for_user);
                const double end_time = ros::Time::now().toSec();
                std::cout << "Started policy exec @ " << start_time << " finished policy exec @ " << end_time << std::endl;
                const double execution_seconds = end_time - start_time;
                policy_execution_times[idx] = execution_seconds;
                particle_executions[idx] = particle_execution.first;
                policy = particle_execution.second.first;
                const int64_t policy_execution_step_count = particle_execution.second.second;
                policy_execution_step_counts[idx] = policy_execution_step_count;
                if (policy_execution_step_count >= 0)
                {
                    reached_goal++;
                    std::cout << "...finished policy execution " << idx + 1 << " of " << num_executions << " successfully in " << execution_seconds << " seconds, " << reached_goal << " successful so far" << std::endl;
                }
                else
                {
                    std::cout << "...finished policy execution " << idx + 1 << " of " << num_executions << " unsuccessfully in " << execution_seconds << " seconds, " << reached_goal << " successful so far" << std::endl;
                }
            }
            // Draw the trajectory in a pretty way
            if (wait_for_user)
            {
                // Wait for input
                std::cout << "Press ENTER to draw pretty simulation tracks..." << std::endl;
                std::cin.get();
            }
            for (size_t idx = 0; idx < num_executions; idx++)
            {
                const std::string ns = "policy_execution_" + std::to_string(idx + 1);
                //DrawParticlePolicyExecution(ns, particle_executions[idx], display_pub, draw_wait, MakeColor(0.0f, 0.8f, 0.0f, 0.25f));
                DrawParticlePolicyExecution(ns, particle_executions[idx], display_pub, draw_wait, MakeColor(0.0f, 0.0f, 0.0f, 1.0f));
            }
            const double policy_success = (double)reached_goal / (double)num_executions;
            std::map<std::string, double> policy_statistics;
            policy_statistics["(Execution) Policy success"] = policy_success;
            if (debug_level_ >= 15)
            {
                LogParticleTrajectories(particle_executions, "/tmp/policy_execution_trajectories.csv");
            }
            return std::make_pair(policy, std::make_pair(policy_statistics, std::make_pair(policy_execution_step_counts, policy_execution_times)));
        }

    protected:

        static inline std_msgs::ColorRGBA MakeColor(const float r, const float g, const float b, const float a)
        {
            return arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(r, g, b, a);
        }

        inline std::pair<std::vector<Configuration, ConfigAlloc>, std::pair<UncertaintyPlanningPolicy, int64_t>> SimulateSinglePolicyExecution(UncertaintyPlanningPolicy policy, const Configuration& start, const Configuration& goal, const uint32_t exec_step_limit, PRNG& rng, ros::Publisher& display_pub, const bool wait_for_user) const
        {
            std::cout << "Drawing environment..." << std::endl;
            display_pub.publish(simulator_ptr_->ExportAllForDisplay());
            if (wait_for_user)
            {
                std::cout << "Press ENTER to continue..." << std::endl;
                std::cin.get();
            }
            else
            {
                // Wait for a bit
                std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
            }
            std::cout << "Drawing initial policy..." << std::endl;
            DrawPolicy(policy, display_pub);
            if (wait_for_user)
            {
                std::cout << "Press ENTER to continue..." << std::endl;
                std::cin.get();
            }
            else
            {
                // Wait for a bit
                std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
            }
            // Let's do this
            std::function<std::vector<std::vector<size_t>>(const std::vector<Configuration, ConfigAlloc>&, const Configuration&)> policy_particle_clustering_fn = [&] (const std::vector<Configuration, ConfigAlloc>& particles, const Configuration& config) { return PolicyParticleClusteringFn(particles, config, display_pub); };
            std::vector<Configuration, ConfigAlloc> particle_trajectory;
            particle_trajectory.push_back(start);
            uint64_t desired_transition_id = 0;
            uint32_t current_exec_step = 0u;
            while (current_exec_step < exec_step_limit)
            {
                current_exec_step++;
                // Get the current configuration
                const Configuration& current_config = particle_trajectory.back();
                // Get the next action
                const std::pair<std::pair<int64_t, uint64_t>, std::pair<Configuration, Configuration>> policy_query = policy.QueryBestAction(desired_transition_id, current_config, policy_particle_clustering_fn);
                const int64_t previous_state_idx = policy_query.first.first;
                desired_transition_id = policy_query.first.second;
                const Configuration& action = policy_query.second.first;
                //const Configuration& expected_result = policy_query.second.second;
                //std::cout << "Queried policy, received action " << PrettyPrint::PrettyPrint(action) << " for current state " << PrettyPrint::PrettyPrint(current_config) << " and parent state index " << parent_state_idx << std::endl;
                //std::cout << "----------\nReceived new action for best matching state index " << previous_state_idx << " with transition ID " << desired_transition_id << "\n==========" << std::endl;
                //std::cout << "Drawing updated policy..." << std::endl;
                DrawPolicy(policy, display_pub);
                DrawLocalPolicy(policy, 0, display_pub, MakeColor(0.0, 1.0, 1.0, 1.0), "policy_start_to_goal");;
                DrawLocalPolicy(policy, previous_state_idx, display_pub, MakeColor(0.0, 0.0, 1.0, 1.0), "policy_here_to_goal");
                //std::cout << "Drawing current config (blue), parent state (cyan), and action (magenta)..." << std::endl;
                const UncertaintyPlanningState& parent_state = policy.GetRawPolicy().GetNodeImmutable(previous_state_idx).GetValueImmutable();
                const Configuration parent_state_config = parent_state.GetExpectation();
                std_msgs::ColorRGBA parent_state_color;
                parent_state_color.r = 0.0f;
                parent_state_color.g = 0.5f;
                parent_state_color.b = 1.0f;
                parent_state_color.a = 0.5f;
                visualization_msgs::Marker parent_state_marker = DrawRobotConfiguration(robot_, parent_state_config, parent_state_color);
                parent_state_marker.id = 1;
                parent_state_marker.ns = "parent_state_marker";
                std_msgs::ColorRGBA current_config_color;
                current_config_color.r = 0.0f;
                current_config_color.g = 0.0f;
                current_config_color.b = 1.0f;
                current_config_color.a = 0.5f;
                visualization_msgs::Marker current_config_marker = DrawRobotConfiguration(robot_, current_config, current_config_color);
                current_config_marker.id = 1;
                current_config_marker.ns = "current_config_marker";
                std_msgs::ColorRGBA action_color;
                action_color.r = 1.0f;
                action_color.g = 0.0f;
                action_color.b = 1.0f;
                action_color.a = 0.5f;
                visualization_msgs::Marker action_marker = DrawRobotConfiguration(robot_, action, action_color);
                action_marker.id = 1;
                action_marker.ns = "action_marker";
                visualization_msgs::MarkerArray policy_query_markers;
                policy_query_markers.markers = {current_config_marker, parent_state_marker, action_marker};
                display_pub.publish(policy_query_markers);
                if (wait_for_user)
                {
                    std::cout << "Press ENTER to continue & execute..." << std::endl;
                    std::cin.get();
                }
                else
                {
                    // Wait for a bit
                    std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
                }
                // Simulate fowards
                const std::vector<Configuration, ConfigAlloc> execution_states = SimulatePolicyStep(current_config, action, rng, display_pub);
                particle_trajectory.insert(particle_trajectory.end(), execution_states.begin(), execution_states.end());
                const Configuration result_config = particle_trajectory.back();
                // Check if we've reached the goal
                if (robot_.ComputeConfigurationDistance(result_config, goal) <= goal_distance_threshold_)
                {
                    // We've reached the goal!
                    std::cout << "Policy simulation reached the goal in " << current_exec_step << " steps out of a maximum of " << exec_step_limit << " steps" << std::endl;
                    return std::make_pair(particle_trajectory, std::make_pair(policy, (int64_t)current_exec_step));
                }
            }
            // If we get here, we haven't reached the goal!
            std::cout << "Policy simulation failed to reach the goal in " << current_exec_step << " steps out of a maximum of " << exec_step_limit << " steps" << std::endl;
            return std::make_pair(particle_trajectory, std::make_pair(policy, -((int64_t)current_exec_step)));
        }

        inline std::pair<std::vector<Configuration, ConfigAlloc>, std::pair<UncertaintyPlanningPolicy, int64_t>> ExecuteSinglePolicyExecution(UncertaintyPlanningPolicy policy, const Configuration& start, const Configuration& goal, const std::function<std::vector<Configuration, ConfigAlloc>(const Configuration&, const Configuration&, const double, const double, const bool)>& move_fn, const double exec_time_limit, ros::Publisher& display_pub, const bool wait_for_user) const
        {
            std::cout << "Drawing environment..." << std::endl;
            display_pub.publish(simulator_ptr_->ExportAllForDisplay());
            if (wait_for_user)
            {
                std::cout << "Press ENTER to continue..." << std::endl;
                std::cin.get();
            }
            else
            {
                // Wait for a bit
                std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
            }
            std::cout << "Drawing initial policy..." << std::endl;
            DrawPolicy(policy, display_pub);
            if (wait_for_user)
            {
                std::cout << "Press ENTER to continue..." << std::endl;
                std::cin.get();
            }
            else
            {
                // Wait for a bit
                std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
            }
            // Let's do this
            std::function<std::vector<std::vector<size_t>>(const std::vector<Configuration, ConfigAlloc>&, const Configuration&)> policy_particle_clustering_fn = [&] (const std::vector<Configuration, ConfigAlloc>& particles, const Configuration& config) { return PolicyParticleClusteringFn(particles, config, display_pub); };
            const double step_shortcut_distance = (goal_distance_threshold_ * 0.5);
            // Reset the robot first
            std::cout << "Reseting before policy execution..." << std::endl;
            move_fn(start, start, step_duration_, step_shortcut_distance, true);
            std::cout << "Executing policy..." << std::endl;
            std::vector<Configuration, ConfigAlloc> particle_trajectory;
            particle_trajectory.push_back(start);
            uint64_t desired_transition_id = 0;
            uint32_t current_exec_step = 0u;
            const double start_time = ros::Time::now().toSec();
            double current_time = start_time;
            while (((wait_for_user == false) && ((current_time - start_time) <= exec_time_limit)) || ((wait_for_user == true) && (current_exec_step < 1000u)))
            {
                current_exec_step++;
                // Get the current configuration
                const Configuration& current_config = particle_trajectory.back();
                // Get the next action
                const std::pair<std::pair<int64_t, uint64_t>, std::pair<Configuration, Configuration>> policy_query = policy.QueryBestAction(desired_transition_id, current_config, policy_particle_clustering_fn);
                const int64_t previous_state_idx = policy_query.first.first;
                desired_transition_id = policy_query.first.second;
                const Configuration& action = policy_query.second.first;
                const Configuration& expected_result = policy_query.second.second;
                std::cout << "----------\nReceived new action for best matching state index " << previous_state_idx << " with transition ID " << desired_transition_id << "\n==========" << std::endl;
                std::cout << "Drawing updated policy..." << std::endl;
                DrawPolicy(policy, display_pub);
                DrawLocalPolicy(policy, 0, display_pub, MakeColor(0.0, 0.0, 1.0, 1.0), "policy_start_to_goal");;
                DrawLocalPolicy(policy, previous_state_idx, display_pub, MakeColor(0.0, 0.0, 1.0, 1.0), "policy_here_to_goal");
                std::cout << "Drawing current config (blue), parent state (cyan), and action (magenta)..." << std::endl;
                const UncertaintyPlanningState& parent_state = policy.GetRawPolicy().GetNodeImmutable(previous_state_idx).GetValueImmutable();
                const Configuration parent_state_config = parent_state.GetExpectation();
                std_msgs::ColorRGBA parent_state_color;
                parent_state_color.r = 0.0f;
                parent_state_color.g = 0.5f;
                parent_state_color.b = 1.0f;
                parent_state_color.a = 0.5f;
                visualization_msgs::Marker parent_state_marker = DrawRobotConfiguration(robot_, parent_state_config, parent_state_color);
                parent_state_marker.id = 1;
                parent_state_marker.ns = "parent_state_marker";
                std_msgs::ColorRGBA current_config_color;
                current_config_color.r = 0.0f;
                current_config_color.g = 0.0f;
                current_config_color.b = 1.0f;
                current_config_color.a = 0.5f;
                visualization_msgs::Marker current_config_marker = DrawRobotConfiguration(robot_, current_config, current_config_color);
                current_config_marker.id = 1;
                current_config_marker.ns = "current_config_marker";
                std_msgs::ColorRGBA action_color;
                action_color.r = 1.0f;
                action_color.g = 0.0f;
                action_color.b = 1.0f;
                action_color.a = 0.5f;
                visualization_msgs::Marker action_marker = DrawRobotConfiguration(robot_, action, action_color);
                action_marker.id = 1;
                action_marker.ns = "action_marker";
                visualization_msgs::MarkerArray policy_query_markers;
                policy_query_markers.markers = {current_config_marker, parent_state_marker, action_marker};
                display_pub.publish(policy_query_markers);
                if (wait_for_user)
                {
                    std::cout << "Press ENTER to continue & execute..." << std::endl;
                    std::cin.get();
                }
                else
                {
                    // Wait for a bit
                    std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
                }
                // Simulate fowards
                const std::vector<Configuration, ConfigAlloc> execution_states = move_fn(action, expected_result, step_duration_, step_shortcut_distance, false);
                particle_trajectory.insert(particle_trajectory.end(), execution_states.begin(), execution_states.end());
                const Configuration result_config = particle_trajectory.back();
                // Check if we've reached the goal
                if (robot_.ComputeConfigurationDistance(result_config, goal) <= goal_distance_threshold_)
                {
                    current_time = ros::Time::now().toSec();
                    // We've reached the goal!
                    std::cout << "Policy execution reached the goal in " << (current_time - start_time) << " seconds out of a maximum of " << exec_time_limit << " seconds" << std::endl;
                    return std::make_pair(particle_trajectory, std::make_pair(policy, (int64_t)current_exec_step));
                }
                else
                {
                    current_time = ros::Time::now().toSec();
                }
            }
            // If we get here, we haven't reached the goal!
            std::cout << "Policy execution failed to reach the goal in " << (current_time - start_time) << " seconds out of a maximum of " << exec_time_limit << " seconds" << std::endl;
            return std::make_pair(particle_trajectory, std::make_pair(policy, -((int64_t)current_exec_step)));
        }

        inline std::vector<Configuration, ConfigAlloc> SimulatePolicyStep(const Configuration& current_config, const Configuration& action, PRNG& rng, ros::Publisher& display_pub) const
        {
            simple_simulator_interface::ForwardSimulationStepTrace<Configuration, ConfigAlloc> trace;
            simulator_ptr_->ForwardSimulateRobot(robot_, current_config, action, rng, step_duration_, (goal_distance_threshold_ * 0.5), simulate_with_individual_jacobians_, true, trace, true, display_pub);
            std::vector<Configuration, ConfigAlloc> execution_trajectory = simple_simulator_interface::ExtractTrajectoryFromTrace(trace);
            if (execution_trajectory.empty())
            {
                std::cerr << "Exec trajectory is empty, this should not happen!" << std::endl;
            }
            return execution_trajectory;
        }

        /*
         * Drawing functions
         */
        inline void DrawParticlePolicyExecution(const std::string& ns, const std::vector<Configuration, ConfigAlloc>& trajectory, ros::Publisher& display_pub, const double draw_wait, const std_msgs::ColorRGBA& color) const
        {
            if (trajectory.size() > 1)
            {
                // Draw one step at a time
                for (size_t idx = 0; idx < trajectory.size(); idx++)
                {
                    const Configuration& current_configuration = trajectory[idx];
                    // Draw a ball at the current location
                    visualization_msgs::Marker current_marker = DrawRobotConfiguration(robot_, current_configuration, color);
                    visualization_msgs::Marker trace_marker = current_marker;
                    trace_marker.ns = ns;
                    trace_marker.id = (int)idx + 1;
                    current_marker.ns = "current_policy_exec";
                    current_marker.id = 1;
                    // Send the markers for display
                    visualization_msgs::MarkerArray display_markers;
                    display_markers.markers = {current_marker, trace_marker};
                    display_pub.publish(display_markers);
                    // Wait for a bit
                    std::this_thread::sleep_for(std::chrono::duration<double>(draw_wait));
                }
            }
            else
            {
                return;
            }
        }

        inline Eigen::Vector4d Get3DPointForConfig(const Robot& immutable_robot, const Configuration& config) const
        {
            Robot robot = immutable_robot;
            const std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> robot_links_points = robot.GetRawLinksPoints();
            robot.UpdatePosition(config);
            const std::string& link_name = robot_links_points.back().first;
            const EigenHelpers::VectorVector4d& link_points = (*robot_links_points.back().second);
            const Eigen::Isometry3d link_transform = robot.GetLinkTransform(link_name);
            //const Eigen::Vector3d link_relative_point(0.0, 0.0, 0.0);
            const Eigen::Vector4d& link_relative_point = link_points.back();
            const Eigen::Vector4d config_point = link_transform * link_relative_point;
            return config_point;
        }

        inline void DrawPolicy(const UncertaintyPlanningPolicy& policy, ros::Publisher& display_pub) const
        {
            const ExecutionPolicyGraph& policy_graph = policy.GetRawPolicy();
            const std::vector<int64_t>& previous_index_map = policy.GetRawPreviousIndexMap();
            assert(policy_graph.GetNodesImmutable().size() == previous_index_map.size());
            visualization_msgs::MarkerArray policy_markers;
            std_msgs::ColorRGBA forward_color;
            forward_color.r = 0.0f;
            forward_color.g = 0.0f;
            forward_color.b = 0.0f;
            forward_color.a = 1.0f;
            std_msgs::ColorRGBA backward_color = forward_color;
            std_msgs::ColorRGBA blue_color;
            blue_color.r = 0.0f;
            blue_color.g = 0.0f;
            blue_color.b = 1.0f;
            blue_color.a = 1.0f;
            for (size_t idx = 0; idx < previous_index_map.size(); idx++)
            {
                const int64_t current_index = (int64_t)idx;
                const int64_t previous_index = previous_index_map[idx];
                assert(previous_index >= 0);
                if (current_index == previous_index)
                {
                    const Configuration current_config = policy_graph.GetNodeImmutable(current_index).GetValueImmutable().GetExpectation();
                    visualization_msgs::Marker target_marker = DrawRobotConfiguration(robot_, current_config, blue_color);
                    target_marker.ns = "policy_graph";
                    target_marker.id = (int)idx + 1;
                    policy_markers.markers.push_back(target_marker);
                }
                else
                {
                    const Configuration current_config = policy_graph.GetNodeImmutable(current_index).GetValueImmutable().GetExpectation();
                    const Configuration previous_config = policy_graph.GetNodeImmutable(previous_index).GetValueImmutable().GetExpectation();
                    const Eigen::Vector4d current_config_point = Get3DPointForConfig(robot_, current_config);
                    const Eigen::Vector4d previous_config_point = Get3DPointForConfig(robot_, previous_config);
                    visualization_msgs::Marker edge_marker;
                    edge_marker.action = visualization_msgs::Marker::ADD;
                    edge_marker.ns = "policy_graph";
                    edge_marker.id = (int)idx + 1;
                    edge_marker.frame_locked = false;
                    edge_marker.lifetime = ros::Duration(0.0);
                    edge_marker.type = visualization_msgs::Marker::ARROW;
                    edge_marker.header.frame_id = simulator_ptr_->GetFrame();
                    edge_marker.scale.x = simulator_ptr_->GetResolution() * 0.5;
                    edge_marker.scale.y = simulator_ptr_->GetResolution() * 1.5;
                    edge_marker.scale.z = simulator_ptr_->GetResolution() * 1.5;
                    const Eigen::Isometry3d base_transform = Eigen::Isometry3d::Identity();
                    edge_marker.pose = EigenHelpersConversions::EigenIsometry3dToGeometryPose(base_transform);
                    if (current_index < previous_index)
                    {
                        edge_marker.color = forward_color;
                    }
                    else if (previous_index < current_index)
                    {
                        edge_marker.color = backward_color;
                    }
                    else
                    {
                        continue;
                    }
                    edge_marker.points.push_back(EigenHelpersConversions::EigenVector4dToGeometryPoint(current_config_point));
                    edge_marker.points.push_back(EigenHelpersConversions::EigenVector4dToGeometryPoint(previous_config_point));
                    policy_markers.markers.push_back(edge_marker);
                }
            }
            std::cout << "Drawing policy graph with " << policy_markers.markers.size() << " edges" << std::endl;
            display_pub.publish(policy_markers);
        }

        inline void DrawLocalPolicy(const UncertaintyPlanningPolicy& policy, const int64_t current_state_idx, ros::Publisher& display_pub, const std_msgs::ColorRGBA& color, const std::string& policy_name) const
        {
            const ExecutionPolicyGraph& policy_graph = policy.GetRawPolicy();
            const std::vector<int64_t>& previous_index_map = policy.GetRawPreviousIndexMap();
            assert(policy_graph.GetNodesImmutable().size() == previous_index_map.size());
            assert(current_state_idx >= 0);
            assert(current_state_idx < (int64_t)previous_index_map.size());
            visualization_msgs::MarkerArray policy_markers;
            std_msgs::ColorRGBA blue_color;
            blue_color.r = 0.0f;
            blue_color.g = 0.0f;
            blue_color.b = 1.0f;
            blue_color.a = 1.0f;
            const Configuration previous_config = policy_graph.GetNodeImmutable(current_state_idx).GetValueImmutable().GetExpectation();
            Eigen::Vector4d previous_point = Get3DPointForConfig(robot_, previous_config);
            int64_t previous_index = previous_index_map[(size_t)current_state_idx];
            int idx = 1;
            while (previous_index != -1)
            {
                const int64_t current_idx = previous_index;
                const Configuration current_config = policy_graph.GetNodeImmutable(current_idx).GetValueImmutable().GetExpectation();
                const Eigen::Vector4d current_config_point = Get3DPointForConfig(robot_, current_config);
                visualization_msgs::Marker edge_marker;
                edge_marker.action = visualization_msgs::Marker::ADD;
                edge_marker.ns = policy_name;
                edge_marker.id = idx;
                idx++;
                edge_marker.frame_locked = false;
                edge_marker.lifetime = ros::Duration(0.0);
                edge_marker.type = visualization_msgs::Marker::ARROW;
                edge_marker.header.frame_id = simulator_ptr_->GetFrame();
                edge_marker.scale.x = simulator_ptr_->GetResolution() * 0.5;
                edge_marker.scale.y = simulator_ptr_->GetResolution() * 1.5;
                edge_marker.scale.z = simulator_ptr_->GetResolution() * 1.5;
                const Eigen::Isometry3d base_transform = Eigen::Isometry3d::Identity();
                edge_marker.pose = EigenHelpersConversions::EigenIsometry3dToGeometryPose(base_transform);
                edge_marker.color = color;
                edge_marker.points.push_back(EigenHelpersConversions::EigenVector4dToGeometryPoint(previous_point));
                edge_marker.points.push_back(EigenHelpersConversions::EigenVector4dToGeometryPoint(current_config_point));
                policy_markers.markers.push_back(edge_marker);
                previous_index = previous_index_map[(size_t)current_idx];
                if (previous_index == current_idx)
                {
                    previous_index = -1;
                }
                previous_point = current_config_point;
            }
            std::cout << "Drawing local policy graph with " << policy_markers.markers.size() << " edges" << std::endl;
            display_pub.publish(policy_markers);
        }

        inline visualization_msgs::Marker DrawRobotConfiguration(const Robot& immutable_robot, const Configuration& configuration, const std_msgs::ColorRGBA& color) const
        {
            Robot robot = immutable_robot;
            std_msgs::ColorRGBA real_color = color;
            visualization_msgs::Marker configuration_marker;
            configuration_marker.action = visualization_msgs::Marker::ADD;
            configuration_marker.ns = "UNKNOWN";
            configuration_marker.id = 1;
            configuration_marker.frame_locked = false;
            configuration_marker.lifetime = ros::Duration(0.0);
            configuration_marker.type = visualization_msgs::Marker::SPHERE_LIST;
            configuration_marker.header.frame_id = simulator_ptr_->GetFrame();
            configuration_marker.scale.x = simulator_ptr_->GetResolution();
            configuration_marker.scale.y = simulator_ptr_->GetResolution();
            configuration_marker.scale.z = simulator_ptr_->GetResolution();
            const Eigen::Isometry3d base_transform = Eigen::Isometry3d::Identity();
            configuration_marker.pose = EigenHelpersConversions::EigenIsometry3dToGeometryPose(base_transform);
            configuration_marker.color = real_color;
            // Make the individual points
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> robot_links_points = robot.GetRawLinksPoints();
            // Update the position of the robot
            robot.UpdatePosition(configuration);
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector4d& link_points = (*robot_links_points[link_idx].second);
                // Get the transform of the current link
                const Eigen::Isometry3d link_transform = robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector4d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector4d environment_relative_point = link_transform * link_relative_point;
                    const geometry_msgs::Point marker_point = EigenHelpersConversions::EigenVector4dToGeometryPoint(environment_relative_point);
                    configuration_marker.points.push_back(marker_point);
                    if (link_relative_point.norm() == 0.0)
                    {
                        std_msgs::ColorRGBA black_color;
                        black_color.r = 0.0f;
                        black_color.g = 0.0f;
                        black_color.b = 0.0f;
                        black_color.a = 1.0f;
                        configuration_marker.colors.push_back(black_color);
                    }
                    else
                    {
                        configuration_marker.colors.push_back(real_color);
                    }
                }
            }
            return configuration_marker;
        }

        inline visualization_msgs::MarkerArray DrawParticles(const Robot& immutable_robot, const std::vector<Configuration, ConfigAlloc>& particles, const std_msgs::ColorRGBA& color, const std::string& ns) const
        {
            Robot robot = immutable_robot;
            visualization_msgs::MarkerArray markers;
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                visualization_msgs::Marker particle_marker = DrawRobotConfiguration(robot, particles[idx], color);
                particle_marker.ns = ns;
                particle_marker.id = (int32_t)idx + 1;
                markers.markers.push_back(particle_marker);
            }
            return markers;
        }

        inline visualization_msgs::MarkerArray DrawParticles(const Robot& immutable_robot, const std::vector<std::pair<Configuration, bool>>& particles, const std_msgs::ColorRGBA& color, const std::string& ns) const
        {
            Robot robot = immutable_robot;
            visualization_msgs::MarkerArray markers;
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                visualization_msgs::Marker particle_marker = DrawRobotConfiguration(robot, particles[idx].first, color);
                particle_marker.ns = ns;
                particle_marker.id = (int32_t)idx + 1;
                markers.markers.push_back(particle_marker);
            }
            return markers;
        }

        inline visualization_msgs::Marker DrawRobotControlInput(const Robot& immutable_robot, const Configuration& configuration, const Eigen::VectorXd& control_input, const std_msgs::ColorRGBA& color) const
        {
            Robot robot = immutable_robot;
            std_msgs::ColorRGBA real_color = color;
            visualization_msgs::Marker configuration_marker;
            configuration_marker.action = visualization_msgs::Marker::ADD;
            configuration_marker.ns = "UNKNOWN";
            configuration_marker.id = 1;
            configuration_marker.frame_locked = false;
            configuration_marker.lifetime = ros::Duration(0.0);
            configuration_marker.type = visualization_msgs::Marker::LINE_LIST;
            configuration_marker.header.frame_id = simulator_ptr_->GetFrame();
            configuration_marker.scale.x = simulator_ptr_->GetResolution() * 0.5;
            configuration_marker.scale.y = simulator_ptr_->GetResolution() * 0.5;
            configuration_marker.scale.z = simulator_ptr_->GetResolution() * 0.5;
            const Eigen::Isometry3d base_transform = Eigen::Isometry3d::Identity();
            configuration_marker.pose = EigenHelpersConversions::EigenIsometry3dToGeometryPose(base_transform);
            configuration_marker.color = real_color;
            // Make the individual points
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> robot_links_points = robot.GetRawLinksPoints();
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector4d& link_points = (*robot_links_points[link_idx].second);
                // Get the current transform
                // Update the position of the robot
                robot.UpdatePosition(configuration);
                // Get the transform of the current link
                const Eigen::Isometry3d current_link_transform = robot.GetLinkTransform(link_name);
                // Apply the control input
                robot.ApplyControlInput(control_input);
                // Get the transform of the current link
                const Eigen::Isometry3d current_plus_control_link_transform = robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector4d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector4d environment_relative_current_point = current_link_transform * link_relative_point;
                    const Eigen::Vector4d environment_relative_current_plus_control_point = current_plus_control_link_transform * link_relative_point;
                    const geometry_msgs::Point current_marker_point = EigenHelpersConversions::EigenVector4dToGeometryPoint(environment_relative_current_point);
                    const geometry_msgs::Point current_plus_control_marker_point = EigenHelpersConversions::EigenVector4dToGeometryPoint(environment_relative_current_plus_control_point);
                    configuration_marker.points.push_back(current_marker_point);
                    configuration_marker.points.push_back(current_plus_control_marker_point);
                    configuration_marker.colors.push_back(real_color);
                    configuration_marker.colors.push_back(real_color);
                }
            }
            return configuration_marker;
        }

        /*
         * Nearest-neighbors functions
         */
        inline double StateDistance(const UncertaintyPlanningState& state1, const UncertaintyPlanningState& state2) const
        {
            // Get the "space independent" expectation distance
            const double expectation_distance = robot_.ComputeConfigurationDistance(state1.GetExpectation(), state2.GetExpectation()) / step_size_;
            // Get the Pfeasibility(start -> state1)
            const double feasibility_weight = (1.0 - state1.GetMotionPfeasibility()) * feasibility_alpha_ + (1.0 - feasibility_alpha_);
            // Get the "space independent" variance of state1
            const Eigen::VectorXd raw_variances = state1.GetSpaceIndependentVariances();
            const double raw_variance = raw_variances.lpNorm<1>();
            // Turn the variance into a weight
            const double variance_weight = erf(raw_variance) * variance_alpha_ + (1.0 - variance_alpha_);
            // Compute the actual distance
            const double distance = (feasibility_weight * expectation_distance * variance_weight);
            return distance;
        }

        inline int64_t GetNearestNeighbor(const UncertaintyPlanningTree& planner_nodes, const UncertaintyPlanningState& random_state) const
        {
            UNUSED(planner_nodes);
            // Get the nearest neighbor (ignoring the disabled states)
            std::vector<std::pair<int64_t, double>> per_thread_bests(get_num_omp_threads(), std::pair<int64_t, double>(-1, INFINITY));
            #pragma omp parallel for
            for (size_t idx = 0; idx < nearest_neighbors_storage_.size(); idx++)
            {
                const UncertaintyPlanningTreeState& current_state = nearest_neighbors_storage_[idx];
                // Only check against states enabled for NN checks
                if (current_state.GetValueImmutable().UseForNearestNeighbors())
                {
                    const double state_distance = StateDistance(current_state.GetValueImmutable(), random_state);
#if defined(_OPENMP)
                    const size_t current_thread_id = (size_t)omp_get_thread_num();
#else
                    const size_t current_thread_id = 0;
#endif
                    if (state_distance < per_thread_bests[current_thread_id].second)
                    {
                        per_thread_bests[current_thread_id].first = (int64_t)idx;
                        per_thread_bests[current_thread_id].second = state_distance;
                    }
                }
            }
            int64_t best_index = -1;
            double best_distance = INFINITY;
            for (size_t idx = 0; idx < per_thread_bests.size(); idx++)
            {
                const double& thread_minimum_distance = per_thread_bests[idx].second;
                if (thread_minimum_distance < best_distance)
                {
                    best_index = per_thread_bests[idx].first;
                    best_distance = thread_minimum_distance;
                }
            }
            arc_helpers::ConditionalPrint("Selected node " + std::to_string(best_index) + " as nearest neighbor (Qnear)", 3, debug_level_);
            return best_index;
        }

        /*
         * State sampling wrapper
         */
        inline UncertaintyPlanningState SampleRandomTargetState()
        {
            const Configuration random_point = sampler_ptr_->Sample(rng_);
            arc_helpers::ConditionalPrint("Sampled config: " + PrettyPrint::PrettyPrint(random_point), 3, debug_level_);
            const UncertaintyPlanningState random_state(random_point);
            return random_state;
        }

        /*
         * Particle clustering function used in policy execution
         */
        inline std::vector<std::vector<size_t>> PolicyParticleClusteringFn(const std::vector<Configuration, ConfigAlloc>& particles, const Configuration& current_config, ros::Publisher& display_pub) const
        {
            assert(particles.size() >= 1);
            // Collect all the particles and our current config together
            std::vector<std::pair<Configuration, bool>> all_particles;
            all_particles.reserve(particles.size() + 1);
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                const Configuration& current_particle = particles[idx];
                //std::cout << "Particle configuration " << PrettyPrint::PrettyPrint(current_particle) << " with distance " << robot_.ComputeConfigurationDistance(current_particle, current_config) << std::endl;
                all_particles.push_back(std::make_pair(current_particle, false));
            }
            all_particles.push_back(std::make_pair(current_config, false));
            all_particles.shrink_to_fit();
            // Do clustering magic
            // Perform a first pass of clustering using spatial features
            const std::vector<std::vector<size_t>> initial_clusters = PerformSpatialFeatureClustering(all_particles, display_pub);
            // Now, for each of the initial clusters, we run a second pass of distance-threshold hierarchical clustering
            const std::vector<std::vector<size_t>> final_index_clusters = RunDistanceClusteringOnInitialClusters(initial_clusters, all_particles);
            return final_index_clusters;
        }

        /*
         * Particle clustering functions
         */
        inline std::vector<std::vector<std::vector<uint32_t>>> GenerateRegionSignatures(const std::vector<std::pair<Configuration, bool>>& particles) const
        {
            // Collect the signatures for each particle
            std::vector<std::vector<std::vector<uint32_t>>> particle_region_signatures(particles.size());
            // Loop through all the particles
            #pragma omp parallel for
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                const std::pair<Configuration, bool>& particle = particles[idx];
                particle_region_signatures[idx] = ComputeConfigurationConvexRegionSignature(robot_, particle.first);
            }
            return particle_region_signatures;
        }

        inline std::vector<std::vector<uint32_t>> ComputeConfigurationConvexRegionSignature(const Robot& immutable_robot, const Configuration& configuration) const
        {
            Robot robot = immutable_robot;
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> robot_links_points = robot.GetRawLinksPoints();
            // Update the position of the robot
            robot.UpdatePosition(configuration);
            std::vector<std::vector<uint32_t>> link_region_signatures(robot_links_points.size());
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector4d& link_points = (*robot_links_points[link_idx].second);
                std::vector<uint32_t> link_region_signature(link_points.size());
                // Get the transform of the current link
                const Eigen::Isometry3d link_transform = robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector4d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector4d environment_relative_point = link_transform * link_relative_point;
                    std::pair<const sdf_tools::TAGGED_OBJECT_COLLISION_CELL&, bool> query = simulator_ptr_->GetEnvironment().GetImmutable4d(environment_relative_point);
                    if (query.second)
                    {
                        const sdf_tools::TAGGED_OBJECT_COLLISION_CELL& environment_cell = query.first;
                        link_region_signature[point_idx] = environment_cell.convex_segment;
                    }
                    else
                    {
                        link_region_signature[point_idx] = 0u;
                        //std::cerr << "WARNING - ROBOT POINT OUTSIDE ENVIRONMENT BOUNDS" << std::endl;
                    }
                }
                link_region_signatures[link_idx] = link_region_signature;
            }
            return link_region_signatures;
        }

        inline double ComputeConvexRegionSignatureDistance(const std::vector<std::vector<uint32_t>>& signature_1, const std::vector<std::vector<uint32_t>>& signature_2) const
        {
            assert(signature_1.size() == signature_2.size());
            size_t total_points = 0u;
            size_t non_matching_points = 0u;
            for (size_t link_idx = 0; link_idx < signature_1.size(); link_idx++)
            {
                const std::vector<uint32_t>& signature_1_link = signature_1[link_idx];
                const std::vector<uint32_t>& signature_2_link = signature_2[link_idx];
                assert(signature_1_link.size() == signature_2_link.size());
                for (size_t point_idx = 0; point_idx < signature_1_link.size(); point_idx++)
                {
                    total_points++;
                    const uint32_t signature_1_point = signature_1_link[point_idx];
                    const uint32_t signature_2_point = signature_2_link[point_idx];
                    if ((signature_1_point != 0u) && (signature_2_point != 0u) && ((signature_1_point & signature_2_point) == 0u))
                    {
                        non_matching_points++;
                    }
                }
            }
            assert(total_points > 0u);
            const double distance = (double)non_matching_points / (double)total_points;
            return distance;
        }

        inline std::vector<std::vector<size_t>> GenerateRegionSignatureClusters(const std::vector<std::vector<std::vector<uint32_t>>>& particle_region_signatures) const
        {
            // Generate an initial "cluster" with everything in it
            std::vector<size_t> initial_cluster(particle_region_signatures.size());
            for (size_t idx = 0; idx < initial_cluster.size(); idx++)
            {
                initial_cluster[idx] = idx;
            }
            // Let's build the distance function
            // This is a little special - we use the lambda to capture the local context, so we can pass indices to the clustering instead of the actual configurations, but have the clustering *operate* over configurations
            std::function<double(const size_t&, const size_t&)> distance_fn = [&] (const size_t& idx1, const size_t& idx2) { return ComputeConvexRegionSignatureDistance(particle_region_signatures[idx1], particle_region_signatures[idx2]); };
            const Eigen::MatrixXd distance_matrix = arc_helpers::BuildDistanceMatrix(initial_cluster, distance_fn);
            //std::cout << "Region signature distance matrix " << PrettyPrint::PrettyPrint(distance_matrix) << std::endl;
            // Check the max element of the distance matrix
            const double max_distance = distance_matrix.maxCoeff();
            if (max_distance <= signature_matching_threshold_)
            {
                return std::vector<std::vector<size_t>>{initial_cluster};
            }
            else
            {
                return simple_hierarchical_clustering::SimpleHierarchicalClustering::Cluster(initial_cluster, distance_matrix, signature_matching_threshold_).first;
            }
        }

        inline std::vector<std::vector<size_t>> PerformConvexRegionSignatureClustering(const std::vector<std::pair<Configuration, bool>>& particles, ros::Publisher& display_pub) const
        {
            UNUSED(display_pub);
            // Collect the signatures for each particle
            const std::vector<std::vector<std::vector<uint32_t>>> particle_region_signatures = GenerateRegionSignatures(particles);
            // Cluster
            const std::vector<std::vector<size_t>> initial_clusters = GenerateRegionSignatureClusters(particle_region_signatures);
            return initial_clusters;
        }

        inline bool CheckStraightLine3DPath(const Eigen::Vector3d& start, const Eigen::Vector3d& end) const
        {
            // Get a reference to the environment grid
            const sdf_tools::TaggedObjectCollisionMapGrid& environment = simulator_ptr_->GetEnvironment();
            const double simulator_resolution = environment.GetResolution();
            const double distance = (end - start).norm();
            const uint32_t steps = std::max(1u, (uint32_t)ceil(distance / simulator_resolution));
            for (uint32_t step = 1u; step <= steps; step++)
            {
                const double percent = (double)step / (double)steps;
                const Eigen::Vector3d intermediate_point = EigenHelpers::Interpolate(start, end, percent);
                const float occupancy = environment.GetImmutable3d(intermediate_point).first.occupancy;
                if (occupancy > 0.5f)
                {
                    return false;
                }
            }
            return true;
        }

        inline bool CheckActuationCenterStraightLinePath(const Robot& immutable_robot, const Configuration& start, const Configuration& end) const
        {
            Robot robot = immutable_robot;
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> robot_links_points = robot.GetRawLinksPoints();
            EigenHelpers::VectorVector3d start_config_actuation_centers(robot_links_points.size());
            EigenHelpers::VectorVector3d end_config_actuation_centers(robot_links_points.size());
            // Update the position of the robot
            robot.UpdatePosition(start);
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                // Get the transform of the current link
                const Eigen::Isometry3d link_transform = robot.GetLinkTransform(link_name);
                const Eigen::Vector3d link_point = link_transform.translation();
                start_config_actuation_centers[link_idx] = link_point;
            }
            // Update the position of the robot
            robot.UpdatePosition(end);
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                // Get the transform of the current link
                const Eigen::Isometry3d link_transform = robot.GetLinkTransform(link_name);
                const Eigen::Vector3d link_point = link_transform.translation();
                end_config_actuation_centers[link_idx] = link_point;
            }
            // Now that we've collected the actuation centers, check the straight-line paths between them
            bool actuation_centers_feasible = true;
            for (size_t idx = 0; idx < robot_links_points.size(); idx++)
            {
                const Eigen::Vector3d& start_link_actuation_center = start_config_actuation_centers[idx];
                const Eigen::Vector3d& end_link_actuation_center = end_config_actuation_centers[idx];
                if (CheckStraightLine3DPath(start_link_actuation_center, end_link_actuation_center) == false)
                {
                    actuation_centers_feasible = false;
                    break;
                }
            }
            return actuation_centers_feasible;
        }

        inline std::vector<std::vector<size_t>> PerformActuationCenterClustering(const std::vector<std::pair<Configuration, bool>>& particles, ros::Publisher& display_pub) const
        {
            UNUSED(display_pub);
            assert(particles.size() > 1);
            // Assemble the actuation center to actuation center feasibility matrix
            Eigen::MatrixXd distance_matrix = Eigen::MatrixXd::Zero((ssize_t)(particles.size()), (ssize_t)(particles.size()));
            #pragma omp parallel for
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                for (size_t jdx = idx; jdx < particles.size(); jdx++)
                {
                    if (idx != jdx)
                    {
                        const double distance = CheckActuationCenterStraightLinePath(robot_, particles[idx].first, particles[jdx].first) ? 0.0 : 1.0;
                        distance_matrix((ssize_t)idx, (ssize_t)jdx) = distance;
                        distance_matrix((ssize_t)jdx, (ssize_t)idx) = distance;
                    }
                }
            }
            // Generate an initial "cluster" with everything in it
            std::vector<size_t> initial_cluster(particles.size());
            for (size_t idx = 0; idx < initial_cluster.size(); idx++)
            {
                initial_cluster[idx] = idx;
            }
            // Check the max element of the distance matrix
            const double max_distance = distance_matrix.maxCoeff();
            if (max_distance <= 0.0)
            {
                return std::vector<std::vector<size_t>>{initial_cluster};
            }
            else
            {
                return simple_hierarchical_clustering::SimpleHierarchicalClustering::Cluster(initial_cluster, distance_matrix, 0.0).first;
            }
        }

        inline std::vector<std::vector<size_t>> PerformPointToPointMovementClustering(const std::vector<std::pair<Configuration, bool>>& particles, ros::Publisher& display_pub) const
        {
            assert(particles.size() > 1);
            const size_t num_simulations = (particles.size() * particles.size()) - particles.size();
            // Assemble the point to point movement feasibility matrix
            Eigen::MatrixXd distance_matrix = Eigen::MatrixXd::Zero((ssize_t)(particles.size()), (ssize_t)(particles.size()));
            std::vector<Configuration, ConfigAlloc> initial_positions;
            initial_positions.reserve(num_simulations);
            std::vector<Configuration, ConfigAlloc> target_positions;
            target_positions.reserve(num_simulations);
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                for (size_t jdx = idx; jdx < particles.size(); jdx++)
                {
                    if (idx != jdx)
                    {
                        const Configuration& idx_position = particles[idx].first;
                        const Configuration& jdx_position = particles[jdx].first;
                        initial_positions.push_back(idx_position);
                        target_positions.push_back(jdx_position);
                        initial_positions.push_back(jdx_position);
                        target_positions.push_back(idx_position);
                    }
                }
            }
            const std::vector<std::pair<Configuration, bool>> result_configs = simulator_ptr_->ForwardSimulateRobots(robot_, initial_positions, target_positions, rngs_, step_duration_, (goal_distance_threshold_ * 0.5), simulate_with_individual_jacobians_, true, display_pub);
            size_t current = 0;
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                for (size_t jdx = idx; jdx < particles.size(); jdx++)
                {
                    if (idx != jdx)
                    {
                        const Configuration& forward_result = result_configs[current + 0].first;
                        const Configuration& backward_result = result_configs[current + 1].first;
                        const Configuration& forward_target = target_positions[current + 0];
                        const Configuration& backward_target = target_positions[current + 1];
                        const bool forward_feasible = (robot_.ComputeConfigurationDistance(forward_result, forward_target) <= goal_distance_threshold_) ? true : false;
                        const bool backward_feasible = (robot_.ComputeConfigurationDistance(backward_result, backward_target) <= goal_distance_threshold_) ? true : false;
                        const double distance = (forward_feasible && backward_feasible) ? 0.0 : 1.0;
                        distance_matrix((ssize_t)idx, (ssize_t)jdx) = distance;
                        distance_matrix((ssize_t)jdx, (ssize_t)idx) = distance;
                        current += 2;
                    }
                }
            }
            // Generate an initial "cluster" with everything in it
            std::vector<size_t> initial_cluster(particles.size());
            for (size_t idx = 0; idx < initial_cluster.size(); idx++)
            {
                initial_cluster[idx] = idx;
            }
            // Check the max element of the distance matrix
            const double max_distance = distance_matrix.maxCoeff();
            if (max_distance <= 0.0)
            {
                return std::vector<std::vector<size_t>>{initial_cluster};
            }
            else
            {
                return simple_hierarchical_clustering::SimpleHierarchicalClustering::Cluster(initial_cluster, distance_matrix, 0.0).first;
            }
        }

        inline double ComputeClusterSimilarity(const std::vector<size_t>& cluster1, const std::vector<size_t>& cluster2) const
        {
            // Compute the Jaccard Similarity Coefficient (# of elements in intersection/# of elements in union)
            // First, insert cluster1 into a map for fast lookups
            std::map<size_t, uint8_t> cluster1_map;
            for (size_t idx = 0; idx < cluster1.size(); idx++)
            {
                const size_t element = cluster1[idx];
                cluster1_map[element] = 0x01;
            }
            // Now, go through cluster 2 and keep track of which are/aren't in cluster1
            size_t in_intersection = 0.0;
            size_t not_in_cluster1 = 0.0;
            for (size_t idx = 0; idx < cluster2.size(); idx++)
            {
                const size_t element = cluster2[idx];
                if (cluster1_map.find(element) != cluster1_map.end())
                {
                    in_intersection++;
                }
                else
                {
                    not_in_cluster1++;
                }
            }
            const size_t in_union = cluster1.size() + not_in_cluster1;
            const double similarity = (double)in_intersection / (double)in_union;
            return similarity;
        }

        inline double ComputeClusteringSimilarity(const std::vector<std::vector<size_t>>& clustering1, const std::vector<std::vector<size_t>>& clustering2) const
        {
            // Implementation of "A Similarity Measure for Clustering and Its Applications" Torres et al
            // Compute the clustering similarity matrix
            Eigen::MatrixXd similarity_matrix = Eigen::MatrixXd::Zero((ssize_t)(clustering1.size()), (ssize_t)(clustering2.size()));
            for (size_t idx = 0; idx < clustering1.size(); idx++)
            {
                for (size_t jdx = 0; jdx < clustering2.size(); jdx++)
                {
                    const std::vector<size_t>& cluster1 = clustering1[idx];
                    const std::vector<size_t>& cluster2 = clustering2[jdx];
                    const double cluster_similarity = ComputeClusterSimilarity(cluster1, cluster2);
                    similarity_matrix((ssize_t)idx, (ssize_t)jdx) = cluster_similarity;
                }
            }
            // Compute the similarity metric from the matrix
            const double similarity = similarity_matrix.sum() / (double)std::max(clustering1.size(), clustering2.size());
            return (similarity * similarity);
        }

        inline std::vector<std::vector<size_t>> PerformSpatialFeatureClustering(const std::vector<std::pair<Configuration, bool>>& particles, ros::Publisher& display_pub) const
        {
            // We have three different options for spatial feature clustering
            if (spatial_feature_clustering_type_ == CONVEX_REGION_SIGNATURE)
            {
                return PerformConvexRegionSignatureClustering(particles, display_pub);
            }
            else if (spatial_feature_clustering_type_ == ACTUATION_CENTER_CONNECTIVITY)
            {
                return PerformActuationCenterClustering(particles, display_pub);
            }
            else if (spatial_feature_clustering_type_ == POINT_TO_POINT_MOVEMENT)
            {
                return PerformPointToPointMovementClustering(particles, display_pub);
            }
            else if (spatial_feature_clustering_type_ == COMPARE)
            {
                const std::vector<std::vector<size_t>> crs_clustering = PerformConvexRegionSignatureClustering(particles, display_pub);
                const std::vector<std::vector<size_t>> ac_clustering = PerformActuationCenterClustering(particles, display_pub);
                const std::vector<std::vector<size_t>> ptpm_clustering = PerformPointToPointMovementClustering(particles, display_pub);
                std::cout << "++++++++++++++++++++++++++++++++++++++++\nClustering results:\nCRS: " << PrettyPrint::PrettyPrint(crs_clustering, true) <<  "\nAC: " << PrettyPrint::PrettyPrint(ac_clustering, true) << "\nPTPM: " << PrettyPrint::PrettyPrint(ptpm_clustering, true) << std::endl;
                clustering_performance_.clustering_splits.push_back((uint32_t)ptpm_clustering.size());
                clustering_performance_.crs_similarities.push_back(ComputeClusteringSimilarity(ptpm_clustering, crs_clustering));
                clustering_performance_.ac_similarities.push_back(ComputeClusteringSimilarity(ptpm_clustering, ac_clustering));
                return ptpm_clustering;
            }
            else
            {
                assert(false);
            }
        }

        inline std::vector<std::vector<size_t>> RunDistanceClusteringOnInitialClusters(const std::vector<std::vector<size_t>>& initial_clusters, const std::vector<std::pair<Configuration, bool>>& particles) const
        {
            const double distance_threshold = distance_clustering_threshold_;
            // Now, for each of the initial clusters, we run a second pass of distance-threshold hierarchical clustering
            std::vector<std::vector<size_t>> intermediate_clusters;
            intermediate_clusters.reserve(initial_clusters.size());
            // Let's build the distance function
            // This is a little special - we use the lambda to capture the local context, so we can pass indices to the clustering instead of the actual configurations, but have the clustering *operate* over configurations
            std::function<double(const size_t&, const size_t&)> distance_fn = [&] (const size_t& idx1, const size_t& idx2) { return robot_.ComputeConfigurationDistance(particles[idx1].first, particles[idx2].first); };
            for (size_t cluster_idx = 0; cluster_idx < initial_clusters.size(); cluster_idx++)
            {
                const std::vector<size_t>& current_cluster = initial_clusters[cluster_idx];
                // First, we build the distance matrix
                const Eigen::MatrixXd distance_matrix = arc_helpers::BuildDistanceMatrix(current_cluster, distance_fn);
                // Check the max element of the distance matrix
                const double max_distance = distance_matrix.maxCoeff();
                //std::cout << "Distance matrix:\n" << PrettyPrint::PrettyPrint(distance_matrix) << std::endl;
                if (max_distance <= distance_threshold)
                {
                    std::cout << "Cluster by convex region of " << current_cluster.size() << " elements has max distance " << max_distance << " below distance threshold " << distance_threshold << ", not performing additional hierarchical clustering" << std::endl;
                    intermediate_clusters.push_back(current_cluster);
                }
                else
                {
                    //std::cout << "Distance matrix:\n" << PrettyPrint::PrettyPrint(distance_matrix) << std::endl;
                    cluster_fallback_calls_++;
                    std::cout << "Cluster by convex region of " << current_cluster.size() << " elements has max distance " << max_distance << " exceeding distance threshold " << distance_threshold << ", performing additional hierarchical clustering" << std::endl;
                    const std::vector<std::vector<size_t>> new_clustering = simple_hierarchical_clustering::SimpleHierarchicalClustering::Cluster(current_cluster, distance_matrix, distance_threshold).first;
                    std::cout << "Additional hierarchical clustering produced " << new_clustering.size() << " clusters" << std::endl;
                    for (size_t ndx = 0; ndx < new_clustering.size(); ndx++)
                    {
                        //std::cout << "Resulting cluster " << ndx << ":\n" << PrettyPrint::PrettyPrint(new_clustering[ndx]) << std::endl;
                        intermediate_clusters.push_back(new_clustering[ndx]);
                    }
                }
            }
            return intermediate_clusters;
        }

        inline std::vector<std::vector<std::pair<Configuration, bool>>> ClusterParticles(const std::vector<std::pair<Configuration, bool>>& particles, const bool allow_contacts, ros::Publisher& display_pub) const
        {
            // Make sure there are particles to cluster
            if (particles.size() == 0)
            {
                return std::vector<std::vector<std::pair<Configuration, bool>>>();
            }
            else if (particles.size() == 1)
            {
                return std::vector<std::vector<std::pair<Configuration, bool>>>{particles};
            }
            const std::chrono::time_point<std::chrono::high_resolution_clock> start = (std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now();
            cluster_calls_++;
            // Perform a first pass of clustering using spatial features
            const std::vector<std::vector<size_t>> initial_clusters = PerformSpatialFeatureClustering(particles, display_pub);
            //std::cout << "Initial clustering: " << PrettyPrint::PrettyPrint(initial_clusters, true) << std::endl;
            // Now, for each of the initial clusters, we run a second pass of distance-threshold hierarchical clustering
            const std::vector<std::vector<size_t>> final_index_clusters = RunDistanceClusteringOnInitialClusters(initial_clusters, particles);
            // Before we return, we need to convert the index clusters to configuration clusters
            std::vector<std::vector<std::pair<Configuration, bool>>> final_clusters;
            final_clusters.reserve(final_index_clusters.size());
            size_t total_particles = 0;
            for (size_t cluster_idx = 0; cluster_idx < final_index_clusters.size(); cluster_idx++)
            {
                const std::vector<size_t>& cluster = final_index_clusters[cluster_idx];
                std::vector<std::pair<Configuration, bool>> final_cluster;
                final_cluster.reserve(cluster.size());
                for (size_t element_idx = 0; element_idx < cluster.size(); element_idx++)
                {
                    total_particles++;
                    const size_t particle_idx = cluster[element_idx];
                    assert(particle_idx < particles.size());
                    const std::pair<Configuration, bool>& particle = particles[particle_idx];
                    if ((particle.second == false) || allow_contacts)
                    {
                        final_cluster.push_back(particle);
                    }
                }
                final_cluster.shrink_to_fit();
                final_clusters.push_back(final_cluster);
            }
            final_clusters.shrink_to_fit();
            assert(total_particles == particles.size());
            // Now, return the clusters and probability table
            const std::chrono::time_point<std::chrono::high_resolution_clock> end = (std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> elapsed = end - start;
            elapsed_clustering_time_ += elapsed.count();
            return final_clusters;
        }

        /*
         * Forward propagation functions
         */
        inline std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>> ForwardSimulateParticles(const UncertaintyPlanningState& nearest, const UncertaintyPlanningState& target, const double step_duration, const bool allow_contacts, ros::Publisher& display_pub) const
        {
            const std::chrono::time_point<std::chrono::high_resolution_clock> start = (std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now();
            // First, compute a target state
            const Configuration target_point = target.GetExpectation();
            // Get the initial particles
            std::vector<Configuration, ConfigAlloc> initial_particles;
            // We'd like to use the particles of the parent directly
            if (nearest.GetNumParticles() == num_particles_)
            {
                initial_particles = nearest.CollectParticles(num_particles_);
            }
            // Otherwise, we resample from the parent
            else
            {
                initial_particles = nearest.ResampleParticles(num_particles_, rng_);
            }
            if (debug_level_ >= 15)
            {
                display_pub.publish(DrawParticles(robot_, initial_particles, MakeColor(0.1f, 0.1f, 0.1f, 1.0f), "initial_particles"));
            }
            // Forward propagate each of the particles
            std::vector<Configuration, ConfigAlloc> target_position;
            target_position.reserve(1);
            target_position.push_back(target_point);
            target_position.shrink_to_fit();
            const std::vector<std::pair<Configuration, bool>> propagated_points = simulator_ptr_->ForwardSimulateRobots(robot_, initial_particles, target_position, rngs_, step_duration, (goal_distance_threshold_ * 0.5), simulate_with_individual_jacobians_, allow_contacts, display_pub);
            particles_simulated_ += num_particles_;
            const std::chrono::time_point<std::chrono::high_resolution_clock> end = (std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> elapsed = end - start;
            elapsed_simulation_time_ += elapsed.count();
            return std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>(initial_particles, propagated_points);
        }

        inline std::pair<uint32_t, uint32_t> ComputeReverseEdgeProbability(const UncertaintyPlanningState& parent, const UncertaintyPlanningState& child, ros::Publisher& display_pub) const
        {
            std::vector<std::pair<Configuration, bool>> simulation_result = ForwardSimulateParticles(child, parent, step_duration_, true, display_pub).second;
            uint32_t reached_parent = 0u;
            // Get the target position
            const Configuration target_position = parent.GetExpectation();
            for (size_t ndx = 0; ndx < simulation_result.size(); ndx++)
            {
                const Configuration& current_particle = simulation_result[ndx].first;
                // Check if the particle got close enough
                const double particle_distance = robot_.ComputeConfigurationDistance(current_particle, target_position);
                if (particle_distance <= goal_distance_threshold_)
                {
                    reached_parent++;
                }
            }
            return std::make_pair(num_particles_, reached_parent);
        }

        inline std::pair<std::vector<std::pair<UncertaintyPlanningState, int64_t>>, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> ForwardSimulateStates(const UncertaintyPlanningState& nearest, const UncertaintyPlanningState& target, const double step_duration, const uint32_t planner_action_try_attempts, const bool allow_contacts, const bool include_reverse_actions, ros::Publisher& display_pub) const
        {
            // Increment the transition ID
            transition_id_++;
            const uint64_t current_forward_transition_id = transition_id_;
            const Configuration control_input = target.GetExpectation();
            // Forward propagate each of the particles
            std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>> simulation_result = ForwardSimulateParticles(nearest, target, step_duration, allow_contacts, display_pub);
            std::vector<Configuration, ConfigAlloc>& initial_particles = simulation_result.first;
            std::vector<std::pair<Configuration, bool>>& propagated_points = simulation_result.second;
            // Cluster the live particles into (potentially) multiple states
            const std::vector<std::vector<std::pair<Configuration, bool>>>& particle_clusters = ClusterParticles(propagated_points, allow_contacts, display_pub);
            bool is_split_child = false;
            if (particle_clusters.size() > 1)
            {
                //std::cout << "Transition produced " << particle_clusters.size() << " split states" << std::endl;
                is_split_child = true;
                split_id_++;
            }
            // Build the forward-propagated states
            std::vector<std::pair<UncertaintyPlanningState, int64_t>> result_states(particle_clusters.size());
            for (size_t idx = 0; idx < particle_clusters.size(); idx++)
            {
                const std::vector<std::pair<Configuration, bool>>& current_cluster = particle_clusters[idx];
                if (debug_level_ >= 15)
                {
                    display_pub.publish(DrawParticles(robot_, current_cluster, arc_helpers::GenerateUniqueColor<std_msgs::ColorRGBA>((uint32_t)(idx + 1), 1.0f), "result_cluster_" + std::to_string(idx + 1)));
                }
                if (particle_clusters[idx].size() > 0)
                {
                    state_counter_++;
                    const uint32_t attempt_count = (uint32_t)num_particles_;
                    const uint32_t reached_count = (uint32_t)current_cluster.size();
                    // Check if any of the particles in the current cluster collided with the environment during simulation.
                    // If all are collision-free, we can safely assume the edge is trivially reversible
                    std::vector<Configuration, ConfigAlloc> particle_locations(current_cluster.size());
                    bool did_collide = false;
                    for (size_t pdx = 0; pdx < current_cluster.size(); pdx++)
                    {
                        particle_locations[pdx] = current_cluster[pdx].first;
                        if (current_cluster[pdx].second)
                        {
                            did_collide = true;
                        }
                    }
                    particles_stored_ += particle_locations.size();
                    uint32_t reverse_attempt_count = (uint32_t)num_particles_;
                    uint32_t reverse_reached_count = (uint32_t)num_particles_;
                    // Don't do extra work with one particle
                    if (did_collide && (num_particles_ > 1))
                    {
                        //std::cout << "Simulation resulted in collision, defering reversibility check to post-processing" << std::endl;
                        reverse_attempt_count = (uint32_t)num_particles_;
                        reverse_reached_count = 0u;
                    }
                    else if (is_split_child)
                    {
                        //std::cout << "Simulation resulted in split, defering reversibility check to post-processing" << std::endl;
                        reverse_attempt_count = (uint32_t)num_particles_;
                        reverse_reached_count = 0u;
                    }
                    const double effective_edge_feasibility = (double)reached_count / (double)attempt_count;
                    transition_id_++;
                    const uint64_t new_state_reverse_transtion_id = transition_id_;
                    //uncertainty_planning_tools::UncertaintyPlannerState<Configuration, ConfigSerializer, Robot, ConfigAlloc> propagated_state(state_counter_, particle_locations, robot_, attempt_count, reached_count, effective_edge_feasibility, reverse_attempt_count, reverse_reached_count, nearest.GetMotionPfeasibility(), step_size_, control_input, current_forward_transition_id, new_state_reverse_transtion_id, ((is_split_child) ? split_id_ : 0u));
                    UncertaintyPlanningState propagated_state(state_counter_, particle_locations, attempt_count, reached_count, effective_edge_feasibility, reverse_attempt_count, reverse_reached_count, nearest.GetMotionPfeasibility(), step_size_, control_input, current_forward_transition_id, new_state_reverse_transtion_id, ((is_split_child) ? split_id_ : 0u));
                    propagated_state.UpdateStatistics(robot_);
                    // Store the state
                    result_states[idx].first = propagated_state;
                    result_states[idx].second = -1;
                }
            }
            // Now that we've built the forward-propagated states, we compute their reverse edge P(feasibility)
            uint32_t computed_reversibility = 0u;
            for (size_t idx = 0; idx < result_states.size(); idx++)
            {
                UncertaintyPlanningState& current_state = result_states[idx].first;
                if (include_reverse_actions)
                {
                    // In some cases, we already know the reverse edge P(feasibility) so we don't need to compute it again
                    if (current_state.GetReverseEdgePfeasibility() < 1.0)
                    {
                        const std::pair<uint32_t, uint32_t> reverse_edge_check = ComputeReverseEdgeProbability(nearest, current_state, display_pub);
                        current_state.UpdateReverseAttemptAndReachedCounts(reverse_edge_check.first, reverse_edge_check.second);
                        computed_reversibility++;
                    }
                }
                else
                {
                    current_state.UpdateReverseAttemptAndReachedCounts((uint32_t)num_particles_, 0u);
                }
            }
            arc_helpers::ConditionalPrint("Forward simultation produced " + std::to_string(result_states.size()) + " states, needed to compute reversibility for " + std::to_string(computed_reversibility) + " of them", 3, debug_level_);
            // We only do further processing if a split happened
            if (result_states.size() > 1)
            {
                // Now that we have the forward-propagated states, we go back and update their effective edge P(feasibility)
                for (size_t idx = 0; idx < result_states.size(); idx++)
                {
                    UncertaintyPlanningState& current_state = result_states[idx].first;
                    double percent_active = 1.0;
                    double p_reached = 0.0;
                    for (uint32_t try_attempt = 0; try_attempt < planner_action_try_attempts; try_attempt++)
                    {
                        // How many particles got to our state on this attempt?
                        p_reached += (percent_active * current_state.GetRawEdgePfeasibility());
                        // Update the percent of particles that are still usefully active
                        double updated_percent_active = 0.0;
                        for (size_t other_idx = 0; other_idx < result_states.size(); other_idx++)
                        {
                            if (other_idx != idx)
                            {
                                const UncertaintyPlanningState& other_state = result_states[other_idx].first;
                                const double p_reached_other = percent_active * other_state.GetRawEdgePfeasibility();
                                const double p_returned_to_parent = p_reached_other * other_state.GetReverseEdgePfeasibility();
                                updated_percent_active += p_returned_to_parent;
                            }
                        }
                        percent_active = updated_percent_active;
                    }
                    assert(p_reached > 0.0);
                    if (p_reached > 1.0)
                    {
                        assert(p_reached <= 1.001);
                        p_reached = 1.0;
                    }
                    assert(p_reached <= 1.0);
                    arc_helpers::ConditionalPrint("Computed effective edge P(feasibility) of " + std::to_string(p_reached) + " for " + std::to_string(planner_action_try_attempts) + " try/retry attempts", 4, debug_level_);
                    current_state.SetEffectiveEdgePfeasibility(p_reached);
                }
            }
            if (debug_level_ >= 30)
            {
                std::cout << "Press ENTER to add new states..." << std::endl;
                std::cin.get();
            }
            return std::pair<std::vector<std::pair<UncertaintyPlanningState, int64_t>>, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>(result_states, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>(initial_particles, propagated_points));
        }

        inline std::vector<std::pair<UncertaintyPlanningState, int64_t>> PropagateForwardsAndDraw(const UncertaintyPlanningState& nearest, const UncertaintyPlanningState& random, const uint32_t planner_action_try_attempts, const bool allow_contacts, const bool include_reverse_actions, ros::Publisher& display_pub)
        {
            // First, perform the forwards propagation
            const std::pair<std::vector<std::pair<UncertaintyPlanningState, int64_t>>, std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>> propagated_state = PerformForwardPropagation(nearest, random, planner_action_try_attempts, allow_contacts, include_reverse_actions, display_pub);
            if (debug_level_ >= 1)
            {
                // Draw the expansion
                visualization_msgs::MarkerArray propagation_display_rep;
                // Check if the expansion was useful
                if (propagated_state.first.size() > 0)
                {
                    for (size_t idx = 0; idx < propagated_state.first.size(); idx++)
                    {
                        int64_t state_index = (int64_t)state_counter_ + ((int64_t)idx - ((int64_t)propagated_state.first.size() - 1));
                        // Yeah, sorry about the ternary. This is so we can still have a const reference
                        //const UncertaintyPlanningState& previous_state = (propagated_state.first[idx].second >= 0) ? propagated_state.first[propagated_state.first[idx].second].first : nearest;
                        const UncertaintyPlanningState& current_state = propagated_state.first[idx].first;
                        // Get the edge feasibility
                        const double edge_Pfeasibility = current_state.GetEffectiveEdgePfeasibility();
                        // Get motion feasibility
                        const double motion_Pfeasibility = current_state.GetMotionPfeasibility();
                        // Get the variance
                        const double raw_variance = current_state.GetSpaceIndependentVariance();
                        // Get the reverse feasibility
                        const double reverse_edge_Pfeasibility = current_state.GetReverseEdgePfeasibility();
                        // Now we get markers corresponding to the current states
                        // Make the display color
                        std_msgs::ColorRGBA forward_color;
                        forward_color.r = (float)(1.0 - motion_Pfeasibility);
                        forward_color.g = (float)(1.0 - motion_Pfeasibility);
                        forward_color.b = (float)(1.0 - motion_Pfeasibility);
                        forward_color.a = 1.0f - ((float)(erf(raw_variance) * variance_alpha_));
                        visualization_msgs::Marker forward_expectation_marker = DrawRobotConfiguration(robot_, current_state.GetExpectation(), forward_color);
                        forward_expectation_marker.id = (int)state_index;
                        if (edge_Pfeasibility == 1.0f)
                        {
                            forward_expectation_marker.ns = "forward_expectation";
                        }
                        else
                        {
                            forward_expectation_marker.ns = "split_forward_expectation";
                        }
                        propagation_display_rep.markers.push_back(forward_expectation_marker);
                        if (reverse_edge_Pfeasibility > 0.5)
                        {
                            // Make the display color
                            std_msgs::ColorRGBA reverse_color;
                            reverse_color.r = (float)(1.0 - motion_Pfeasibility);
                            reverse_color.g = (float)(1.0 - motion_Pfeasibility);
                            reverse_color.b = (float)(1.0 - motion_Pfeasibility);
                            reverse_color.a = (float)reverse_edge_Pfeasibility;
                            visualization_msgs::Marker reverse_expectation_marker = DrawRobotConfiguration(robot_, current_state.GetExpectation(), reverse_color);
                            reverse_expectation_marker.id = (int)state_index;
                            if (edge_Pfeasibility == 1.0)
                            {
                                reverse_expectation_marker.ns = "reverse_expectation";
                            }
                            else
                            {
                                reverse_expectation_marker.ns = "split_reverse_expectation";
                            }
                            propagation_display_rep.markers.push_back(reverse_expectation_marker);
                        }
                    }
                }
                display_pub.publish(propagation_display_rep);
            }
            return propagated_state.first;
        }

        inline std::pair<std::vector<std::pair<UncertaintyPlanningState, int64_t>>, std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>> PerformForwardPropagation(const UncertaintyPlanningState& nearest, const UncertaintyPlanningState& random, const uint32_t planner_action_try_attempts, const bool allow_contacts, const bool include_reverse_actions, ros::Publisher& display_debug_publisher)
        {
            const bool solution_already_found = (total_goal_reached_probability_ >= goal_probability_threshold_);
            bool use_extend = false;
            if (solution_already_found)
            {
                std::uniform_real_distribution<double> temp_dist(0.0, 1.0);
                const double draw = temp_dist(rng_);
                if (draw < connect_after_first_solution_)
                {
                    use_extend = false;
                }
                else
                {
                    use_extend = true;
                }
            }
            // First, check if we're going to use RRT-Connect or RRT-Extend
            // If we've already found a solution, we use RRT-Extend
            if (use_extend)
            {
                // Compute a single target state
                Configuration target_point = random.GetExpectation();
                const double target_distance = robot_.ComputeConfigurationDistance(nearest.GetExpectation(), target_point);
                if (target_distance > step_size_)
                {
                    const double step_fraction = step_size_ / target_distance;
                    arc_helpers::ConditionalPrint("Forward simulating for " + std::to_string(step_fraction) + " step fraction, step size is " + std::to_string(step_size_) + ", target distance is " + std::to_string(target_distance), 3, debug_level_);
                    const Configuration interpolated_target_point = robot_.InterpolateBetweenConfigurations(nearest.GetExpectation(), target_point, step_fraction);
                    target_point = interpolated_target_point;
                }
                else
                {
                    arc_helpers::ConditionalPrint("Forward simulating, step size is " + std::to_string(step_size_) + ", target distance is " + std::to_string(target_distance), 3, debug_level_);
                }
                UncertaintyPlanningState target_state(target_point);
                std::pair<std::vector<std::pair<UncertaintyPlanningState, int64_t>>, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> propagation_results = ForwardSimulateStates(nearest, target_state, step_duration_, planner_action_try_attempts, allow_contacts, include_reverse_actions, display_debug_publisher);
                std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> raw_particle_propagations = {propagation_results.second};
                return std::pair<std::vector<std::pair<UncertaintyPlanningState, int64_t>>, std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>>(propagation_results.first, raw_particle_propagations);
            }
            // If we haven't found a solution yet, we use RRT-Connect
            else
            {
                std::vector<std::pair<UncertaintyPlanningState, int64_t>> propagated_states;
                std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> raw_particle_propagations;
                int64_t parent_offset = -1;
                // Compute a maximum number of steps to take
                const Configuration target_point = random.GetExpectation();
                const uint32_t total_steps = (uint32_t)ceil(robot_.ComputeConfigurationDistance(nearest.GetExpectation(), target_point) / step_size_);
                UncertaintyPlanningState current = nearest;
                uint32_t steps = 0;
                bool completed = false;
                while ((completed == false) && (steps < total_steps))
                {
                    // Compute a single target state
                    Configuration current_target_point = target_point;
                    const double target_distance = robot_.ComputeConfigurationDistance(current.GetExpectation(), current_target_point);
                    if (target_distance > step_size_)
                    {
                        const double step_fraction = step_size_ / target_distance;
                        const Configuration interpolated_target_point = robot_.InterpolateBetweenConfigurations(current.GetExpectation(), target_point, step_fraction);
                        current_target_point = interpolated_target_point;
                        arc_helpers::ConditionalPrint("Forward simulating for " + std::to_string(step_fraction) + " step fraction, step size is " + std::to_string(step_size_) + ", target distance is " + std::to_string(target_distance), 3, debug_level_);
                    }
                    // If we've reached the target state, stop
                    else if (std::abs(target_distance) < std::numeric_limits<double>::epsilon())
                    {
                        completed = true;
                        break;
                    }
                    // If we're less than step size away, this is our last step
                    else
                    {
                        arc_helpers::ConditionalPrint("Forward simulating last step towars target, step size is " + std::to_string(step_size_) + ", target distance is " + std::to_string(target_distance), 3, debug_level_);
                        completed = true;
                    }
                    // Take a step forwards
                    UncertaintyPlanningState target_state(current_target_point);
                    std::pair<std::vector<std::pair<UncertaintyPlanningState, int64_t>>, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> propagation_results = ForwardSimulateStates(nearest, target_state, step_duration_, planner_action_try_attempts, allow_contacts, include_reverse_actions, display_debug_publisher);
                    raw_particle_propagations.push_back(propagation_results.second);
                    const std::vector<std::pair<UncertaintyPlanningState, int64_t>>& simulation_results = propagation_results.first;
                    // If simulation results in a single new state, we keep going
                    if (simulation_results.size() == 1)
                    {
                        const UncertaintyPlanningState& new_state = simulation_results[0].first;
                        propagated_states.push_back(std::pair<UncertaintyPlanningState, int64_t>(new_state, parent_offset));
                        current = propagated_states.back().first;
                        parent_offset++;
                        steps++;
                    }
                    // If simulation results in multiple new states, this is the end
                    else if (simulation_results.size() > 1)
                    {
                        for (size_t idx = 0; idx < simulation_results.size(); idx++)
                        {
                            const UncertaintyPlanningState& new_state = simulation_results[idx].first;
                            propagated_states.push_back(std::pair<UncertaintyPlanningState, int64_t>(new_state, parent_offset));
                        }
                        completed = true;
                    }
                    // Otherwise, we're done
                    else
                    {
                        completed = true;
                    }
                }
                return std::pair<std::vector<std::pair<UncertaintyPlanningState, int64_t>>, std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>>(propagated_states, raw_particle_propagations);
            }
        }

        /*
         * Goal check and solution handling functions
         */
        inline double ComputeGoalReachedProbability(const UncertaintyPlanningState& state, const Configuration& goal) const
        {
            size_t within_distance = 0;
            std::pair<const std::vector<Configuration, ConfigAlloc>&, bool> particle_check = state.GetParticlePositionsImmutable();
            const std::vector<Configuration, ConfigAlloc>& particles = particle_check.first;
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                const double distance = robot_.ComputeConfigurationDistance(particles[idx], goal);
                if (distance < goal_distance_threshold_)
                {
                    within_distance++;
                }
            }
            double percent_in_range = (double)within_distance / (double)particles.size();
            return percent_in_range;
        }

        inline bool GoalReached(const UncertaintyPlanningState& state, const UncertaintyPlanningState& goal_state, const uint32_t planner_action_try_attempts, const bool allow_contacts) const
        {
            // *** WARNING ***
            // !!! WE IGNORE THE PROVIDED GOAL STATE, AND INSTEAD ACCESS IT VIA NEAREST-NEIGHBORS STORAGE !!!
            UNUSED(state);
            UNUSED(planner_action_try_attempts);
            UNUSED(allow_contacts);
            UncertaintyPlanningState& goal_state_candidate = nearest_neighbors_storage_.back().GetValueMutable();
            // NOTE - this assumes (safely) that the state passed to this function is the last state added to the tree, which we can safely mutate!
            // We only care about states with control input == goal position (states that are directly trying to go to the goal)
            if (robot_.ComputeConfigurationDistance(goal_state_candidate.GetCommand(), goal_state.GetExpectation()) == 0.0)
            {
                goal_candidates_evaluated_++;
                double goal_reached_probability = ComputeGoalReachedProbability(goal_state_candidate, goal_state.GetExpectation());
                double goal_probability = goal_reached_probability * goal_state_candidate.GetMotionPfeasibility();
                if (goal_probability >= goal_probability_threshold_)
                {
                    // Update the state
                    goal_state_candidate.SetGoalPfeasibility(goal_reached_probability);
                    arc_helpers::ConditionalPrint("Goal reached with state " + goal_state_candidate.Print() + " with probability(this->goal): " + std::to_string(goal_reached_probability) + " and probability(start->goal): " + std::to_string(goal_probability), 3, debug_level_);
                    return true;
                }
            }
            return false;
        }

        inline void GoalReachedCallback(UncertaintyPlanningTreeState& new_goal, const uint32_t planner_action_try_attempts, const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time) const
        {
            // Update the time-to-first-solution if need be
            if (time_to_first_solution_ == 0.0)
            {
                const std::chrono::time_point<std::chrono::high_resolution_clock> current_time = (std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double> elapsed = current_time - start_time;
                time_to_first_solution_ = elapsed.count();
            }
            // Backtrack through the solution path until we reach the root of the current "goal branch"
            // A goal branch is the entire branch leading to the goal
            // Make sure the goal state isn't a branch root itself
            if (CheckIfGoalBranchRoot(new_goal))
            {
                ;//std::cout << "Goal state is the root of its own goal branch, no need to blacklist" << std::endl;
            }
            else
            {
                int64_t current_index = new_goal.GetParentIndex();
                int64_t goal_branch_root_index = -1; // Initialize to an invalid index so we can detect later if it isn't valid
                while (current_index > 0)
                {
                    // Get the current state that we're looking at
                    UncertaintyPlanningTreeState& current_state = nearest_neighbors_storage_[(size_t)current_index];
                    // Check if we've reached the root of the goal branch
                    bool is_branch_root = CheckIfGoalBranchRoot(current_state);
                    // If we haven't reached the root of goal branch
                    if (!is_branch_root)
                    {
                        current_index = current_state.GetParentIndex();
                    }
                    else
                    {
                        goal_branch_root_index = current_index;
                        break;
                    }
                }
                //std::cout << "Backtracked to state " << current_index << " for goal branch blacklisting" << std::endl;
                BlacklistGoalBranch(goal_branch_root_index);
                //std::cout << "Goal branch blacklisting complete" << std::endl;
            }
            // Update the goal reached probability
            // Backtrack all the way to the goal, updating each state's goal_Pfeasbility
            // Make sure something hasn't gone wrong
            assert(new_goal.GetValueImmutable().GetGoalPfeasibility() > 0.0);
            // Backtrack up the tree, updating states as we go
            int64_t current_index = new_goal.GetParentIndex();
            while (current_index >= 0)
            {
                // Get the current state that we're looking at
                UncertaintyPlanningTreeState& current_state = nearest_neighbors_storage_[(size_t)current_index];
                // Update the state
                UpdateNodeGoalReachedProbability(current_state, planner_action_try_attempts);
                current_index = current_state.GetParentIndex();
            }
            // Get the goal reached probability that we use to decide when we're done
            total_goal_reached_probability_ = nearest_neighbors_storage_[0].GetValueImmutable().GetGoalPfeasibility();
            std::cout << "Updated goal reached probability to " << total_goal_reached_probability_ << std::endl;
        }

        inline void BlacklistGoalBranch(const int64_t goal_branch_root_index) const
        {
            if (goal_branch_root_index < 0)
            {
                ;
            }
            else if (goal_branch_root_index == 0)
            {
                std::cerr << "Blacklisting with goal branch root == tree root is not possible!" << std::endl;
            }
            else
            {
                //std::cout << "Blacklisting goal branch starting at index " << goal_branch_root_index << std::endl;
                // Get the current node
                simple_rrt_planner::SimpleRRTPlannerState<UncertaintyPlanningState>& current_state = nearest_neighbors_storage_[(size_t)goal_branch_root_index];
                // Recursively blacklist it
                current_state.GetValueMutable().DisableForNearestNeighbors();
                assert(current_state.GetValueImmutable().UseForNearestNeighbors() == false);
                // Blacklist each child
                const std::vector<int64_t>& child_indices = current_state.GetChildIndices();
                for (size_t idx = 0; idx < child_indices.size(); idx++)
                {
                    int64_t child_index = child_indices[idx];
                    BlacklistGoalBranch(child_index);
                }
            }
        }

        inline bool CheckIfGoalBranchRoot(const UncertaintyPlanningTreeState& state) const
        {
            // There are three ways a state can be the the root of a goal branch
            // 1) The transition leading to the state is low-probability
            const bool has_low_probability_transition = (state.GetValueImmutable().GetEffectiveEdgePfeasibility() < goal_probability_threshold_);
            // 2) The transition leading to the state is the result of an unresolved split
            const bool is_child_of_split = (state.GetValueImmutable().GetSplitId() > 0u) ? true : false;
            // If we're a child of a split, check to see if the split has been resolved:
            // 2a) - the P(goal reached) of the parent is 1
            // 2b) - all the other children with the same transition are already blacklisted
            bool is_child_of_unresolved_split = false;
            if (is_child_of_split)
            {
                const UncertaintyPlanningTreeState& parent_tree_state = nearest_neighbors_storage_[(size_t)state.GetParentIndex()];
                const UncertaintyPlanningState& parent_state = parent_tree_state.GetValueImmutable();
                if (parent_state.GetGoalPfeasibility() >= 1.0)
                {
                    is_child_of_unresolved_split = false;
                }
                else
                {
                    bool other_children_blacklisted = true;
                    const std::vector<int64_t>& other_parent_children = parent_tree_state.GetChildIndices();
                    for (size_t idx = 0; idx < other_parent_children.size(); idx++)
                    {
                        const int64_t other_child_index = other_parent_children[idx];
                        const UncertaintyPlanningTreeState& other_child_tree_state = nearest_neighbors_storage_[(size_t)other_child_index];
                        const UncertaintyPlanningState& other_child_state = other_child_tree_state.GetValueImmutable();
                        if (other_child_state.GetTransitionId() == state.GetValueImmutable().GetTransitionId() && other_child_state.UseForNearestNeighbors())
                        {
                            other_children_blacklisted = false;
                        }
                    }
                    if (other_children_blacklisted)
                    {
                        is_child_of_unresolved_split = false;
                    }
                    else
                    {
                        is_child_of_unresolved_split = true;
                    }
                }
            }
            // 3) The parent of the current node is the root of the tree
            const bool parent_is_root = (state.GetParentIndex() == 0);
            // If one or more condition is true, the state is a branch root
            if (has_low_probability_transition || is_child_of_unresolved_split || parent_is_root)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        inline void UpdateNodeGoalReachedProbability(UncertaintyPlanningTreeState& current_node, const uint32_t planner_action_try_attempts) const
        {
            // Check all the children of the current node, and update the node's goal reached probability accordingly
            //
            // Naively, the goal reached probability of a node is the maximum of the child goal reached probabilities;
            // intuitively, the probability of reaching the goal is that of reaching the goal if we follow the best child.
            //
            // HOWEVER - the existence of "split" child states, where multiple states result from a single control input,
            // makes this more compilcated. For split child states, the goal reached probability of the split is the sum
            // over every split option of (split goal probability * probability of split)
            //
            // We can identify split nodes as children which share a transition id
            // First, we go through the children and separate them based on transition id (this puts all the children of a
            // split together in one place)
            std::map<uint64_t, std::vector<int64_t>> effective_child_branches;
            for (size_t idx = 0; idx < current_node.GetChildIndices().size(); idx++)
            {
                const int64_t& current_child_index = current_node.GetChildIndices()[idx];
                const uint64_t& child_transition_id = nearest_neighbors_storage_[(size_t)current_child_index].GetValueImmutable().GetTransitionId();
                effective_child_branches[child_transition_id].push_back(current_child_index);
            }
            // Now that we have the transitions separated out, compute the goal probability of each transition
            std::vector<double> effective_child_branch_probabilities;
            for (auto itr = effective_child_branches.begin(); itr != effective_child_branches.end(); ++itr)
            {
                double transtion_goal_probability = ComputeTransitionGoalProbability(itr->second, planner_action_try_attempts);
                effective_child_branch_probabilities.push_back(transtion_goal_probability);
            }
            // Now, get the highest transtion probability
            double max_transition_probability = 0.0;
            if (effective_child_branch_probabilities.size() > 0)
            {
                max_transition_probability = *std::max_element(effective_child_branch_probabilities.begin(), effective_child_branch_probabilities.end());
            }
            assert(max_transition_probability > 0.0);
            assert(max_transition_probability <= 1.0);
            // Update the current state
            current_node.GetValueMutable().SetGoalPfeasibility(max_transition_probability);
        }

        inline double ComputeTransitionGoalProbability(const std::vector<int64_t>& child_node_indices, const uint32_t planner_action_try_attempts) const
        {
            std::vector<UncertaintyPlanningState> child_states(child_node_indices.size());
            for (size_t idx = 0; idx < child_node_indices.size(); idx++)
            {
                // Get the current child
                const int64_t& current_child_index = child_node_indices[idx];
                const UncertaintyPlanningState& current_child = nearest_neighbors_storage_[(size_t)current_child_index].GetValueImmutable();
                child_states[idx] = current_child;
            }
            return ComputeTransitionGoalProbability(child_states, planner_action_try_attempts);
        }

        inline double ComputeTransitionGoalProbability(const std::vector<UncertaintyPlanningState>& child_nodes, const uint32_t planner_action_try_attempts) const
        {
            // Let's handle the special cases first
            // The most common case - a non-split transition
            if (child_nodes.size() == 1)
            {
                const UncertaintyPlanningState& current_child = child_nodes.front();
                return (current_child.GetGoalPfeasibility() * current_child.GetEffectiveEdgePfeasibility());
            }
            // IMPOSSIBLE (but we handle it just to be sure)
            else if (child_nodes.size() == 0)
            {
                return 0.0;
            }
            // Let's handle the split case(s)
            else
            {
                // We do this the right way
                std::vector<double> child_goal_reached_probabilities(child_nodes.size(), 0.0);
                // For each child state, we compute the probability that we'll end up at each of the result states, accounting for try/retry with reversibility
                // This lets us compare child states as if they were separate actions, so the overall P(goal reached) = max(child) P(goal reached | child)
                for (size_t idx = 0; idx < child_nodes.size(); idx++)
                {
                    // Get the current child
                    const UncertaintyPlanningState& current_child = child_nodes[idx];
                    // For the selected child, we keep track of the probability that we reach the goal directly via the child state AND the probability that we reach the goal from unintended other child states
                    double percent_active = 1.0;
                    double p_we_reached_goal = 0.0;
                    double p_others_reached_goal = 0.0;
                    for (uint32_t try_attempt = 0; try_attempt < planner_action_try_attempts; try_attempt++)
                    {
                        // How many particles got to our state on this attempt?
                        const double p_reached = percent_active * current_child.GetRawEdgePfeasibility();
                        p_we_reached_goal += (p_reached * current_child.GetGoalPfeasibility());
                        // Update the percent of particles that are still usefully active
                        // and the probability that the goal was reached via a different child
                        double updated_percent_active = 0.0;
                        for (size_t other_idx = 0; other_idx < child_nodes.size(); other_idx++)
                        {
                            if (other_idx != idx)
                            {
                                // Get the other child
                                const UncertaintyPlanningState& other_child = child_nodes[other_idx];
                                const double p_reached_other = percent_active * other_child.GetRawEdgePfeasibility();
                                const double p_returned_to_parent = p_reached_other * other_child.GetReverseEdgePfeasibility();
                                const double p_stuck_at_other = p_reached_other * (1.0 - other_child.GetReverseEdgePfeasibility());
                                const double p_reached_goal_from_other = p_stuck_at_other * other_child.GetGoalPfeasibility();
                                p_others_reached_goal += p_reached_goal_from_other;
                                updated_percent_active += p_returned_to_parent;
                            }
                        }
                        percent_active = updated_percent_active;
                    }
                    double p_reached_goal = p_we_reached_goal + p_others_reached_goal;
                    assert(p_reached_goal >= 0.0);
                    if (p_reached_goal > 1.0)
                    {
                        assert(p_reached_goal <= 1.001);
                        p_reached_goal = 1.0;
                    }
                    assert(p_reached_goal <= 1.0);
                    child_goal_reached_probabilities[idx] = p_reached_goal;
                }
                const double max_child_transition_goal_reached_probability = *std::max_element(child_goal_reached_probabilities.begin(), child_goal_reached_probabilities.end());
                return max_child_transition_goal_reached_probability;
            }
        }

        /*
         * Check if we should stop planning (have we reached the time limit?)
         */
        inline bool PlannerTerminationCheck(const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time, const std::chrono::duration<double>& time_limit) const
        {
            const bool planner_termination_reached = (debug_level_ >= 20) ? (total_goal_reached_probability_ > goal_probability_threshold_) : (((std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now() - start_time) > time_limit);
            if (planner_termination_reached)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    };
}

#endif // UNCERTAINTY_CONTACT_PLANNING_HPP
