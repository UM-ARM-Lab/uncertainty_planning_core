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
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/simple_hierarchical_clustering.hpp>
#include <arc_utilities/simple_hausdorff_distance.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>
#include <sdf_tools/tagged_object_collision_map.hpp>
#include <sdf_tools/sdf.hpp>
#include <nomdp_planning/simple_pid_controller.hpp>
#include <nomdp_planning/simple_uncertainty_models.hpp>
#include <nomdp_planning/nomdp_planner_state.hpp>
#include <nomdp_planning/simple_particle_contact_simulator.hpp>
#include <nomdp_planning/execution_policy.hpp>

#ifndef NOMDP_CONTACT_PLANNING_HPP
#define NOMDP_CONTACT_PLANNING_HPP

#ifndef DISABLE_ROS_INTERFACE
    #define USE_ROS
#endif

#ifdef USE_ROS
    #include <ros/ros.h>
    #include <visualization_msgs/MarkerArray.h>
    #include <arc_utilities/eigen_helpers_conversions.hpp>
    #include <cv_bridge/cv_bridge.h>
    #include <sensor_msgs/Image.h>
    #include <sensor_msgs/image_encodings.h>
#endif

#ifdef ENABLE_PARALLEL
    #include <omp.h>
#endif

namespace nomdp_contact_planning
{
    template<typename Robot, typename Sampler, typename Configuration, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename InterpolateFn, typename ConfigAlloc=std::allocator<Configuration>, typename PRNG=std::mt19937_64>
    class NomdpPlanningSpace
    {
    protected:

        struct SplitProbabilityEntry
        {
            double probability;
            std::vector<u_int64_t> child_state_ids;

            SplitProbabilityEntry(const double in_probability, const std::vector<u_int64_t>& in_child_state_ids) : probability(in_probability), child_state_ids(in_child_state_ids) {}

            SplitProbabilityEntry() : probability(0.0)
            {
                child_state_ids.clear();
            }
        };

        struct SplitProbabilityTable
        {
            std::vector<SplitProbabilityEntry> split_entries;

            SplitProbabilityTable(const std::vector<SplitProbabilityEntry>& in_split_entries) : split_entries(in_split_entries) {}

            SplitProbabilityTable()
            {
                split_entries.clear();
            }
        };

        // Typedef so we don't hate ourselves
        typedef nomdp_planning_tools::NomdpPlannerState<Configuration, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc> NomdpPlanningState;
        typedef execution_policy::ExecutionPolicy<Configuration, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc> NomdpPlanningPolicy;
        typedef simple_rrt_planner::SimpleRRTPlannerState<NomdpPlanningState, std::allocator<NomdpPlanningState>> NomdpPlanningTreeState;
        typedef std::vector<NomdpPlanningTreeState> NomdpPlanningTree;

        bool allow_contacts_;
        bool resample_particles_;
        size_t num_particles_;
        double step_size_;
        double goal_distance_threshold_;
        double goal_probability_threshold_;
        double signature_matching_threshold_;
        double feasibility_alpha_;
        double variance_alpha_;
        Robot robot_;
        Sampler sampler_;
        nomdp_planning_tools::SimpleParticleContactSimulator simulator_;
        mutable PRNG rng_;
        mutable std::vector<PRNG> rngs_;
        std::uniform_real_distribution<double> x_distribution_;
        std::uniform_real_distribution<double> y_distribution_;
        std::uniform_real_distribution<double> z_distribution_;
        mutable u_int64_t state_counter_;
        mutable u_int64_t transition_id_;
        mutable u_int64_t split_id_;
        mutable u_int64_t cluster_calls_;
        mutable u_int64_t cluster_fallback_calls_;
        mutable double total_goal_reached_probability_;
        mutable std::unordered_map<u_int64_t, SplitProbabilityTable> split_probability_tables_;
        mutable NomdpPlanningTree nearest_neighbors_storage_;

        inline static void ExtractChildStates(const NomdpPlanningTree& raw_planner_tree, const int64_t raw_parent_index, const int64_t pruned_parent_index, NomdpPlanningTree& pruned_planner_tree)
        {
            assert((raw_parent_index >= 0) && (raw_parent_index < (int64_t)raw_planner_tree.size()));
            assert((pruned_parent_index >= 0) && (pruned_parent_index < (int64_t)pruned_planner_tree.size()));
            assert(raw_planner_tree[raw_parent_index].IsInitialized());
            assert(pruned_planner_tree[pruned_parent_index].IsInitialized());
            // Clear the child indices, so we can update them with new values later
            pruned_planner_tree[pruned_parent_index].ClearChildIndicies();
            const std::vector<int64_t>& current_child_indices = raw_planner_tree[raw_parent_index].GetChildIndices();
            for (size_t idx = 0; idx < current_child_indices.size(); idx++)
            {
                const int64_t raw_child_index = current_child_indices[idx];
                assert((raw_child_index > 0) && (raw_child_index < (int64_t)raw_planner_tree.size()));
                const NomdpPlanningTreeState& current_child_state = raw_planner_tree[raw_child_index];
                if (current_child_state.GetParentIndex() >= 0)
                {
                    // Get the new child index
                    const int64_t pruned_child_index = (int64_t)pruned_planner_tree.size();
                    // Add to the pruned tree
                    pruned_planner_tree.push_back(current_child_state);
                    // Update parent indices
                    pruned_planner_tree[pruned_child_index].SetParentIndex(pruned_parent_index);
                    // Update the parent
                    pruned_planner_tree[pruned_parent_index].AddChildIndex(pruned_child_index);
                    // Recursive call
                    ExtractChildStates(raw_planner_tree, raw_child_index, pruned_child_index, pruned_planner_tree);
                }
            }
        }

    public:

        inline NomdpPlanningSpace(const bool allow_contacts, const size_t num_particles, const double step_size, const double goal_distance_threshold, const double goal_probability_threshold, const double signature_matching_threshold, const double feasibility_alpha, const double variance_alpha, const Robot& robot, const Sampler& sampler, const std::vector<nomdp_planning_tools::OBSTACLE_CONFIG>& environment_objects, const double environment_resolution, const u_int32_t num_threads) : robot_(robot), sampler_(sampler)
        {
            // Prepare the default RNG
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            PRNG prng(seed);
            rng_ = prng;
            // Temp seed distribution
            std::uniform_int_distribution<u_int64_t> seed_dist(0, std::numeric_limits<u_int64_t>::max());
            assert(num_threads >= 1);
            // Prepare a number of PRNGs for each thread
            for (u_int32_t tidx = 0; tidx < num_threads; tidx++)
            {
                rngs_.push_back(PRNG(seed_dist(rng_)));
            }
            allow_contacts_ = allow_contacts;
            num_particles_ = num_particles;
            step_size_ = step_size;
            goal_distance_threshold_ = goal_distance_threshold;
            goal_probability_threshold_ = goal_probability_threshold;
            signature_matching_threshold_ = signature_matching_threshold;
            feasibility_alpha_ = feasibility_alpha;
            variance_alpha_ = variance_alpha;
            simulator_ = nomdp_planning_tools::SimpleParticleContactSimulator(environment_objects, environment_resolution);
            state_counter_ = 0;
            transition_id_ = 0;
            split_id_ = 0;
            cluster_calls_ = 0;
            cluster_fallback_calls_ = 0;
            nearest_neighbors_storage_.clear();
            resample_particles_ = false;
        }

#ifdef USE_ROS
        inline nomdp_planning_tools::ForwardSimulationStepTrace<Configuration, ConfigAlloc> DemonstrateSimulator(const Configuration& start, const Configuration& goal, ros::Publisher& display_pub) const
        {
            // Draw the simulation environment
            display_pub.publish(simulator_.ExportAllForDisplay());
            // Wait for input
            std::cout << "Press ENTER to continue..." << std::endl;
            std::cin.get();
            nomdp_planning_tools::ForwardSimulationStepTrace<Configuration, ConfigAlloc> trace;
            const double target_distance = DistanceFn::Distance(start, goal);
            const u_int32_t number_of_steps = (u_int32_t)ceil(target_distance / step_size_) * 40u;
            simulator_.ForwardSimulatePointRobot(robot_, start, goal, rng_, number_of_steps, (goal_distance_threshold_ * 0.5), trace, true);
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
            for (size_t step_idx = 0; step_idx < trace.resolver_steps.size(); step_idx++)
            {
                const nomdp_planning_tools::ForwardSimulationResolverTrace<Configuration, ConfigAlloc>& step_trace = trace.resolver_steps[step_idx];
                for (size_t resolver_step_idx = 0; resolver_step_idx < step_trace.contact_resolver_steps.size(); resolver_step_idx++)
                {
                    const nomdp_planning_tools::ForwardSimulationContactResolverStepTrace<Configuration, ConfigAlloc>& contact_resolution_trace = step_trace.contact_resolver_steps[resolver_step_idx];
                    for (size_t contact_resolution_step_idx = 0; contact_resolution_step_idx < contact_resolution_trace.contact_resolution_steps.size(); contact_resolution_step_idx++)
                    {
                        const Configuration& current_config = contact_resolution_trace.contact_resolution_steps[contact_resolution_step_idx];
                        const std_msgs::ColorRGBA& current_color = (contact_resolution_step_idx == (contact_resolution_trace.contact_resolution_steps.size() - 1)) ? free_color : colliding_color;
                        visualization_msgs::Marker step_marker = DrawRobotConfiguration(robot_, current_config, current_color);
                        step_marker.ns = "step_state";
                        step_marker.id = 1;
                        visualization_msgs::MarkerArray simulator_step_display_rep;
                        simulator_step_display_rep.markers.push_back(step_marker);
                        display_pub.publish(simulator_step_display_rep);
                        // Wait for input
                        std::cout << "Press ENTER to continue..." << std::endl;
                        ros::Duration(0.01).sleep();
                        //std::cin.get();
                    }
                }
            }
            return trace;
        }
#endif

#ifdef USE_ROS
        inline std::pair<NomdpPlanningPolicy, std::map<std::string, double>> Plan(const Configuration& start, const Configuration& goal, const double goal_bias, const std::chrono::duration<double>& time_limit, const bool include_reverse_actions, ros::Publisher& display_pub)
        {
            // Draw the simulation environment
            display_pub.publish(simulator_.ExportAllForDisplay());
            // Wait for input
            std::cout << "Press ENTER to continue..." << std::endl;
#ifndef FORCE_DEBUG
            std::cin.get();
#endif
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
            int wait = 0;
            while (ros::ok() && wait < 10)
            {
                ros::spinOnce();
                ros::Rate(10.0).sleep();
                wait++;
            }
            NomdpPlanningState start_state(start);
            NomdpPlanningState goal_state(goal);
            // Bind the helper functions
            std::function<int64_t(const NomdpPlanningTree&, const NomdpPlanningState&)> nearest_neighbor_fn = std::bind(&NomdpPlanningSpace::GetNearestNeighbor, this, std::placeholders::_1, std::placeholders::_2);
            std::function<bool(const NomdpPlanningState&)> goal_reached_fn = std::bind(&NomdpPlanningSpace::GoalReached, this, std::placeholders::_1, goal);
            std::function<void(NomdpPlanningTreeState&)> goal_reached_callback = std::bind(&NomdpPlanningSpace::GoalReachedCallback, this, std::placeholders::_1);
            std::function<NomdpPlanningState(void)> state_sampling_fn = std::bind(&NomdpPlanningSpace::SampleRandomTargetState, this);
            std::uniform_real_distribution<double> goal_bias_distribution(0.0, 1.0);
            std::function<NomdpPlanningState(void)> complete_sampling_fn = [&](void) { return ((goal_bias_distribution(rng_) > goal_bias) ? state_sampling_fn() : goal_state); };
            std::function<std::vector<std::pair<NomdpPlanningState, int64_t>>(const NomdpPlanningState&, const NomdpPlanningState&)> forward_propagation_fn = std::bind(&NomdpPlanningSpace::PropagateForwardsAndDraw, this, std::placeholders::_1, std::placeholders::_2, display_pub);
            std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
            std::function<bool(void)> termination_check_fn = std::bind(&NomdpPlanningSpace::PlannerTerminationCheck, this, start_time, time_limit);
            // Call the planner
            total_goal_reached_probability_ = 0.0;
            std::pair<std::vector<std::vector<NomdpPlanningState>>, std::map<std::string, double>> planning_results = simple_rrt_planner::SimpleHybridRRTPlanner::PlanMultiPath(nearest_neighbors_storage_, start_state, nearest_neighbor_fn, goal_reached_fn, goal_reached_callback, complete_sampling_fn, forward_propagation_fn, termination_check_fn);
            // Make sure we got somewhere
            std::cout << "Planner terminated with goal reached probability: " << total_goal_reached_probability_ << std::endl;
            std::cout << "Planner performed " << cluster_calls_ << " clustering calls, of which " << cluster_fallback_calls_ << " required hierarchical clustering" << std::endl;
            std::cout << "Planner statistics: " << PrettyPrint::PrettyPrint(planning_results.second) << std::endl;
            NomdpPlanningTree postprocessed_tree = PostProcessTree(nearest_neighbors_storage_);
            NomdpPlanningTree pruned_tree = PruneTree(postprocessed_tree);
            NomdpPlanningPolicy policy = ExtractPolicy(pruned_tree, goal, include_reverse_actions);
            // Draw the final path(s)
            for (size_t pidx = 0; pidx < planning_results.first.size(); pidx++)
            {
                const std::vector<NomdpPlanningState>& planned_path = planning_results.first[pidx];
                if (planned_path.size() >= 2)
                {
                    double goal_reached_probability = planned_path[planned_path.size() - 1].GetGoalPfeasibility() * planned_path[planned_path.size() - 1].GetMotionPfeasibility();
                    visualization_msgs::MarkerArray path_display_rep;
                    for (size_t idx = 0; idx < planned_path.size(); idx++)
                    {
                        const NomdpPlanningState& current_state = planned_path[idx];
                        const Configuration& current_configuration = current_state.GetExpectation();
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
            return std::pair<NomdpPlanningPolicy, std::map<std::string, double>>(policy, planning_results.second);
        }
#endif

        inline std::pair<NomdpPlanningPolicy, std::map<std::string, double>> Plan(const Configuration& start, const Configuration& goal, const double goal_bias, const std::chrono::duration<double>& time_limit, const bool include_reverse_actions)
        {
            NomdpPlanningState start_state(start);
            NomdpPlanningState goal_state(goal);
            // Bind the helper functions
            std::function<int64_t(const NomdpPlanningTree&, const NomdpPlanningState&)> nearest_neighbor_fn = std::bind(&NomdpPlanningSpace::GetNearestNeighbor, this, std::placeholders::_1, std::placeholders::_2);
            std::function<bool(const NomdpPlanningState&)> goal_reached_fn = std::bind(&NomdpPlanningSpace::GoalReached, this, std::placeholders::_1, goal);
            std::function<void(NomdpPlanningTreeState&)> goal_reached_callback = std::bind(&NomdpPlanningSpace::GoalReachedCallback, this, std::placeholders::_1);
            std::function<NomdpPlanningState(void)> state_sampling_fn = std::bind(&NomdpPlanningSpace::SampleRandomTargetState, this);
            std::uniform_real_distribution<double> goal_bias_distribution(0.0, 1.0);
            std::function<NomdpPlanningState(void)> complete_sampling_fn = [&](void) { return ((goal_bias_distribution(rng_) > goal_bias) ? state_sampling_fn() : goal_state); };
            std::function<std::vector<std::pair<NomdpPlanningState, int64_t>>(const NomdpPlanningState&, const NomdpPlanningState&)> forward_propagation_fn = std::bind(&NomdpPlanningSpace::PropagateForwards, this, std::placeholders::_1, std::placeholders::_2);
            std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
            std::function<bool(void)> termination_check_fn = std::bind(&NomdpPlanningSpace::PlannerTerminationCheck, this, start_time, time_limit);
            // Call the planner
            total_goal_reached_probability_ = 0.0;
            std::pair<std::vector<std::vector<NomdpPlanningState>>, std::map<std::string, double>> planning_results = simple_rrt_planner::SimpleHybridRRTPlanner::PlanMultiPath(nearest_neighbors_storage_, start_state, nearest_neighbor_fn, goal_reached_fn, goal_reached_callback, complete_sampling_fn, forward_propagation_fn, termination_check_fn);
            // Make sure we got somewhere
            std::cout << "Planner terminated with goal reached probability: " << total_goal_reached_probability_ << std::endl;
            std::cout << "Planner performed " << cluster_calls_ << " clustering calls, of which " << cluster_fallback_calls_ << " required hierarchical clustering" << std::endl;
            std::cout << "Planner statistics: " << PrettyPrint::PrettyPrint(planning_results.second) << std::endl;
            NomdpPlanningTree postprocessed_tree = PostProcessTree(nearest_neighbors_storage_);
            NomdpPlanningTree pruned_tree = PruneTree(postprocessed_tree);
            NomdpPlanningPolicy policy = ExtractPolicy(pruned_tree, goal, include_reverse_actions);
            return std::pair<NomdpPlanningPolicy, std::map<std::string, double>>(policy, planning_results.second);
        }

        inline NomdpPlanningTree PostProcessTree(const NomdpPlanningTree& planner_tree) const
        {
            std::cout << "Postprocessing planner tree in preparation for policy extraction..." << std::endl;
            std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
            // Let's do some post-processing to the planner tree - we don't want to mess with the original tree, so we copy it
            NomdpPlanningTree postprocessed_planner_tree = planner_tree;
            // We want to figure out which edges of the tree are reversible. Some of these have already been set in the planning process - edges with no collisions on *any* particles and no splits are "trivially reversible"
            // We start with a naive implementation of this - this works because given the process that the tree is generated, children *MUST* have higher indices than their parents, so we can depend on the parents
            // having been updated first by the time we get to an index. To make this parallelizable, we'll need to switch to an explicitly branch-based approach
            // Go through each state in the tree - we skip the initial state, since it has no transition
            u_int64_t updated_states = 0;
            for (size_t sdx = 1; sdx < postprocessed_planner_tree.size(); sdx++)
            {
                // Get the current state
                NomdpPlanningTreeState& current_state = postprocessed_planner_tree[sdx];
                const int64_t parent_index = current_state.GetParentIndex();
                // Get the parent state
                const NomdpPlanningTreeState& parent_state = postprocessed_planner_tree[parent_index];
                // If the current state is on a goal branch
                if (current_state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
                {
                    // If the reversibility has already been set, do nothing
                    if (current_state.GetValueImmutable().GetReverseEdgePfeasibility() > 0.0)
                    {
                        continue;
                    }
                    // If not, we need to compute it
                    else
                    {
                        const double reverse_edge_probability = ComputeReverseEdgeProbability(parent_state.GetValueImmutable(), current_state.GetValueImmutable());
                        current_state.GetValueMutable().SetReverseEdgePfeasibility(reverse_edge_probability);
                        updated_states++;
                    }
                }
                // If we are a non-goal child of a goal branch state
                else if (parent_state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
                {
                    // If the reversibility has already been set, do nothing
                    if (current_state.GetValueImmutable().GetReverseEdgePfeasibility() > 0.0)
                    {
                        continue;
                    }
                    // If not, we need to compute it
                    else
                    {
                        const double reverse_edge_probability = ComputeReverseEdgeProbability(parent_state.GetValueImmutable(), current_state.GetValueImmutable());
                        current_state.GetValueMutable().SetReverseEdgePfeasibility(reverse_edge_probability);
                        const double parent_pgoalreached = parent_state.GetValueImmutable().GetGoalPfeasibility();
                        const double new_pgoalreached = -(parent_pgoalreached * reverse_edge_probability); // We use negative goal reached probabilities to signal probability due to reversing
                        current_state.GetValueMutable().SetGoalPfeasibility(new_pgoalreached);
                        updated_states++;
                    }
                }
                // If we're not on a goal branch
        //        else
        //        {
        //            // If the parent is reversible, we compute reversibility - if not, we don't because we can't reverse before it!
        //            if (parent_state.GetValueImmutable().GetReverseEdgePfeasibility() > 0.5)
        //            {
        //                // We only compute reversibility for states we haven't computed it for already
        //                if (current_state.GetValueImmutable().GetReverseEdgePfeasibility() > 0.0)
        //                {
        //                    double reverse_edge_probability = ComputeReverseEdgeProbability(parent_state.GetValueImmutable(), current_state.GetValueImmutable());
        //                    // We update the reverse edge probability
        //                    current_state.GetValueMutable().SetReverseEdgePfeasibility(reverse_edge_probability);
        //                    // We update the control input to point "backwards" to the parent state
        //                    const Eigen::Vector3d reverse_control_input = parent_state.GetValueImmutable().GetExpectation();
        //                    current_state.GetValueMutable().SetControlInput(reverse_control_input);
        //                    updated_states++;
        //                }
        //            }
        //            // If the parent isn't reversible, we ignore its children
        //            else
        //            {
        //                current_state.GetValueMutable().SetReverseEdgePfeasibility(0.0);
        //            }
        //            // Update the P(goal reached) of the current state to account for reversing
        //            // Get the parent's P(goal reached)
        //            const double parent_pgoalreached = std::max(parent_state.GetValueImmutable().GetGoalPfeasibility(), 0.0);
        //            const double reverse_feasibility = current_state.GetValueImmutable().GetReverseEdgePfeasibility();
        //            const double new_pgoalreached = -(parent_pgoalreached * reverse_feasibility); // We use negative goal reached probabilities to signal probability due to reversing
        //            current_state.GetValueMutable().SetGoalPfeasibility(new_pgoalreached);
        //        }
            }
            std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> postprocessing_time(end_time - start_time);
            std::cout << "...postprocessing complete, updated " << updated_states << " states, took " << postprocessing_time.count() << " seconds" << std::endl;
            return postprocessed_planner_tree;
        }

        inline NomdpPlanningTree PruneTree(const NomdpPlanningTree& planner_tree) const
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
            NomdpPlanningTree intermediate_planner_tree = planner_tree;
            // Loop through the tree and prune unproductive nodes+edges
            for (size_t idx = 0; idx < intermediate_planner_tree.size(); idx++)
            {
                NomdpPlanningTreeState& current_state = intermediate_planner_tree[idx];
                assert(current_state.IsInitialized());
                // Simplest pruning (useful for testing ONLY!) we prune all non-goalpath nodes
                if (current_state.GetValueImmutable().GetGoalPfeasibility() <= 0.0)
                {
                    current_state.SetParentIndex(-1);
                    current_state.ClearChildIndicies();
                }
            }
            // Now, extract the unpruned tree
            NomdpPlanningTree pruned_planner_tree;
            // Add root state
            NomdpPlanningTreeState root_state = intermediate_planner_tree[0];
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

        inline double ComputeReverseEdgeProbability(const NomdpPlanningState& parent, const NomdpPlanningState& child) const
        {
            std::vector<std::pair<NomdpPlanningState, int64_t>> simulation_result = ForwardSimulateParticles(child, parent, 400).first;
            if (simulation_result.size() > 1)
            {
                ;//std::cerr << "WARNING - Reverse edge simulation produced multiple states" << std::endl;
            }
            u_int32_t reached_parent = 0u;
            // Get the target position
            const Configuration target_position = parent.GetExpectation();
            for (size_t ndx = 0; ndx < simulation_result.size(); ndx++)
            {
                const NomdpPlanningState& current_state = simulation_result[ndx].first;
                const std::vector<Configuration, ConfigAlloc>& current_particles = current_state.GetParticlePositionsImmutable().first;
                for (size_t pdx = 0; pdx < current_particles.size(); pdx++)
                {
                    const Configuration& current_particle = current_particles[pdx];
                    // Check if the particle got close enough
                    const double particle_distance = DistanceFn::Distance(current_particle, target_position);
                    if (particle_distance <= goal_distance_threshold_)
                    {
                        reached_parent++;
                    }
                }
            }
            return (double)reached_parent / (double)num_particles_;
        }

        NomdpPlanningPolicy ExtractPolicy(const NomdpPlanningTree& planner_tree, const Configuration& goal, const bool allow_reverse_actions) const
        {
            UNUSED(planner_tree);
            UNUSED(goal);
            UNUSED(allow_reverse_actions);
            NomdpPlanningPolicy policy;
            return policy;
        }

#ifdef USE_ROS
        inline double SimulateExectionPolicy(const NomdpPlanningPolicy& policy, const Configuration& start, const Configuration& goal, const u_int32_t num_particles, const std::chrono::duration<double>& time_limit, ros::Publisher& display_pub, const bool wait_for_user) const
        {
            std::vector<std::vector<Configuration, ConfigAlloc>> particle_executions(num_particles);
            u_int32_t reached_goal = 0;
            for (size_t idx = 0; idx < num_particles; idx++)
            {
                std::pair<std::vector<Configuration, ConfigAlloc>, bool> particle_execution = SimulateSinglePolicyExecution(policy, start, goal, time_limit, rng_);
                particle_executions[idx] = particle_execution.first;
                if (particle_execution.second)
                {
                    reached_goal++;
                }
            }
            // Draw the trajectory in a pretty way
            if (wait_for_user)
            {
                // Wait for input
                std::cout << "Press ENTER to draw pretty particle track..." << std::endl;
                std::cin.get();
            }
            for (size_t idx = 0; idx < num_particles; idx++)
            {
                DrawParticlePolicyExecution((u_int32_t)idx, particle_executions[idx], display_pub);
            }
            return (double)reached_goal / (double)num_particles;
        }
#endif

        inline double SimulateExectionPolicy(const NomdpPlanningPolicy& policy, const Configuration& start, const Configuration& goal, const u_int32_t num_particles, const std::chrono::duration<double>& time_limit) const
        {
            std::vector<std::vector<Configuration, ConfigAlloc>> particle_executions(num_particles);
        #ifdef ENABLE_PARALLEL
            std::atomic<u_int32_t> reached_goal(0);
            #pragma omp parallel for schedule(guided)
        #else
            u_int32_t reached_goal = 0;
        #endif
            for (size_t idx = 0; idx < num_particles; idx++)
            {
        #ifdef ENABLE_PARALLEL
                int th_id = omp_get_thread_num();
                std::pair<std::vector<Configuration, ConfigAlloc>, bool> particle_execution = SimulateSinglePolicyExecution(policy, start, goal, time_limit, rngs_[th_id]);
        #else
                std::pair<std::vector<Configuration, ConfigAlloc>, bool> particle_execution = SimulateSinglePolicyExecution(policy, start, goal, time_limit, rng_);
        #endif
                particle_executions[idx] = particle_execution.first;
                if (particle_execution.second)
                {
                    reached_goal++;
                }
            }
            return (double)reached_goal / (double)num_particles;
        }

        std::pair<std::vector<Configuration, ConfigAlloc>, bool> SimulateSinglePolicyExecution(NomdpPlanningPolicy policy, const Configuration& start, const Configuration& goal, const std::chrono::duration<double>& time_limit, PRNG& rng) const
        {
            UNUSED(policy);
            UNUSED(start);
            UNUSED(goal);
            UNUSED(time_limit);
            UNUSED(rng);
            /*
            EigenHelpers::VectorVector3d trajectory;
            // Make a new robot
            simple_uncertainty_models::Simple3dRobot robot(start, robot_config_.kp, robot_config_.ki, robot_config_.kd, robot_config_.integral_clamp, robot_config_.velocity_limit, robot_config_.max_sensor_noise, robot_config_.max_actuator_noise);
            // Make the simulation callback function
            std::function<std::pair<Eigen::Vector3d, bool>(const Eigen::Vector3d&, const Eigen::Vector3d&)> forward_simulation_callback_fn = std::bind(&SimpleParticleContactSimulator::ForwardSimulationCallback, simulator_, std::placeholders::_1, std::placeholders::_2, 10u);
            // Keep track of where we are
            bool allow_backtracking = true;
            std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
            while (((std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now() - start_time) < time_limit)
            {
                // Get a distribution that represents the sensor uncertainty of the robot
                size_t sensor_belief_samples = 100u;
                EigenHelpers::VectorVector3d sensor_belief_distribution(sensor_belief_samples);
                for (size_t idx = 0; idx < sensor_belief_samples; idx++)
                {
                    sensor_belief_distribution[idx] = robot.GetSensedPosition(rng);
                }
                // Lookup an action from the policy
                std::pair<std::pair<EigenHelpers::VectorVector3d, Eigen::Vector3d>, std::pair<double, double>> policy_action; // = policy.GetAction(sensor_belief_distribution, INFINITY, allow_backtracking);
                Eigen::Vector3d action = policy_action.first.second;
                if (policy_action.second.first > goal_distance_threshold_)
                {
                    // Go towards the observation instead
                    action = EigenVector3dAverager::Average(policy_action.first.first);
                }
                // Execute forwards
                u_int32_t forward_simulation_steps = 40u; // Magic empirically-derived number
                for (u_int32_t step = 0; step < forward_simulation_steps; step++)
                {
                    std::pair<Eigen::Vector3d, bool> result = robot.MoveTowardsTarget(action, 1.0, forward_simulation_callback_fn, rng);
                    trajectory.push_back(result.first);
                    // Last, but not least, check if we've gotten close enough the target state to short-circut the simulation
                    const double target_distance = (result.first - action).norm();
                    if (target_distance < (goal_distance_threshold_ * 0.5))
                    {
                        break;
                    }
                }
                // Check if we've reached the goal
                const Eigen::Vector3d step_end_position = robot.GetPosition();
                if ((step_end_position - goal).norm() <= goal_distance_threshold_)
                {
                    // We've reached the goal!
                    return std::pair<EigenHelpers::VectorVector3d, bool>(trajectory, true);
                }
            }
            // If we get here, we haven't reached the goal!
            return std::pair<EigenHelpers::VectorVector3d, bool>(trajectory, false);
            */
            std::vector<Configuration, ConfigAlloc> particle_trajectory;
            return std::pair<std::vector<Configuration, ConfigAlloc>, bool>(particle_trajectory, false);
        }

#ifdef USE_ROS
        inline void DrawParticlePolicyExecution(const u_int32_t particle_idx, const std::vector<Configuration, ConfigAlloc>& trajectory, ros::Publisher& display_pub) const
        {
            if (trajectory.size() > 1)
            {
                // Draw one step at a time
                for (size_t idx = 0; idx < trajectory.size(); idx++)
                {
                    const Configuration& current_configuration = trajectory[idx];
                    // Draw a ball at the current location
                    std_msgs::ColorRGBA exec_color;
                    exec_color.r = 0.0;
                    exec_color.g = 0.0;
                    exec_color.b = 1.0;
                    exec_color.a = 1.0;
                    visualization_msgs::Marker current_marker = DrawRobotConfiguration(robot_, current_configuration, exec_color);
                    current_marker.ns = "particle_policy_exec_" + std::to_string(particle_idx);
                    current_marker.id = (int)idx;
                    // Send the markers for display
                    visualization_msgs::MarkerArray display_markers;
                    display_markers.markers.push_back(current_marker);
                    display_pub.publish(display_markers);
                    // Wait for a bit
                    std::this_thread::sleep_for(std::chrono::duration<double>(0.05));
                }
            }
            else
            {
                return;
            }
        }
#endif

        inline double ComputeGoalReachedProbability(const NomdpPlanningState& state, const Configuration& goal) const
        {
            size_t within_distance = 0;
            size_t num_particles = num_particles_;
            std::pair<const std::vector<Configuration, ConfigAlloc>&, bool> particle_check = state.GetParticlePositionsImmutable();
            if (!particle_check.second)
            {
                num_particles = 1;
            }
            const std::vector<Configuration, ConfigAlloc>& particles = particle_check.first;
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                const double distance = DistanceFn::Distance(particles[idx], goal);
                if (distance < goal_distance_threshold_)
                {
                    within_distance++;
                }
            }
            double percent_in_range = (double)within_distance / (double)num_particles;
            return percent_in_range;
        }

        inline double StateDistance(const NomdpPlanningState& state1, const NomdpPlanningState& state2) const
        {
            // Get the "space independent" expectation distance
            double expectation_distance = DistanceFn::Distance(state1.GetExpectation(), state2.GetExpectation()) / step_size_;
            // Get the Pfeasibility(start -> state1)
            double feasibility_weight = (1.0 - state1.GetMotionPfeasibility()) * feasibility_alpha_ + (1.0 - feasibility_alpha_);
            // Get the "space independent" variance of state1
            Eigen::VectorXd raw_variances = state1.GetSpaceIndependentVariances();
            double raw_variance = raw_variances.lpNorm<1>();
            // Turn the variance into a weight
            double variance_weight = erf(raw_variance) * variance_alpha_ + (1.0 - variance_alpha_);
            // Compute the actual distance
            double distance = (feasibility_weight * expectation_distance * variance_weight);
            return distance;
        }

        inline int64_t GetNearestNeighbor(const NomdpPlanningTree& planner_nodes, const NomdpPlanningState& random_state) const
        {
            UNUSED(planner_nodes);
            // Get the nearest neighbor (ignoring the disabled states)
            int64_t best_index = -1;
            double best_distance = INFINITY;
        #ifdef ENABLE_PARALLEL
            std::mutex nn_mutex;
            #pragma omp parallel for schedule(guided)
        #endif
            for (size_t idx = 0; idx < nearest_neighbors_storage_.size(); idx++)
            {
                const NomdpPlanningTreeState& current_state = nearest_neighbors_storage_[idx];
                // Only check against states enabled for NN checks
                if (current_state.GetValueImmutable().UseForNearestNeighbors())
                {
                    double state_distance = StateDistance(current_state.GetValueImmutable(), random_state);
        #ifdef ENABLE_PARALLEL
                    std::lock_guard<std::mutex> lock(nn_mutex);
        #endif
                    if (state_distance < best_distance)
                    {
                        best_distance = state_distance;
                        best_index = idx;
                    }
                }
            }
            return best_index;
        }

        inline NomdpPlanningState SampleRandomTargetState()
        {
            const Configuration random_point = sampler_.Sample(rng_);
            NomdpPlanningState random_state(random_point);
            return random_state;
        }

        inline std::pair<std::vector<std::pair<NomdpPlanningState, int64_t>>, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> ForwardSimulateParticles(const NomdpPlanningState& nearest, const NomdpPlanningState& target, const u_int32_t num_simulation_steps) const
        {
            // First, compute a target state
            const Configuration target_point = target.GetExpectation();
            transition_id_++;
            const Configuration control_input = target_point;
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
            // Forward propagate each of the particles
            std::vector<std::pair<Configuration, bool>> propagated_points(num_particles_);
            // We want to parallelize this as much as possible!
#ifdef ENABLE_PARALLEL
            #pragma omp parallel for schedule(guided)
#endif
            for (size_t idx = 0; idx < num_particles_; idx++)
            {
                const Configuration& initial_particle = initial_particles[idx];
                nomdp_planning_tools::ForwardSimulationStepTrace<Configuration, ConfigAlloc> trace;
#ifdef ENABLE_PARALLEL
                int th_id = omp_get_thread_num();
                propagated_points[idx] = simulator_.ForwardSimulatePointRobot(robot_, initial_particle, target_point, rngs_[th_id], num_simulation_steps, (goal_distance_threshold_ * 0.5), trace, false);
#else
                propagated_points[idx] = simulator_.ForwardSimulatePointRobot(robot_, initial_particle, target_point, rng_, num_simulation_steps, (goal_distance_threshold_ * 0.5), trace, false);
#endif
            }
            // Cluster the live particles into (potentially) multiple states
            std::pair<std::vector<std::vector<std::pair<Configuration, bool>>>, SplitProbabilityTable> clustering_result = ClusterParticles(propagated_points);
            const std::vector<std::vector<std::pair<Configuration, bool>>>& particle_clusters = clustering_result.first;
            bool is_split_child = false;
            if (particle_clusters.size() > 1)
            {
                //std::cout << "Transition produced " << particle_clusters.size() << " split states" << std::endl;
                is_split_child = true;
                split_id_++;
                split_probability_tables_[split_id_] = clustering_result.second;
            }
            // Build the forward-propagated states
            std::vector<std::pair<NomdpPlanningState, int64_t>> result_states(particle_clusters.size());
            for (size_t idx = 0; idx < particle_clusters.size(); idx++)
            {
                const std::vector<std::pair<Configuration, bool>>& current_cluster = particle_clusters[idx];
                if (particle_clusters[idx].size() > 0)
                {
                    state_counter_++;
                    double edge_feasibility = (double)current_cluster.size() / (double)num_particles_;
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
                    double reverse_edge_feasibility = 1.0;
                    if (did_collide)
                    {
                        //std::cout << "Simulation resulted in collision, defering reversibility check to post-processing" << std::endl;
                        reverse_edge_feasibility = 0.0;
                    }
                    else if (is_split_child)
                    {
                        //std::cout << "Simulation resulted in split, defering reversibility check to post-processing" << std::endl;
                        reverse_edge_feasibility = 0.0;
                    }
                    NomdpPlanningState propagated_state(state_counter_, particle_locations, edge_feasibility, reverse_edge_feasibility, nearest.GetMotionPfeasibility(), step_size_, control_input, transition_id_, ((is_split_child) ? split_id_ : 0u));
                    // Store the state
                    result_states[idx].first = propagated_state;
                    result_states[idx].second = -1;
                }
            }
            return std::pair<std::vector<std::pair<NomdpPlanningState, int64_t>>, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>(result_states, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>(initial_particles, propagated_points));
        }

        /* OLD NON-DISJOINT CLUSTER CODE
        inline std::vector<std::vector<double>> ComputeConfigurationConvexRegionSignature(Robot robot, const Configuration configuration) const
        {
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points = robot.GetRawLinksPoints();
            // Update the position of the robot
            robot.UpdatePosition(configuration);
            std::vector<std::vector<double>> link_region_signatures(robot_links_points.size());
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                std::vector<double> region_signature(32, 0.0);
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector3d& link_points = robot_links_points[link_idx].second;
                // Get the transform of the current link
                const Eigen::Affine3d link_transform = robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector3d environment_relative_point = link_transform * link_relative_point;
                    std::pair<const sdf_tools::TAGGED_OBJECT_COLLISION_CELL&, bool> query = simulator_.GetEnvironment().GetImmutable(environment_relative_point);
                    if (query.second)
                    {
                        const sdf_tools::TAGGED_OBJECT_COLLISION_CELL& environment_cell = query.first;
                        for (u_int32_t region = 1; region <=32; region++)
                        {
                            if (environment_cell.IsPartOfConvexSegment(region))
                            {
                                region_signature[region - 1]++;
                            }
                        }
                    }
                    else
                    {
                        std::cerr << "WARNING - ROBOT POINT OUTSIDE ENVIRONMENT BOUNDS" << std::endl;
                    }
                }
                // "Normalize"
                std::vector<double> normalized_region_signature(32, 0.0);
                for (size_t idx = 0; idx < region_signature.size(); idx++)
                {
                    normalized_region_signature[idx] = region_signature[idx] * (double)link_points.size();
                }
                link_region_signatures[link_idx] = region_signature;
            }
            return link_region_signatures;
        }

        inline bool DoLinkConvexRegionSignaturesMatch(const std::vector<double>& signature_1, const std::vector<double>& signature_2) const
        {
            assert(signature_1.size() == signature_2.size());
            // Two link signatures match if they share >threshold points in the same region
            for (size_t idx = 0; idx < signature_1.size(); idx++)
            {
                const double signature_1_region_value = signature_1[idx];
                const double signature_2_region_value = signature_2[idx];
                if ((signature_1_region_value >= signature_matching_threshold_) && (signature_2_region_value >= signature_matching_threshold_))
                {
                    return true;
                }
            }
            return false;
        }

        inline bool DoConvexRegionSignaturesMatch(const std::vector<std::vector<double>>& signature_1, const std::vector<std::vector<double>>& signature_2) const
        {
            assert(signature_1.size() == signature_2.size());
            for (size_t idx = 0; idx < signature_1.size(); idx++)
            {
                const std::vector<double>& link_signature_1 = signature_1[idx];
                const std::vector<double>& link_signature_2 = signature_2[idx];
                if (DoLinkConvexRegionSignaturesMatch(link_signature_1, link_signature_2) == false)
                {
                    return false;
                }
            }
            return true;
        }

        inline bool IsClusterSubset(const std::vector<size_t>& candidate_superset, const std::vector<size_t>& candidate_subset) const
        {
            if (candidate_subset.size() > candidate_superset.size())
            {
                return false;
            }
            // We add every element in the superset to a hashmap
            std::unordered_map<size_t, u_int8_t> superset_map(candidate_superset.size());
            for (size_t idx = 0; idx < candidate_superset.size(); idx++)
            {
                const size_t candidate_superset_idx = candidate_superset[idx];
                superset_map[candidate_superset_idx] = 1u;
            }
            // Now, we go through the subset and make sure every element is in the superset
            for (size_t idx = 0; idx < candidate_subset.size(); idx++)
            {
                const size_t candidate_subset_idx = candidate_subset[idx];
                auto found_itr = superset_map.find(candidate_subset_idx);
                // If the subset element is not in the superset, then we're done
                if (found_itr == superset_map.end())
                {
                    return false;
                }
            }
            return true;
        }

        inline std::vector<std::vector<size_t>> GenerateParticleClusterPermutations(const std::vector<size_t>& particle_clusters) const
        {
            assert(particle_clusters.size() > 0);
            assert(particle_clusters.size() <= 8);
            // Reserve the space we need up front
            std::vector<std::vector<size_t>> particle_clusters_permutations;
            particle_clusters_permutations.reserve((size_t)ceil(pow(2.0, (double)particle_clusters.size())) - 1);
            // Now, we need to actually generate all subsets of the set defined by particle_clusters (excluding the empty set)
            // We're going to use the loop iteration variable as a binary counter
#ifdef ENABLE_PARALLEL
            #pragma omp parallel for schedule(guided)
#endif
            for (u_int32_t counter = 1; counter <= (u_int32_t)particle_clusters_permutations.size(); counter++)
            {
                std::vector<size_t> current_permutation;
                current_permutation.reserve(32);
                const u_int32_t current_counter_val = counter;
                // Use the current counter as a binary bitmask that tells us which particle clusters to add to the current permutation
                for (size_t idx = 0; idx < particle_clusters.size(); idx++)
                {
                    const u_int32_t new_counter_val = current_counter_val >> idx;
                    if ((new_counter_val & 0x00000001) == 1)
                    {
                        const size_t corresponding_cluster_id = particle_clusters[idx];
                        current_permutation.push_back(corresponding_cluster_id);
                    }
                }
                current_permutation.shrink_to_fit();
                // MAKE SURE EACH PERMUTATION IS SORTED - this ensures that equivalent permutations are identical
                std::sort(current_permutation.begin(), current_permutation.end());
                particle_clusters_permutations.push_back(current_permutation);
            }
            // This shouldn't affect anything
            particle_clusters_permutations.shrink_to_fit();
            return particle_clusters_permutations;
        }

        inline std::vector<std::vector<std::vector<double>>> GenerateRegionSignatures(const std::vector<std::pair<Configuration, bool>>& particles) const
        {
            // Collect the signatures for each particle
            std::vector<std::vector<std::vector<double>>> particle_region_signatures(particles.size());
            // Loop through all the particles
#ifdef ENABLE_PARALLEL
            #pragma omp parallel for schedule(guided)
#endif
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                const std::pair<Configuration, bool>& particle = particles[idx];
                particle_region_signatures[idx] = ComputeConfigurationConvexRegionSignature(robot_, particle.first);
            }
            return particle_region_signatures;
        }

        inline std::vector<std::vector<size_t>> GenerateInitialRegionSignatureClusters(const std::vector<std::pair<Configuration, bool>>& particles, const std::vector<std::vector<std::vector<double>>>& particle_region_signatures) const
        {
            // We attempt to "grow" a region cluster from every particle. Yes, this is O(n^2), but we can parallelize it
            std::vector<std::vector<size_t>> initial_clusters(particles.size());
#ifdef ENABLE_PARALLEL
            #pragma omp parallel for schedule(guided)
#endif
            for (size_t cluster_idx = 0; cluster_idx < initial_clusters.size(); cluster_idx++)
            {
                const std::vector<std::vector<double>>& initial_cluster_region_signature = particle_region_signatures[cluster_idx];
                std::vector<size_t> cluster_particles;
                cluster_particles.reserve(particles.size());
                for (size_t particle_idx = 0; particle_idx < particles.size(); particle_idx++)
                {
                    if (cluster_idx != particle_idx)
                    {
                        const std::vector<std::vector<double>>& particle_region_signature = particle_region_signatures[particle_idx];
                        if (DoConvexRegionSignaturesMatch(initial_cluster_region_signature, particle_region_signature))
                        {
                            cluster_particles.push_back(particle_idx);
                        }
                    }
                    else
                    {
                        cluster_particles.push_back(particle_idx);
                    }
                }
                cluster_particles.shrink_to_fit();
                initial_clusters[cluster_idx] = cluster_particles;
            }
            return initial_clusters;
        }

        inline std::vector<std::vector<size_t>> RunDistanceClusteringOnInitialClusters(const std::vector<std::vector<size_t>>& initial_clusters, const std::vector<std::pair<Configuration, bool>>& particles) const
        {
            // Now, for each of the initial clusters, we run a second pass of distance-threshold hierarchical clustering
            std::vector<std::vector<size_t>> intermediate_clusters;
            intermediate_clusters.reserve(initial_clusters.size());
            // Let's build the distance function
            // This is a little special - we use the lambda to capture the local context, so we can pass indices to the clustering instead of the actual configurations, but have the clustering *operate* over configurations
            std::function<double(const size_t&, const size_t&)> distance_fn = [&] (const size_t& idx1, const size_t& idx2) { return DistanceFn::Distance(particles[idx1].first, particles[idx2].first); };
            for (size_t cluster_idx = 0; cluster_idx < initial_clusters.size(); cluster_idx++)
            {
                const std::vector<size_t>& current_cluster = initial_clusters[cluster_idx];
                // First, we build the distance matrix
                const Eigen::MatrixXd distance_matrix = arc_helpers::BuildDistanceMatrix(current_cluster, distance_fn);
                // Check the max element of the distance matrix
                const double max_distance = distance_matrix.maxCoeff();
                if (max_distance <= step_size_)
                {
                    intermediate_clusters.push_back(current_cluster);
                }
                else
                {
                    cluster_fallback_calls_++;
                    std::cout << "Cluster by convex region of " << current_cluster.size() << " elements exceeds distance threshold, performing additional hierarchical clustering" << std::endl;
                    const std::vector<std::vector<size_t>> new_clustering = simple_hierarchical_clustering::SimpleHierarchicalClustering::Cluster(current_cluster, distance_matrix, step_size_).first;
                    std::cout << "Additional hierarchical clustering produced " << new_clustering.size() << " clusters" << std::endl;
                    for (size_t ndx = 0; ndx < new_clustering.size(); ndx++)
                    {
                        intermediate_clusters.push_back(new_clustering[ndx]);
                    }
                }
            }
            return intermediate_clusters;
        }

        inline std::vector<std::vector<size_t>> RemoveDuplicateClusters(std::vector<std::vector<size_t>> intermediate_clusters) const
        {
            // Now that we have the intermediate clusters, we need to go through and remove duplicate clusters
            // THIS MUTATES THE CLUSTERS AS IT GOES - THIS CANNOT BE PARALLELIZED!!!
            for (size_t cluster_idx = 0; cluster_idx < intermediate_clusters.size(); cluster_idx++)
            {
                const std::vector<size_t>& our_cluster = intermediate_clusters[cluster_idx];
                if (our_cluster.size() > 0)
                {
                    for (size_t other_cluster_idx = 0; other_cluster_idx < intermediate_clusters.size(); other_cluster_idx++)
                    {
                        if (other_cluster_idx != cluster_idx)
                        {
                            const std::vector<size_t>& other_cluster = intermediate_clusters[other_cluster_idx];
                            if (other_cluster.size() > 0)
                            {
                                // We check if out cluster is a subset of the other cluster
                                if (IsClusterSubset(other_cluster, our_cluster))
                                {
                                    // If we are a subset, then our cluster is unnecessary, and can be removed
                                    intermediate_clusters[cluster_idx].clear();
                                }
                            }
                        }
                    }
                }
            }
            // Now, extract the final index clusters (removing emptied clusters from before)
            std::vector<std::vector<size_t>> final_index_clusters;
            final_index_clusters.reserve(intermediate_clusters.size());
            for (size_t cluster_idx = 0; cluster_idx < intermediate_clusters.size(); cluster_idx++)
            {
                const std::vector<size_t>& cluster = intermediate_clusters[cluster_idx];
                if (cluster.size() > 0)
                {
                    final_index_clusters.push_back(cluster);
                }
            }
            final_index_clusters.shrink_to_fit();
            return final_index_clusters;
        }

        inline SplitProbabilityTable GenerateSplitProbabilityTable(const std::vector<std::vector<size_t>>& final_index_clusters, const std::vector<std::pair<Configuration, bool>>& particles) const
        {
            // Now that we have completed the clustering, we need to build the probability table
            // First, we build a table of particle index -> clusters the particle is in
            std::unordered_map<size_t, std::vector<size_t>> particle_cluster_map(particles.size());
            for (size_t cluster_idx = 0; cluster_idx < final_index_clusters.size(); cluster_idx++)
            {
                const std::vector<size_t>& cluster = final_index_clusters[cluster_idx];
                for (size_t element_idx = 0; element_idx < cluster.size(); element_idx++)
                {
                    const size_t particle_idx = cluster[element_idx];
                    particle_cluster_map[particle_idx].push_back(cluster_idx);
                }
            }
            // Now, we want to invert that table, to make a clusters -> particle count table
            std::map<std::vector<size_t>, u_int32_t> outcome_counts;
            for (auto cluster_map_itr = particle_cluster_map.begin(); cluster_map_itr != particle_cluster_map.end(); ++cluster_map_itr)
            {
                const std::vector<size_t> particle_clusters = cluster_map_itr->second;
                // Now, we need to generate *all* permutations of the particle clusters
                const std::vector<std::vector<size_t>> particle_clusters_permutations = GenerateParticleClusterPermutations(particle_clusters);
                // Now, for each permutation, we increment the outcome count
                for (size_t idx = 0; idx < particle_clusters_permutations.size(); idx++)
                {
                    const std::vector<size_t>& permutation = particle_clusters_permutations[idx];
                    outcome_counts[permutation]++;
                }
            }
            // Now that we have a table mapping outcomes to counts, we can build the split probability table
            std::vector<SplitProbabilityEntry> table_entries;
            table_entries.reserve(outcome_counts.size());
            for (auto outcome_counts_itr = outcome_counts.begin(); outcome_counts_itr != outcome_counts.end(); ++outcome_counts_itr)
            {
                const std::vector<size_t>& outcome_clusters = outcome_counts_itr->first;
                const u_int32_t outcome_count = outcome_counts_itr->second;
                if (outcome_count > 0)
                {
                    // Compute the outcome probability
                    const double raw_outcome_probability = (double)outcome_count / (double)particles.size();
                    // In the final probability computation, some intersections are excluded (i.e. subtracted)
                    // and we handle that here by computing a negative probability
                    double outcome_probability = raw_outcome_probability;
                    if ((outcome_clusters.size() % 2) == 0)
                    {
                        outcome_probability = -1.0 * outcome_probability;
                    }
                    // Make the list of outcome states
                    std::vector<u_int64_t> outcome_states;
                    outcome_states.reserve(outcome_clusters.size());
                    const u_int64_t next_state_id = state_counter_ + 1;
                    for (size_t idx = 0; idx < outcome_clusters.size(); idx++)
                    {
                        const size_t outcome_cluster_index = outcome_clusters[idx];
                        u_int64_t outcome_state_id = next_state_id + outcome_cluster_index;
                        outcome_states.push_back(outcome_state_id);
                    }
                    outcome_states.shrink_to_fit();
                    SplitProbabilityEntry entry(outcome_probability, outcome_states);
                    table_entries.push_back(entry);
                }
            }
            table_entries.shrink_to_fit();
            return SplitProbabilityTable (table_entries);
        }

        inline std::pair<std::vector<std::vector<std::pair<Configuration, bool>>>, SplitProbabilityTable> ClusterParticles(const std::vector<std::pair<Configuration, bool>>& particles) const
        {
            // Make sure there are particles to cluster
            if (particles.size() == 0)
            {
                return std::pair<std::vector<std::vector<std::pair<Configuration, bool>>>, SplitProbabilityTable>(std::vector<std::vector<std::pair<Configuration, bool>>>(), SplitProbabilityTable());
            }
            else if (particles.size() == 1)
            {
                return std::pair<std::vector<std::vector<std::pair<Configuration, bool>>>, SplitProbabilityTable>(std::vector<std::vector<std::pair<Configuration, bool>>>{particles}, SplitProbabilityTable());
            }
            cluster_calls_++;
            // Collect the signatures for each particle
            const std::vector<std::vector<std::vector<double>>> particle_region_signatures = GenerateRegionSignatures(particles);
            // We attempt to "grow" a region cluster from every particle. Yes, this is O(n^2), but we can parallelize it
            const std::vector<std::vector<size_t>> initial_clusters = GenerateInitialRegionSignatureClusters(particles, particle_region_signatures);
            // Now, for each of the initial clusters, we run a second pass of distance-threshold hierarchical clustering
            const std::vector<std::vector<size_t>> intermediate_clusters = RunDistanceClusteringOnInitialClusters(initial_clusters, particles);
            // Now that we have the intermediate clusters, we need to go through and remove duplicate clusters
            const std::vector<std::vector<size_t>> final_index_clusters = RemoveDuplicateClusters(intermediate_clusters);
            // Now that we have completed the clustering, we need to build the probability table
            const SplitProbabilityTable split_probability_table = GenerateSplitProbabilityTable(final_index_clusters, particles);
            // Before we return, we need to convert the index clusters to configuration clusters
            std::vector<std::vector<std::pair<Configuration, bool>>> final_clusters(final_index_clusters.size());
#ifdef ENABLE_PARALLEL
            #pragma omp parallel for schedule(guided)
#endif
            for (size_t cluster_idx = 0; cluster_idx < final_index_clusters.size(); cluster_idx++)
            {
                const std::vector<size_t>& cluster = final_index_clusters[cluster_idx];
                std::vector<std::pair<Configuration, bool>> final_cluster(cluster.size());
                for (size_t element_idx = 0; element_idx < cluster.size(); element_idx++)
                {
                    const size_t particle_idx = cluster[element_idx];
                    assert(particle_idx < particles.size());
                    const std::pair<Configuration, bool>& particle = particles[particle_idx];
                    final_cluster[element_idx] = particle;
                }
                final_clusters[cluster_idx] = final_cluster;
            }
            // Now, return the clusters and probability table
            return std::pair<std::vector<std::vector<std::pair<Configuration, bool>>>, SplitProbabilityTable>(final_clusters, split_probability_table);
        }
        */

        inline std::vector<std::vector<std::vector<u_int32_t>>> GenerateRegionSignatures(const std::vector<std::pair<Configuration, bool>>& particles) const
        {
            // Collect the signatures for each particle
            std::vector<std::vector<std::vector<u_int32_t>>> particle_region_signatures(particles.size());
            // Loop through all the particles
#ifdef ENABLE_PARALLEL
            #pragma omp parallel for schedule(guided)
#endif
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                const std::pair<Configuration, bool>& particle = particles[idx];
                particle_region_signatures[idx] = ComputeConfigurationConvexRegionSignature(robot_, particle.first);
            }
            return particle_region_signatures;
        }

        inline std::vector<std::vector<u_int32_t>> ComputeConfigurationConvexRegionSignature(Robot robot, const Configuration& configuration) const
        {
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points = robot.GetRawLinksPoints();
            // Update the position of the robot
            robot.UpdatePosition(configuration);
            std::vector<std::vector<u_int32_t>> link_region_signatures(robot_links_points.size());
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector3d& link_points = robot_links_points[link_idx].second;
                std::vector<u_int32_t> link_region_signature(link_points.size());
                // Get the transform of the current link
                const Eigen::Affine3d link_transform = robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector3d environment_relative_point = link_transform * link_relative_point;
                    std::pair<const sdf_tools::TAGGED_OBJECT_COLLISION_CELL&, bool> query = simulator_.GetEnvironment().GetImmutable(environment_relative_point);
                    if (query.second)
                    {
                        const sdf_tools::TAGGED_OBJECT_COLLISION_CELL& environment_cell = query.first;
                        link_region_signature[point_idx] = environment_cell.convex_segment;
                    }
                    else
                    {
                        link_region_signature[point_idx] = 0u;
                        std::cerr << "WARNING - ROBOT POINT OUTSIDE ENVIRONMENT BOUNDS" << std::endl;
                    }
                }
                link_region_signatures[link_idx] = link_region_signature;
            }
            return link_region_signatures;
        }

        inline double ComputeConvexRegionSignatureDistance(const std::vector<std::vector<u_int32_t>>& signature_1, const std::vector<std::vector<u_int32_t>>& signature_2) const
        {
            assert(signature_1.size() == signature_2.size());
            size_t total_points = 0u;
            size_t non_matching_points = 0u;
            for (size_t link_idx = 0; link_idx < signature_1.size(); link_idx++)
            {
                const std::vector<u_int32_t>& signature_1_link = signature_1[link_idx];
                const std::vector<u_int32_t>& signature_2_link = signature_2[link_idx];
                assert(signature_1_link.size() == signature_2_link.size());
                for (size_t point_idx = 0; point_idx < signature_1_link.size(); point_idx++)
                {
                    total_points++;
                    const u_int32_t signature_1_point = signature_1_link[point_idx];
                    const u_int32_t signature_2_point = signature_2_link[point_idx];
                    if ((signature_1_point & signature_2_point) == 0u)
                    {
                        non_matching_points++;
                    }
                }
            }
            assert(total_points > 0u);
            const double distance = (double)non_matching_points / (double)total_points;
            return distance;
        }

        inline std::vector<std::vector<size_t>> GenerateInitialRegionSignatureClusters(const std::vector<std::vector<std::vector<u_int32_t>>>& particle_region_signatures) const
        {
            // Generate an initial "cluster" with everything in it
            std::vector<size_t> initial_cluster(particle_region_signatures.size());
#ifdef ENABLE_PARALLEL
            #pragma omp parallel for schedule(guided)
#endif
            for (size_t idx = 0; idx < initial_cluster.size(); idx++)
            {
                initial_cluster[idx] = idx;
            }
            // Let's build the distance function
            // This is a little special - we use the lambda to capture the local context, so we can pass indices to the clustering instead of the actual configurations, but have the clustering *operate* over configurations
            std::function<double(const size_t&, const size_t&)> distance_fn = [&] (const size_t& idx1, const size_t& idx2) { return ComputeConvexRegionSignatureDistance(particle_region_signatures[idx1], particle_region_signatures[idx2]); };
            const Eigen::MatrixXd distance_matrix = arc_helpers::BuildDistanceMatrix(initial_cluster, distance_fn);
            // Check the max element of the distance matrix
            const double max_distance = distance_matrix.maxCoeff();
            if (max_distance <= signature_matching_threshold_)
            {
                return std::vector<std::vector<size_t>>{initial_cluster};
            }
            else
            {
                return simple_hierarchical_clustering::SimpleHierarchicalClustering::Cluster(initial_cluster, distance_matrix, step_size_).first;
            }
        }

        inline std::vector<std::vector<size_t>> RunDistanceClusteringOnInitialClusters(const std::vector<std::vector<size_t>>& initial_clusters, const std::vector<std::pair<Configuration, bool>>& particles) const
        {
            // Now, for each of the initial clusters, we run a second pass of distance-threshold hierarchical clustering
            std::vector<std::vector<size_t>> intermediate_clusters;
            intermediate_clusters.reserve(initial_clusters.size());
            // Let's build the distance function
            // This is a little special - we use the lambda to capture the local context, so we can pass indices to the clustering instead of the actual configurations, but have the clustering *operate* over configurations
            std::function<double(const size_t&, const size_t&)> distance_fn = [&] (const size_t& idx1, const size_t& idx2) { return DistanceFn::Distance(particles[idx1].first, particles[idx2].first); };
            for (size_t cluster_idx = 0; cluster_idx < initial_clusters.size(); cluster_idx++)
            {
                const std::vector<size_t>& current_cluster = initial_clusters[cluster_idx];
                // First, we build the distance matrix
                const Eigen::MatrixXd distance_matrix = arc_helpers::BuildDistanceMatrix(current_cluster, distance_fn);
                // Check the max element of the distance matrix
                const double max_distance = distance_matrix.maxCoeff();
                if (max_distance <= step_size_)
                {
                    intermediate_clusters.push_back(current_cluster);
                }
                else
                {
                    cluster_fallback_calls_++;
                    std::cout << "Cluster by convex region of " << current_cluster.size() << " elements exceeds distance threshold, performing additional hierarchical clustering" << std::endl;
                    const std::vector<std::vector<size_t>> new_clustering = simple_hierarchical_clustering::SimpleHierarchicalClustering::Cluster(current_cluster, distance_matrix, step_size_).first;
                    std::cout << "Additional hierarchical clustering produced " << new_clustering.size() << " clusters" << std::endl;
                    for (size_t ndx = 0; ndx < new_clustering.size(); ndx++)
                    {
                        intermediate_clusters.push_back(new_clustering[ndx]);
                    }
                }
            }
            return intermediate_clusters;
        }

        inline std::pair<std::vector<std::vector<std::pair<Configuration, bool>>>, SplitProbabilityTable> ClusterParticles(const std::vector<std::pair<Configuration, bool>>& particles) const
        {
            // Make sure there are particles to cluster
            if (particles.size() == 0)
            {
                return std::pair<std::vector<std::vector<std::pair<Configuration, bool>>>, SplitProbabilityTable>(std::vector<std::vector<std::pair<Configuration, bool>>>(), SplitProbabilityTable());
            }
            else if (particles.size() == 1)
            {
                return std::pair<std::vector<std::vector<std::pair<Configuration, bool>>>, SplitProbabilityTable>(std::vector<std::vector<std::pair<Configuration, bool>>>{particles}, SplitProbabilityTable());
            }
            cluster_calls_++;
            // Collect the signatures for each particle
            const std::vector<std::vector<std::vector<u_int32_t>>> particle_region_signatures = GenerateRegionSignatures(particles);
            // We attempt to "grow" a region cluster from every particle. Yes, this is O(n^2), but we can parallelize it
            const std::vector<std::vector<size_t>> initial_clusters = GenerateInitialRegionSignatureClusters(particle_region_signatures);
            // Now, for each of the initial clusters, we run a second pass of distance-threshold hierarchical clustering
            const std::vector<std::vector<size_t>> final_index_clusters = RunDistanceClusteringOnInitialClusters(initial_clusters, particles);
            // Now that we have completed the clustering, we need to build the probability table
            const SplitProbabilityTable split_probability_table;
            // Before we return, we need to convert the index clusters to configuration clusters
            std::vector<std::vector<std::pair<Configuration, bool>>> final_clusters(final_index_clusters.size());
#ifdef ENABLE_PARALLEL
            #pragma omp parallel for schedule(guided)
#endif
            for (size_t cluster_idx = 0; cluster_idx < final_index_clusters.size(); cluster_idx++)
            {
                const std::vector<size_t>& cluster = final_index_clusters[cluster_idx];
                std::vector<std::pair<Configuration, bool>> final_cluster(cluster.size());
                for (size_t element_idx = 0; element_idx < cluster.size(); element_idx++)
                {
                    const size_t particle_idx = cluster[element_idx];
                    assert(particle_idx < particles.size());
                    const std::pair<Configuration, bool>& particle = particles[particle_idx];
                    final_cluster[element_idx] = particle;
                }
                final_clusters[cluster_idx] = final_cluster;
            }
            // Now, return the clusters and probability table
            return std::pair<std::vector<std::vector<std::pair<Configuration, bool>>>, SplitProbabilityTable>(final_clusters, split_probability_table);
        }

#ifdef USE_ROS
        inline visualization_msgs::Marker DrawRobotConfiguration(Robot robot, const Configuration& configuration, const std_msgs::ColorRGBA& color) const
        {
            std_msgs::ColorRGBA real_color = color;
            visualization_msgs::Marker configuration_marker;
            configuration_marker.action = visualization_msgs::Marker::ADD;
            configuration_marker.ns = "UNKNOWN";
            configuration_marker.id = 1;
            configuration_marker.frame_locked = false;
            configuration_marker.lifetime = ros::Duration(0.0);
            configuration_marker.type = visualization_msgs::Marker::SPHERE_LIST;
            configuration_marker.header.frame_id = simulator_.GetFrame();
            configuration_marker.scale.x = simulator_.GetResolution();
            configuration_marker.scale.y = simulator_.GetResolution();
            configuration_marker.scale.z = simulator_.GetResolution();
            configuration_marker.pose = EigenHelpersConversions::EigenAffine3dToGeometryPose(Eigen::Affine3d::Identity());
            configuration_marker.color = real_color;
            // Make the indivudal points
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points = robot.GetRawLinksPoints();
            // Update the position of the robot
            robot.UpdatePosition(configuration);
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector3d link_points = robot_links_points[link_idx].second;
                // Get the transform of the current link
                const Eigen::Affine3d link_transform = robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector3d environment_relative_point = link_transform * link_relative_point;
                    const geometry_msgs::Point marker_point = EigenHelpersConversions::EigenVector3dToGeometryPoint(environment_relative_point);
                    configuration_marker.points.push_back(marker_point);
                    configuration_marker.colors.push_back(real_color);
                }
            }
            return configuration_marker;
        }
#endif

#ifdef USE_ROS
        inline std::vector<std::pair<NomdpPlanningState, int64_t>> PropagateForwardsAndDraw(const NomdpPlanningState& nearest, const NomdpPlanningState& random, ros::Publisher& display_pub)
        {
            // First, perform the forwards propagation
            std::pair<std::vector<std::pair<NomdpPlanningState, int64_t>>, std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>> propagated_state = PerformForwardPropagation(nearest, random);
            // Draw the expansion
            visualization_msgs::MarkerArray propagation_display_rep;
            // Check if the expansion was useful
            if (propagated_state.first.size() > 0)
            {
                for (size_t idx = 0; idx < propagated_state.first.size(); idx++)
                {
                    int64_t state_index = state_counter_ + (idx - (propagated_state.first.size() - 1));
                    // Yeah, sorry about the ternary. This is so we can still have a const reference
                    //const NomdpPlanningState& previous_state = (propagated_state.first[idx].second >= 0) ? propagated_state.first[propagated_state.first[idx].second].first : nearest;
                    const NomdpPlanningState& current_state = propagated_state.first[idx].first;
                    // Get the edge feasibility
                    const double edge_Pfeasibility = current_state.GetEdgePfeasibility();
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
            return propagated_state.first;
        }
#endif

        inline std::vector<std::pair<NomdpPlanningState, int64_t>> PropagateForwards(const NomdpPlanningState& nearest, const NomdpPlanningState& random)
        {
            return PerformForwardPropagation(nearest, random).first;
        }

        inline std::pair<std::vector<std::pair<NomdpPlanningState, int64_t>>, std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>> PerformForwardPropagation(const NomdpPlanningState& nearest, const NomdpPlanningState& random)
        {
            // First, check if we're going to use RRT-Connect or RRT-Extend
            // If we've already found a solution, we use RRT-Extend
            if (total_goal_reached_probability_ > 0.0)
            {
                // Compute a single target state
                Configuration target_point = random.GetExpectation();
                const double target_distance = DistanceFn::Distance(nearest.GetExpectation(), target_point);
                if (target_distance > step_size_)
                {
                    const double step_fraction = step_size_ / target_distance;
                    const Configuration interpolated_target_point = InterpolateFn::Interpolate(nearest.GetExpectation(), target_point, step_fraction);
                    target_point = interpolated_target_point;
                }
                NomdpPlanningState target_state(target_point);

                std::pair<std::vector<std::pair<NomdpPlanningState, int64_t>>, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> propagation_results = ForwardSimulateParticles(nearest, target_state, 40u);
                std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> raw_particle_propagations = {propagation_results.second};
                return std::pair<std::vector<std::pair<NomdpPlanningState, int64_t>>, std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>>(propagation_results.first, raw_particle_propagations);
            }
            // If we haven't found a solution yet, we use RRT-Connect
            else
            {
                std::vector<std::pair<NomdpPlanningState, int64_t>> propagated_states;
                std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> raw_particle_propagations;
                int64_t parent_offset = -1;
                // Compute a maximum number of steps to take
                const Configuration target_point = random.GetExpectation();
                const u_int32_t total_steps = (u_int32_t)ceil(DistanceFn::Distance(nearest.GetExpectation(), target_point) / step_size_);
                NomdpPlanningState current = nearest;
                u_int32_t steps = 0;
                bool completed = false;
                while ((completed == false) && (steps < total_steps))
                {
                    // Compute a single target state
                    Configuration current_target_point = target_point;
                    const double target_distance = DistanceFn::Distance(current.GetExpectation(), current_target_point);
                    if (target_distance > step_size_)
                    {
                        const double step_fraction = step_size_ / target_distance;
                        const Configuration interpolated_target_point = InterpolateFn::Interpolate(current.GetExpectation(), target_point, step_fraction);
                        current_target_point = interpolated_target_point;
                    }
                    // If we're less than step size away, this is our last step
                    else
                    {
                        completed = true;
                    }
                    // Take a step forwards
                    NomdpPlanningState target_state(current_target_point);
                    std::pair<std::vector<std::pair<NomdpPlanningState, int64_t>>, std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>> propagation_results = ForwardSimulateParticles(nearest, target_state, 40u);
                    raw_particle_propagations.push_back(propagation_results.second);
                    const std::vector<std::pair<NomdpPlanningState, int64_t>>& simulation_results = propagation_results.first;
                    // If simulation results in a single new state, we keep going
                    if (simulation_results.size() == 1)
                    {
                        const NomdpPlanningState& new_state = simulation_results[0].first;
                        propagated_states.push_back(std::pair<NomdpPlanningState, int64_t>(new_state, parent_offset));
                        current = propagated_states.back().first;
                        parent_offset++;
                        steps++;
                    }
                    // If simulation results in multiple new states, this is the end
                    else if (simulation_results.size() > 1)
                    {
                        for (size_t idx = 0; idx < simulation_results.size(); idx++)
                        {
                            const NomdpPlanningState& new_state = simulation_results[idx].first;
                            propagated_states.push_back(std::pair<NomdpPlanningState, int64_t>(new_state, parent_offset));
                        }
                        completed = true;
                    }
                    // Otherwise, we're done
                    else
                    {
                        completed = true;
                    }
                }
                return std::pair<std::vector<std::pair<NomdpPlanningState, int64_t>>, std::vector<std::pair<std::vector<Configuration, ConfigAlloc>, std::vector<std::pair<Configuration, bool>>>>>(propagated_states, raw_particle_propagations);
            }
        }

        inline bool GoalReached(const NomdpPlanningState& state, const Configuration& goal_position) const
        {
            // *** WARNING ***
            // !!! WE IGNORE THE PROVIDED GOAL STATE, AND INSTEAD ACCESS IT VIA NEAREST-NEIGHBORS STORAGE !!!
            UNUSED(state);
            NomdpPlanningState& goal_state_candidate = nearest_neighbors_storage_.back().GetValueMutable();
            // NOTE - this assumes (safely) that the state passed to this function is the last state added to the tree, which we can safely mutate!
            // We only care about states with control input == goal position (states that are directly trying to go to the goal)
            if (DistanceFn::Distance(goal_state_candidate.GetCommand(), goal_position) == 0.0)
            {
                double goal_reached_probability = ComputeGoalReachedProbability(goal_state_candidate, goal_position);
                double goal_probability = goal_reached_probability * goal_state_candidate.GetMotionPfeasibility();
                if (goal_reached_probability >= goal_probability_threshold_)
                {
                    // Update the state
                    goal_state_candidate.SetGoalPfeasibility(goal_reached_probability);
                    std::cout << "Goal reached with state " << PrettyPrint::PrettyPrint(goal_state_candidate) << " with probability: " << goal_reached_probability << " and path probability: " << goal_probability << std::endl;
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }

        inline void GoalReachedCallback(NomdpPlanningTreeState& new_goal) const
        {
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
                    NomdpPlanningTreeState& current_state = nearest_neighbors_storage_[current_index];
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
                NomdpPlanningTreeState& current_state = nearest_neighbors_storage_[current_index];
                // Update the state
                UpdateNodeGoalReachedProbability(current_state);
                current_index = current_state.GetParentIndex();
            }
            // Get the goal reached probability that we use to decide when we're done
            total_goal_reached_probability_ = nearest_neighbors_storage_[0].GetValueImmutable().GetGoalPfeasibility();
            //std::cout << "Updated goal reached probability to " << total_goal_reached_probability_ << std::endl;
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
                simple_rrt_planner::SimpleRRTPlannerState<NomdpPlanningState>& current_state = nearest_neighbors_storage_[goal_branch_root_index];
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

        inline bool CheckIfGoalBranchRoot(const simple_rrt_planner::SimpleRRTPlannerState<NomdpPlanningState>& state) const
        {
            // There are three ways a state can be the the root of a goal branch
            // 1) The transition leading to the state is low-probability
            const bool has_low_probability_transition = (state.GetValueImmutable().GetEdgePfeasibility() < goal_probability_threshold_);
            // 2) The transition leading to the state is the result of a split
            const bool is_child_of_split = (state.GetValueImmutable().GetSplitId() > 0u) ? true : false;
            // 3) The parent of the current node is the root of the tree
            const bool parent_is_root = (state.GetParentIndex() == 0);
            // If one or more condition is true, the state is a branch root
            if (has_low_probability_transition || is_child_of_split || parent_is_root)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        inline void UpdateNodeGoalReachedProbability(simple_rrt_planner::SimpleRRTPlannerState<NomdpPlanningState>& current_node) const
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
            std::map<u_int64_t, std::vector<int64_t>> effective_child_branches;
            for (size_t idx = 0; idx < current_node.GetChildIndices().size(); idx++)
            {
                const int64_t& current_child_index = current_node.GetChildIndices()[idx];
                const u_int64_t& child_transition_id = nearest_neighbors_storage_[current_child_index].GetValueImmutable().GetTransitionId();
                effective_child_branches[child_transition_id].push_back(current_child_index);
            }
            // Now that we have the transitions separated out, compute the goal probability of each transition
            std::vector<double> effective_child_branch_probabilities;
            for (auto itr = effective_child_branches.begin(); itr != effective_child_branches.end(); ++itr)
            {
                double transtion_goal_probability = ComputeTransitionGoalProbability(itr->second);
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

        inline double ComputeTransitionGoalProbability(const std::vector<int64_t>& child_node_indices) const
        {
            double total_transition_goal_probability = 0.0;
            for (size_t idx = 0; idx < child_node_indices.size(); idx++)
            {
                const int64_t& current_child_index = child_node_indices[idx];
                const NomdpPlanningState& current_child = nearest_neighbors_storage_[current_child_index].GetValueImmutable();
                total_transition_goal_probability += (current_child.GetGoalPfeasibility() * current_child.GetEdgePfeasibility());
            }
            return total_transition_goal_probability;
        }

        /* OLD NON-DISJOINT CLUSTER CODE
        inline double GetMaxChildGoalProbability(const std::vector<u_int64_t>& child_state_ids) const
        {
            double best_child_goal_probability = 0.0;
            for (size_t idx = 0; idx < child_state_ids.size(); idx++)
            {
                const u_int64_t& current_child_index = child_state_ids[idx];
                assert(current_child_index < nearest_neighbors_storage_.size());
                const NomdpPlanningState& current_child = nearest_neighbors_storage_[current_child_index].GetValueImmutable();
                const double child_goal_probability = current_child.GetGoalPfeasibility();
                if (child_goal_probability > best_child_goal_probability)
                {
                    best_child_goal_probability = child_goal_probability;
                }
            }
            return best_child_goal_probability;
        }

        inline double ComputeTransitionGoalProbability(const std::vector<int64_t>& child_node_indices) const
        {
            assert(child_node_indices.size() > 0);
            // If no splits occurred, we take the simple path
            if (child_node_indices.size() == 1)
            {
                const int64_t& current_child_index = child_node_indices[0];
                const NomdpPlanningState& current_child = nearest_neighbors_storage_[current_child_index].GetValueImmutable();
                return (current_child.GetGoalPfeasibility() * current_child.GetEdgePfeasibility());
            }
            // Otherwise, we lookup the corresponding split probability table
            else
            {
                // Make sure we agree on the split ID
                const u_int64_t split_id = nearest_neighbors_storage_[child_node_indices[0]].GetValueImmutable().GetSplitId();
                for (size_t idx = 0; idx < child_node_indices.size(); idx++)
                {
                    const int64_t& current_child_index = child_node_indices[idx];
                    const NomdpPlanningState& current_child = nearest_neighbors_storage_[current_child_index].GetValueImmutable();
                    assert(current_child.GetSplitId() == split_id);
                }
                // Get the split probability table
                auto found_itr = split_probability_tables_.find(split_id);
                assert(found_itr != split_probability_tables_.end());
                const SplitProbabilityTable& split_probability_table = found_itr->second;
                // Now, compute the total transition goal proability
                double total_transition_goal_probability = 0.0;
                for (size_t idx = 0; idx < split_probability_table.split_entries.size(); idx++)
                {
                    const SplitProbabilityEntry& current_entry = split_probability_table.split_entries[idx];
                    const double transition_probability = current_entry.probability;
                    const double goal_probability = GetMaxChildGoalProbability(current_entry.child_state_ids);
                    const double outcome_probability = transition_probability * goal_probability;
                    total_transition_goal_probability += outcome_probability;
                }
                return total_transition_goal_probability;
            }
        }
        */

        inline bool PlannerTerminationCheck(const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time, const std::chrono::duration<double>& time_limit) const
        {
#ifdef FORCE_DEBUG
            UNUSED(start_time);
            UNUSED(time_limit);
            if (total_goal_reached_probability_ > goal_probability_threshold_)
#else
            if (((std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now() - start_time) > time_limit)
#endif
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

#endif // NOMDP_CONTACT_PLANNING_HPP
