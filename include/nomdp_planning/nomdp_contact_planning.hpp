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
#include "arc_utilities/eigen_helpers.hpp"
#include "arc_utilities/pretty_print.hpp"
#include "nomdp_planning/simple_pid_controller.hpp"
#include "nomdp_planning/simple_uncertainty_models.hpp"
#include "arc_utilities/simple_rrt_planner.hpp"
#include "sdf_tools/collision_map.hpp"
#include "sdf_tools/sdf.hpp"

#ifndef NOMDP_CONTACT_PLANNING_HPP
#define NOMDP_CONTACT_PLANNING_HPP

#ifndef DISABLE_ROS_INTERFACE
    #define USE_ROS
#endif

#ifdef USE_ROS
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#endif

namespace nomdp_contact_planning
{
    class NomdpPlannerState
    {
    protected:

        bool initialized_;
        bool has_particles_;
        bool use_for_nearest_neighbors_;
        bool split_child_;
        double step_size_;
        double edge_Pfeasibility_;
        double motion_Pfeasibility_;
        double variance_;
        double space_independent_variance_;
        u_int64_t transition_id_;
        double goal_Pfeasibility_;
        Eigen::Vector3d control_input_;
        Eigen::Vector3d variances_;
        Eigen::Vector3d space_independent_variances_;
        Eigen::Vector3d expectation_;
        EigenHelpers::VectorVector3d particle_positions_;

    public:

        inline NomdpPlannerState(const Eigen::Vector3d& expectation)
        {
            step_size_ = 0.0;
            expectation_ = expectation;
            particle_positions_ = EigenHelpers::VectorVector3d{expectation_};
            variance_ = 0.0;
            variances_ = Eigen::Vector3d(0.0, 0.0, 0.0);
            space_independent_variance_ = 0.0;
            space_independent_variances_ = Eigen::Vector3d(0.0, 0.0, 0.0);
            edge_Pfeasibility_ = 1.0;
            motion_Pfeasibility_ = 1.0;
            initialized_ = true;
            has_particles_ = false;
            use_for_nearest_neighbors_ = true;
            split_child_ = false;
            transition_id_ = 0;
            goal_Pfeasibility_ = 0.0;
        }

        inline NomdpPlannerState(const EigenHelpers::VectorVector3d& particle_positions, const double edge_Pfeasibility, const double parent_motion_Pfeasibility, const double step_size, const Eigen::Vector3d& control_input, const u_int64_t transition_id, const bool is_split_child)
        {
            step_size_ = step_size;
            particle_positions_ = particle_positions;
            UpdateStatistics();
            edge_Pfeasibility_ = edge_Pfeasibility;
            motion_Pfeasibility_ = edge_Pfeasibility_ * parent_motion_Pfeasibility;
            initialized_ = true;
            has_particles_ = true;
            use_for_nearest_neighbors_ = true;
            control_input_ = control_input;
            transition_id_ = transition_id;
            split_child_ = is_split_child;
            goal_Pfeasibility_ = 0.0;
        }

        inline NomdpPlannerState() : initialized_(false), has_particles_(false), use_for_nearest_neighbors_(false), split_child_(false), transition_id_(0), goal_Pfeasibility_(0.0) {}

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline bool HasParticles() const
        {
            return has_particles_;
        }

        inline bool UseForNearestNeighbors() const
        {
            return use_for_nearest_neighbors_;
        }

        inline bool IsSplitChild() const
        {
            return split_child_;
        }

        inline void EnableForNearestNeighbors()
        {
            use_for_nearest_neighbors_ = true;
        }

        inline void DisableForNearestNeighbors()
        {
            use_for_nearest_neighbors_ = false;
        }

        inline double GetStepSize() const
        {
            return step_size_;
        }

        inline double GetEdgePfeasibility() const
        {
            return edge_Pfeasibility_;
        }

        inline double GetMotionPfeasibility() const
        {
            return motion_Pfeasibility_;
        }

        inline Eigen::Vector3d GetExpectation() const
        {
            return expectation_;
        }

        inline double GetVariance() const
        {
            return variance_;
        }

        inline Eigen::Vector3d GetVariances() const
        {
            return variances_;
        }

        inline double GetSpaceIndependentVariance() const
        {
            return space_independent_variance_;
        }

        inline Eigen::Vector3d GetSpaceIndependentVariances() const
        {
            return space_independent_variances_;
        }

        inline double GetGoalPfeasibility() const
        {
            return goal_Pfeasibility_;
        }

        inline void SetGoalPfeasibility(const double goal_Pfeasibility)
        {
            goal_Pfeasibility_ = goal_Pfeasibility;
        }

        inline u_int64_t GetTransitionId() const
        {
            return transition_id_;
        }

        inline Eigen::Vector3d GetControlInput() const
        {
            return control_input_;
        }

        inline std::pair<const EigenHelpers::VectorVector3d&, bool> GetParticlePositionsImmutable() const
        {
            if (has_particles_)
            {
                return std::pair<const EigenHelpers::VectorVector3d&, bool>(particle_positions_, true);
            }
            else
            {
                return std::pair<const EigenHelpers::VectorVector3d&, bool>(particle_positions_, false);
            }
        }

        inline std::pair<EigenHelpers::VectorVector3d&, bool>  GetParticlePositionsMutable()
        {
            if (has_particles_)
            {
                return std::pair<EigenHelpers::VectorVector3d&, bool>(particle_positions_, true);
            }
            else
            {
                return std::pair<EigenHelpers::VectorVector3d&, bool>(particle_positions_, false);
            }
        }

        inline Eigen::Vector3d ComputeExpectation() const
        {
            if (particle_positions_.size() == 0)
            {
                if (has_particles_)
                {
                    return Eigen::Vector3d(NAN, NAN, NAN);
                }
                else
                {
                    return expectation_;
                }
            }
            else if (particle_positions_.size() == 1)
            {
                return particle_positions_[0];
            }
            else
            {
                double weight = 1.0 / (double)particle_positions_.size();
                Eigen::Vector3d avg_sum(0.0, 0.0, 0.0);
                for (size_t idx = 0; idx < particle_positions_.size(); idx++)
                {
                    avg_sum += (particle_positions_[idx] * weight);
                }
                return avg_sum;
            }
        }

        inline double ComputeVariance(const Eigen::Vector3d& expectation) const
        {
            if (particle_positions_.size() == 0)
            {
                return 0.0;
            }
            else if (particle_positions_.size() == 1)
            {
                return 0.0;
            }
            else
            {
                double weight = 1.0 / (double)particle_positions_.size();
                double var_sum = 0.0;
                for (size_t idx = 0; idx < particle_positions_.size(); idx++)
                {
                    double squared_distance = pow(((particle_positions_[idx] - expectation).norm()), 2.0);
                    var_sum += (squared_distance * weight);
                }
                return var_sum;
            }
        }

        inline Eigen::Vector3d ComputeDirectionalVariance(const Eigen::Vector3d& expectation) const
        {
            if (particle_positions_.size() == 0)
            {
                return Eigen::Vector3d(0.0, 0.0, 0.0);
            }
            else if (particle_positions_.size() == 1)
            {
                return Eigen::Vector3d(0.0, 0.0, 0.0);
            }
            else
            {
                double weight = 1.0 / (double)particle_positions_.size();
                Eigen::Vector3d variances(0.0, 0.0, 0.0);
                for (size_t idx = 0; idx < particle_positions_.size(); idx++)
                {
                    Eigen::Vector3d error = particle_positions_[idx] - expectation;
                    Eigen::Vector3d squared_error(error.x() * error.x(), error.y() * error.y(), error.z() * error.z());
                    Eigen::Vector3d weighted_squared_error = squared_error * weight;
                    variances += weighted_squared_error;
                }
                return variances;
            }
        }

        inline Eigen::Vector3d GetParticleExtents(const Eigen::Vector3d& expectation) const
        {
            if (particle_positions_.size() == 0)
            {
                return Eigen::Vector3d(0.0, 0.0, 0.0);
            }
            else if (particle_positions_.size() == 1)
            {
                return Eigen::Vector3d(0.0, 0.0, 0.0);
            }
            else
            {
                double x_extent = 0.0;
                double y_extent = 0.0;
                double z_extent = 0.0;
                for (size_t idx = 0; idx < particle_positions_.size(); idx++)
                {
                    Eigen::Vector3d error = particle_positions_[idx] - expectation;
                    if (fabs(error.x()) > x_extent)
                    {
                        x_extent = fabs(error.x());
                    }
                    if (fabs(error.y()) > y_extent)
                    {
                        y_extent = fabs(error.y());
                    }
                    if (fabs(error.z()) > z_extent)
                    {
                        z_extent = fabs(error.z());
                    }
                }
                return Eigen::Vector3d(x_extent, y_extent, z_extent);
            }
        }

        inline double ComputeSpaceIndependentVariance(const Eigen::Vector3d& expectation, const double step_size) const
        {
            if (particle_positions_.size() == 0)
            {
                return 0.0;
            }
            else if (particle_positions_.size() == 1)
            {
                return 0.0;
            }
            else
            {
                double weight = 1.0 / (double)particle_positions_.size();
                double var_sum = 0.0;
                for (size_t idx = 0; idx < particle_positions_.size(); idx++)
                {
                    double squared_distance = pow((((particle_positions_[idx] - expectation).norm()) / step_size), 2.0);
                    var_sum += (squared_distance * weight);
                }
                return var_sum;
            }
        }

        inline Eigen::Vector3d ComputeSpaceIndependentDirectionalVariance(const Eigen::Vector3d& expectation, const double step_size) const
        {
            if (particle_positions_.size() == 0)
            {
                return Eigen::Vector3d(0.0, 0.0, 0.0);
            }
            else if (particle_positions_.size() == 1)
            {
                return Eigen::Vector3d(0.0, 0.0, 0.0);
            }
            else
            {
                double weight = 1.0 / (double)particle_positions_.size();
                Eigen::Vector3d variances(0.0, 0.0, 0.0);
                for (size_t idx = 0; idx < particle_positions_.size(); idx++)
                {
                    Eigen::Vector3d error = particle_positions_[idx] - expectation;
                    Eigen::Vector3d si_error = error / step_size;
                    Eigen::Vector3d squared_error(si_error.x() * si_error.x(), si_error.y() * si_error.y(), si_error.z() * si_error.z());
                    Eigen::Vector3d weighted_squared_error = squared_error * weight;
                    variances += weighted_squared_error;
                }
                return variances;
            }
        }

        inline std::pair<Eigen::Vector3d, std::pair<std::pair<double, Eigen::Vector3d>, std::pair<double, Eigen::Vector3d>>> UpdateStatistics()
        {
            expectation_ = ComputeExpectation();
            variance_ = ComputeVariance(expectation_);
            variances_ = ComputeDirectionalVariance(expectation_);
            space_independent_variance_ = ComputeSpaceIndependentVariance(expectation_, step_size_);
            space_independent_variances_ = ComputeSpaceIndependentDirectionalVariance(expectation_, step_size_);
            return std::pair<Eigen::Vector3d, std::pair<std::pair<double, Eigen::Vector3d>, std::pair<double, Eigen::Vector3d>>>(expectation_, std::pair<std::pair<double, Eigen::Vector3d>, std::pair<double, Eigen::Vector3d>>(std::pair<double, Eigen::Vector3d>(variance_, variances_), std::pair<double, Eigen::Vector3d>(space_independent_variance_, space_independent_variances_)));
        }

        inline EigenHelpers::VectorVector3d CollectParticles(const size_t num_particles) const
        {
            if (particle_positions_.size() == 0)
            {
                return EigenHelpers::VectorVector3d(num_particles, expectation_);
            }
            else if (particle_positions_.size() == 1)
            {
                return EigenHelpers::VectorVector3d(num_particles, particle_positions_[0]);
            }
            else
            {
                assert(num_particles == particle_positions_.size());
                EigenHelpers::VectorVector3d resampled_particles = particle_positions_;
                return resampled_particles;
            }
        }

        inline EigenHelpers::VectorVector3d ResampleParticles(const size_t num_particles, std::mt19937_64& rng) const
        {
            if (particle_positions_.size() == 0)
            {
                return EigenHelpers::VectorVector3d(num_particles, expectation_);
            }
            else if (particle_positions_.size() == 1)
            {
                return EigenHelpers::VectorVector3d(num_particles, particle_positions_[0]);
            }
            else
            {
                EigenHelpers::VectorVector3d resampled_particles(num_particles);
                //EigenHelpers::VectorVector3d resampled_particles;
                double particle_probability = 1.0 / (double)particle_positions_.size();
                std::uniform_int_distribution<size_t> resampling_distribution(0, particle_positions_.size() - 1);
                std::uniform_real_distribution<double> importance_sampling_distribution(0.0, 1.0);
                size_t resampled = 0;
                while (resampled < num_particles)
                {
                    int random_index = resampling_distribution(rng);
                    Eigen::Vector3d random_particle = particle_positions_[random_index];
                    if (importance_sampling_distribution(rng) < particle_probability)
                    {
                        resampled_particles[resampled] = random_particle;
                        //resampled_particles.push_back(random_particle);
                        resampled++;
                    }
                }
                return resampled_particles;
            }
        }
    };

    struct POINT_ROBOT_CONFIG
    {
        double kp;
        double ki;
        double kd;
        double integral_clamp;
        double velocity_limit;
        double max_sensor_noise;
        double max_actuator_noise;

        POINT_ROBOT_CONFIG()
        {
            kp = 0.0;
            ki = 0.0;
            kd = 0.0;
            integral_clamp = 0.0;
            velocity_limit = 0.0;
            max_sensor_noise = 0.0;
            max_actuator_noise = 0.0;
        }

        POINT_ROBOT_CONFIG(const double in_kp, const double in_ki, const double in_kd, const double in_integral_clamp, const double in_velocity_limit, const double in_max_sensor_noise, const double in_max_actuator_noise)
        {
            kp = in_kp;
            ki = in_ki;
            kd = in_kd;
            integral_clamp = in_integral_clamp;
            velocity_limit = in_velocity_limit;
            max_sensor_noise = in_max_sensor_noise;
            max_actuator_noise = in_max_actuator_noise;
        }
    };

    struct OBSTACLE_CONFIG
    {
        Eigen::Affine3d pose;
        Eigen::Vector3d extents;

        OBSTACLE_CONFIG(const Eigen::Affine3d& in_pose, const Eigen::Vector3d& in_extents) : pose(in_pose), extents(in_extents) {}

        OBSTACLE_CONFIG(const Eigen::Vector3d& in_translation, const Eigen::Quaterniond& in_orientation, const Eigen::Vector3d& in_extents)
        {
            pose = (Eigen::Translation3d)in_translation * in_orientation;
            extents = in_extents;
        }

        OBSTACLE_CONFIG() : pose(Eigen::Affine3d::Identity()), extents(0.0, 0.0, 0.0) {}
    };

    template<typename Observation, typename Action>
    class ExecutionPolicy
    {
    protected:

        bool initialized_;
        std::vector<std::pair<std::pair<Observation, Action>, double>> policy_;
        std::function<double(const Observation&, const Observation&)> distance_fn_;

    public:

        inline ExecutionPolicy(const std::function<double(const Observation&, const Observation&)>& distance_fn) : initialized_(true), distance_fn_(distance_fn) {}

        inline ExecutionPolicy() : initialized_(false), distance_fn_([] (const Observation&, const Observation&) { return INFINITY; }) {}

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline void ExtendPolicy(const Observation& observation, const Action& action, const double confidence)
        {
            assert(initialized_);
            assert(confidence > 0.0);
            assert(confidence <= 1.0);
            policy_.push_back(std::pair<std::pair<Observation, Action>, double>(std::pair<Observation, Action>(observation, action), confidence));
        }

        inline std::pair<Action, std::pair<double, double>> GetAction(const Observation& observation, const double max_distance) const
        {
            assert(initialized_);
            double min_dist = INFINITY;
            double best_confidence = 0.0;
            Action best_action;
            for (size_t idx = 0; idx < policy_.size(); idx++)
            {
                const std::pair<Observation, Action>& candidate = policy_[idx].first;
                const double& confidence = policy_[idx].second;
                const double raw_distance = distance_fn_(observation, candidate.first);
                if (raw_distance <= fabs(max_distance))
                {
                    const double confidence_weight = 1.0 / confidence;
                    const double weighted_distance = raw_distance * confidence_weight;
                    if (weighted_distance < min_dist)
                    {
                        min_dist = raw_distance;
                        best_confidence = confidence;
                        best_action = candidate.second;
                    }
                }
            }
            return std::pair<Action, std::pair<double, double>>(best_action, std::pair<double, double>(min_dist, best_confidence));
        }

        inline const std::vector<std::pair<std::pair<Observation, Action>, double>>& GetRawPolicy() const
        {
            return policy_;
        }
    };

    template<typename Datatype, typename Allocator=std::allocator<Datatype>>
    class SimpleHierarchicalClustering
    {
    protected:

        ;

    public:

        inline SimpleHierarchicalClustering() {}

        std::pair<std::pair<std::pair<bool, int64_t>, std::pair<bool, int64_t>>, double> GetClosestPair(const std::vector<u_int8_t>& datapoint_mask, const Eigen::MatrixXd& distance_matrix, const std::vector<std::vector<int64_t>>& clusters) const
        {
            // Compute distances between unclustered points <-> unclustered points, unclustered_points <-> clusters, and clusters <-> clusters
            // Compute the minimum unclustered point <-> unclustered point / unclustered_point <-> cluster distance
            double min_distance = INFINITY;
            std::pair<int64_t, std::pair<bool, int64_t>> min_element_pair(-1, std::pair<bool, int64_t>(false, -1));
            for (size_t idx = 0; idx < datapoint_mask.size(); idx++)
            {
                // Make sure we aren't in a cluster already
                if (datapoint_mask[idx] == 0)
                {
                    // Compute the minimum unclustered point <-> unclustered point distance
                    double min_point_point_distance = INFINITY;
                    int64_t min_point_index = -1;
                    for (size_t jdx = 0; jdx < datapoint_mask.size(); jdx++)
                    {
                        // Make sure the other point isn't us, and isn't already in a cluster
                        if ((idx != jdx) && (datapoint_mask[jdx] == 0))
                        {
                            const double& current_distance = distance_matrix(idx, jdx);
                            // Update the clkosest point
                            if (current_distance < min_point_point_distance)
                            {
                                min_point_point_distance = current_distance;
                                min_point_index = jdx;
                            }
                        }
                    }
                    // Compute the minimum unclustered point <-> cluster distance
                    double min_point_cluster_distance = INFINITY;
                    int64_t min_cluster_index = -1;
                    for (size_t cdx = 0; cdx < clusters.size(); cdx++)
                    {
                        // We only work with clusters that aren't empty
                        if (clusters[cdx].size() > 0)
                        {
                            // Compute the distance to the current cluster
                            double current_distance = 0.0;
                            for (size_t cpdx = 0; cpdx < clusters[cdx].size(); cpdx++)
                            {
                                const int64_t& current_cluster_point_index = clusters[cdx][cpdx];
                                const double& new_distance = distance_matrix(idx, current_cluster_point_index);
                                if (new_distance > current_distance)
                                {
                                    current_distance = new_distance;
                                }
                            }
                            // Update the closest cluster
                            if (current_distance < min_point_cluster_distance)
                            {
                                min_point_cluster_distance = current_distance;
                                min_cluster_index = cdx;
                            }
                        }
                    }
                    // Update the closest index
                    if (min_point_point_distance < min_distance)
                    {
                        min_distance = min_point_point_distance;
                        min_element_pair.first = idx;
                        min_element_pair.second.first = false;
                        min_element_pair.second.second = min_point_index;
                    }
                    if (min_point_cluster_distance < min_distance)
                    {
                        min_distance = min_point_cluster_distance;
                        min_element_pair.first = idx;
                        min_element_pair.second.first = true;
                        min_element_pair.second.second = min_cluster_index;
                    }
                }
            }
            // Compute the minimum cluster <-> cluster distance
            double min_cluster_cluster_distance = INFINITY;
            std::pair<int64_t, int64_t> min_cluster_pair(-1, -1);
            for (size_t fcdx = 0; fcdx < clusters.size(); fcdx++)
            {
                const std::vector<int64_t>& first_cluster = clusters[fcdx];
                // Don't evaluate empty clusters
                if (first_cluster.size() > 0)
                {
                    for (size_t scdx = 0; scdx < clusters.size(); scdx++)
                    {
                        // Don't compare against ourself
                        if (fcdx != scdx)
                        {
                            const std::vector<int64_t>& second_cluster = clusters[scdx];
                            // Don't evaluate empty clusters
                            if (second_cluster.size() > 0)
                            {
                                // Compute the cluster <-> cluster distance
                                double max_point_point_distance = 0.0;
                                // Find the maximum-pointwise distance between clusters
                                for (size_t fcpx = 0; fcpx < first_cluster.size(); fcpx++)
                                {
                                    for (size_t scpx = 0; scpx < second_cluster.size(); scpx++)
                                    {
                                        const int64_t& fcp_index = first_cluster[fcpx];
                                        const int64_t& scp_index = second_cluster[scpx];
                                        const double& new_distance = distance_matrix(fcp_index, scp_index);
                                        if (new_distance > max_point_point_distance)
                                        {
                                            max_point_point_distance = new_distance;
                                        }
                                    }
                                }
                                double cluster_cluster_distance = max_point_point_distance;
                                if (cluster_cluster_distance < min_cluster_cluster_distance)
                                {
                                    min_cluster_cluster_distance = cluster_cluster_distance;
                                    min_cluster_pair.first = fcdx;
                                    min_cluster_pair.second = scdx;
                                }
                            }
                        }
                    }
                }
            }
            // Return the minimum-distance pair
            if (min_distance <= min_cluster_cluster_distance)
            {
                // Set the indices
                std::pair<bool, int64_t> first_index(false, min_element_pair.first);
                std::pair<bool, int64_t> second_index = min_element_pair.second;
                std::pair<std::pair<bool, int64_t>, std::pair<bool, int64_t>> indices(first_index, second_index);
                std::pair<std::pair<std::pair<bool, int64_t>, std::pair<bool, int64_t>>, double> minimum_pair(indices, min_distance);
                return minimum_pair;
            }
            // A cluster <-> cluster pair is closest
            else
            {
                // Set the indices
                std::pair<bool, int64_t> first_index(true, min_cluster_pair.first);
                std::pair<bool, int64_t> second_index(true, min_cluster_pair.second);
                std::pair<std::pair<bool, int64_t>, std::pair<bool, int64_t>> indices(first_index, second_index);
                std::pair<std::pair<std::pair<bool, int64_t>, std::pair<bool, int64_t>>, double> minimum_pair(indices, min_cluster_cluster_distance);
                return minimum_pair;
            }
        }

        std::vector<std::vector<Datatype, Allocator>> Cluster(const std::vector<Datatype, Allocator>& data, std::function<double(const Datatype&, const Datatype&)>& distance_fn, const double max_cluster_distance) const
        {
            Eigen::MatrixXd distance_matrix = BuildDistanceMatrix(data, distance_fn);
            std::vector<u_int8_t> datapoint_mask(data.size(), 0u);
            std::vector<std::vector<int64_t>> cluster_indices;
            bool complete = false;
            while (!complete)
            {
                // Get closest pair of elements (an element can be a cluster or single data value!)
                std::pair<std::pair<std::pair<bool, int64_t>, std::pair<bool, int64_t>>, double> closest_element_pair = GetClosestPair(datapoint_mask, distance_matrix, cluster_indices);
                //std::cout << "Element pair: " << PrettyPrint::PrettyPrint(closest_element_pair, true) << std::endl;
                if (closest_element_pair.second <= max_cluster_distance)
                {
                    // If both elements are points, create a new cluster
                    if ((closest_element_pair.first.first.first == false) && (closest_element_pair.first.second.first == false))
                    {
                        //std::cout << "New point-point cluster" << std::endl;
                        int64_t first_element_index = closest_element_pair.first.first.second;
                        assert(first_element_index >= 0);
                        int64_t second_element_index = closest_element_pair.first.second.second;
                        assert(second_element_index >= 0);
                        // Add a cluster
                        cluster_indices.push_back(std::vector<int64_t>{first_element_index, second_element_index});
                        // Mask out the indices
                        datapoint_mask[first_element_index] += 1;
                        datapoint_mask[second_element_index] += 1;
                    }
                    // If both elements are clusters, merge the clusters
                    else if ((closest_element_pair.first.first.first == true) && (closest_element_pair.first.second.first == true))
                    {
                        //std::cout << "Combining clusters" << std::endl;
                        // Get the cluster indices
                        int64_t first_cluster_index = closest_element_pair.first.first.second;
                        assert(first_cluster_index >= 0);
                        int64_t second_cluster_index = closest_element_pair.first.second.second;
                        assert(second_cluster_index >= 0);
                        // Merge the second cluster into the first
                        cluster_indices[first_cluster_index].insert(cluster_indices[first_cluster_index].end(), cluster_indices[second_cluster_index].begin(), cluster_indices[second_cluster_index].end());
                        // Empty the second cluster (we don't remove, because this triggers move)
                        cluster_indices[second_cluster_index].clear();
                    }
                    // If one of the elements is a cluster and the other is a point, add the point to the existing cluster
                    else
                    {
                        //std::cout << "Adding to an existing cluster" << std::endl;
                        int64_t cluster_index = -1;
                        int64_t element_index = -1;
                        if (closest_element_pair.first.first.first)
                        {
                            cluster_index = closest_element_pair.first.first.second;
                            element_index = closest_element_pair.first.second.second;
                        }
                        else if (closest_element_pair.first.second.first)
                        {
                            cluster_index = closest_element_pair.first.second.second;
                            element_index = closest_element_pair.first.first.second;
                        }
                        else
                        {
                            assert(false);
                        }
                        assert(cluster_index >= 0);
                        assert(element_index >= 0);
                        // Add the element to the cluster
                        cluster_indices[cluster_index].push_back(element_index);
                        // Mask out the element index
                        datapoint_mask[element_index] += 1;
                    }
                }
                else
                {
                    complete = true;
                }
            }
            // Extract the actual cluster data
            std::vector<std::vector<Datatype, Allocator>> clusters;
            for (size_t idx = 0; idx < cluster_indices.size(); idx++)
            {
                const std::vector<int64_t>& current_cluster = cluster_indices[idx];
                // Ignore empty clusters
                if (current_cluster.size() > 0)
                {
                    std::vector<Datatype, Allocator> new_cluster;
                    for (size_t cdx = 0; cdx < current_cluster.size(); cdx++)
                    {
                        int64_t index = current_cluster[cdx];
                        new_cluster.push_back(data[index]);
                    }
                    clusters.push_back(new_cluster);
                }
            }
            return clusters;
            ///////////////////////////////////////////////////////////////
            /* Start at implementing the SLINK algorithm (eventually CLINK)
            assert(data.size() > 0);
            // Make containers
            std::vector<int64_t> pi(data.size()); // "Pointer representation"
            std::vector<double> lambda(data.size()); // Distance for each pointer
            std::vector<double> distance(data.size()); // Current row in distance matrix
            // Init values
            pi[0] = 0;
            lambda[0] = INFINITY;
            // Loop and update
            for (size_t idx = 1; idx < data.size(); idx++)
            {
                pi[idx] = idx;
                lambda[idx] = INFINITY;
                for (size_t jdx = 0; jdx < (idx - 1); jdx++)
                {
                    distance[jdx] = distance_fn(data[idx], data[jdx]);
                }
                for (size_t jdx = 0; jdx < (idx - 1); jdx++)
                {
                    int64_t next = pi[jdx];
                    if (lambda[jdx] < distance[jdx])
                    {
                        distance[next] = std::min(distance[next], distance[jdx]);
                    }
                    else
                    {
                        distance[next] = std::min(lambda[jdx], distance[next]);
                        pi[jdx] = idx;
                        lambda[jdx] = distance[jdx];
                    }
                }
                for (size_t jdx = 0; jdx < (idx - 1); jdx++)
                {
                    int64_t next = pi[jdx];
                    if (lambda[next] < lambda[jdx])
                    {
                        pi[jdx] = idx;
                    }
                }
            }
            std::cout << "Data: " << PrettyPrint::PrettyPrint(data) << std::endl;
            std::cout << "Pi: " << PrettyPrint::PrettyPrint(pi) << std::endl;
            std::cout << "Lambda: " << PrettyPrint::PrettyPrint(lambda) << std::endl;
            */
        }

        Eigen::MatrixXd BuildDistanceMatrix(const std::vector<Datatype, Allocator>& data, std::function<double(const Datatype&, const Datatype&)>& distance_fn) const
        {
            Eigen::MatrixXd distance_matrix(data.size(), data.size());
#ifdef ENABLE_PARALLEL
            #pragma omp parallel for schedule(guided)
#endif
            for (size_t idx = 0; idx < data.size(); idx++)
            {
                for (size_t jdx = 0; jdx < data.size(); jdx++)
                {
                    distance_matrix(idx, jdx) = distance_fn(data[idx], data[jdx]);
                }
            }
            return distance_matrix;
        }
    };

    class SimpleParticleContactSimulator
    {
    protected:

        bool initialized_;
        double env_min_x_;
        double env_min_y_;
        double env_min_z_;
        double env_max_x_;
        double env_max_y_;
        double env_max_z_;
        sdf_tools::CollisionMapGrid environment_;
        sdf_tools::SignedDistanceField environment_sdf_;

        EigenHelpers::VectorVector3d DiscretizeObstacle(const OBSTACLE_CONFIG& obstacle, const double resolution) const;

        sdf_tools::CollisionMapGrid BuildEnvironment(const std::vector<OBSTACLE_CONFIG>& obstacles, const double resolution) const;

    public:

        inline SimpleParticleContactSimulator(const std::vector<OBSTACLE_CONFIG>& environment_objects, const double environment_resolution, const double min_x, const double min_y, const double min_z, const double max_x, const double max_y, const double max_z)
        {
            env_min_x_ = min_x;
            env_min_y_ = min_y;
            env_min_z_ = min_z;
            env_max_x_ = max_x;
            env_max_y_ = max_y;
            env_max_z_ = max_z;
            environment_ = BuildEnvironment(environment_objects, environment_resolution);
            environment_sdf_ = environment_.ExtractSignedDistanceField(INFINITY).first;
            initialized_ = true;
        }

        inline SimpleParticleContactSimulator() : initialized_(false) {}

        inline Eigen::Affine3d GetOriginTransform() const
        {
            return environment_.GetOriginTransform();
        }

        inline std::string GetFrame() const
        {
            return environment_.GetFrame();
        }

        inline double GetResolution() const
        {
            return environment_.GetResolution();
        }

        inline visualization_msgs::Marker ExportForDisplay() const
        {
            std_msgs::ColorRGBA object_color;
            object_color.r = 0.33;
            object_color.g = 0.42;
            object_color.b = 0.18;
            object_color.a = 1.0;
            std_msgs::ColorRGBA empty_color;
            empty_color.r = 0.0;
            empty_color.g = 0.0;
            empty_color.b = 0.0;
            empty_color.a = 0.0;
            return environment_.ExportForDisplay(object_color, empty_color, empty_color);
        }

        double ComputeTrajectoryCurvature(const EigenHelpers::VectorVector3d& trajectory) const;

        std::pair<Eigen::Vector3d, bool> ForwardSimulatePointRobot(const Eigen::Vector3d& start_position, const Eigen::Vector3d& target_position, const POINT_ROBOT_CONFIG& robot_config, std::mt19937_64& rng, const u_int32_t forward_simulation_steps, const u_int32_t num_simulation_microsteps, double max_curvature, bool allow_contacts) const;

        std::pair<Eigen::Vector3d, bool> ForwardSimulationCallback(const Eigen::Vector3d& start_position, const Eigen::Vector3d& control_input, const u_int32_t num_microsteps) const;
    };

    class NomdpPlanningSpace
    {
    protected:

        bool allow_contacts_;
        bool resample_particles_;
        size_t num_particles_;
        double step_size_;
        double goal_distance_threshold_;
        double goal_probability_threshold_;
        double max_robot_trajectory_curvature_;
        double feasibility_alpha_;
        double variance_alpha_;
        POINT_ROBOT_CONFIG robot_config_;
        simple_rrt_planner::SimpleHybridRRTPlanner<NomdpPlannerState> planner_;
        nomdp_contact_planning::SimpleParticleContactSimulator simulator_;
        nomdp_contact_planning::SimpleHierarchicalClustering<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> clustering_;
        mutable std::mt19937_64 rng_;
        mutable std::vector<std::mt19937_64> rngs_;
        std::uniform_real_distribution<double> x_distribution_;
        std::uniform_real_distribution<double> y_distribution_;
        std::uniform_real_distribution<double> z_distribution_;
        mutable u_int64_t state_counter_;
        mutable u_int64_t transition_id_;
        mutable double total_goal_reached_probability_;
        mutable std::vector<simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState, std::allocator<NomdpPlannerState>>> nearest_neighbors_storage_;

    public:

        inline NomdpPlanningSpace(const bool allow_contacts, const size_t num_particles, const double step_size, const double goal_distance_threshold, const double goal_probability_threshold, const double max_robot_trajectory_curvature, const double feasibility_alpha, const double variance_alpha, const POINT_ROBOT_CONFIG& robot_config, const double min_x_bound, const double min_y_bound, const double min_z_bound, const double max_x_bound, const double max_y_bound, const double max_z_bound, const std::vector<OBSTACLE_CONFIG>& environment_objects, const double environment_resolution, const u_int32_t num_threads) : x_distribution_(min_x_bound, max_x_bound), y_distribution_(min_y_bound, max_y_bound), z_distribution_(min_z_bound, max_z_bound)
        {
            // Prepare the default RNG
            unsigned long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            std::mt19937_64 prng(seed);
            rng_ = prng;
            // Temp seed distribution
            std::uniform_int_distribution<u_int64_t> seed_dist(0, std::numeric_limits<uint64_t>::max());
            assert(num_threads >= 1);
            // Prepare a number of PRNGs for each thread
            for (u_int32_t tidx = 0; tidx < num_threads; tidx++)
            {
                rngs_.push_back(std::mt19937_64(seed_dist(rng_)));
            }
            allow_contacts_ = allow_contacts;
            num_particles_ = num_particles;
            step_size_ = step_size;
            goal_distance_threshold_ = goal_distance_threshold;
            goal_probability_threshold_ = goal_probability_threshold;
            max_robot_trajectory_curvature_ = max_robot_trajectory_curvature;
            feasibility_alpha_ = feasibility_alpha;
            variance_alpha_ = variance_alpha;
            robot_config_ = robot_config;
            planner_ = simple_rrt_planner::SimpleHybridRRTPlanner<NomdpPlannerState>();
            simulator_ = SimpleParticleContactSimulator(environment_objects, environment_resolution, min_x_bound, min_y_bound, min_z_bound, max_x_bound, max_y_bound, max_z_bound);
            state_counter_ = 0;
            transition_id_ = 0;
            nearest_neighbors_storage_.clear();
            resample_particles_ = false;
        }

#ifdef USE_ROS
        std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> Plan(const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const double goal_bias, const std::chrono::duration<double>& time_limit, ros::Publisher& display_pub);
#endif

        std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> Plan(const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const double goal_bias, const std::chrono::duration<double>& time_limit);

        ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d> ExtractPolicy(const std::vector<simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState, std::allocator<NomdpPlannerState>>>& planner_tree, const Eigen::Vector3d& goal_position) const;

#ifdef USE_ROS
        double SimulateExectionPolicy(const ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>& policy, const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const u_int32_t num_particles, const std::chrono::duration<double>& time_limit, ros::Publisher& display_pub) const;
#endif

        double SimulateExectionPolicy(const ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>& policy, const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const u_int32_t num_particles, const std::chrono::duration<double>& time_limit) const;

        std::pair<EigenHelpers::VectorVector3d, bool> SimulateSinglePolicyExecution(const ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>& policy, const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const std::chrono::duration<double>& time_limit, std::mt19937_64& rng) const;

        inline double ComputeGoalReachedProbability(const NomdpPlannerState& state, const Eigen::Vector3d& goal_position) const
        {
            size_t within_distance = 0;
            size_t num_particles = num_particles_;
            std::pair<const EigenHelpers::VectorVector3d&, bool> particle_check = state.GetParticlePositionsImmutable();
            if (!particle_check.second)
            {
                num_particles = 1;
            }
            const EigenHelpers::VectorVector3d& particles = particle_check.first;
            for (size_t idx = 0; idx < particles.size(); idx++)
            {
                if ((particles[idx] - goal_position).norm() < goal_distance_threshold_)
                {
                    within_distance++;
                }
            }
            double percent_in_range = (double)within_distance / (double)num_particles;
            return percent_in_range;
        }

        double StateDistance(const NomdpPlannerState& state1, const NomdpPlannerState& state2) const;

        int64_t GetNearestNeighbor(const std::vector<simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState, std::allocator<NomdpPlannerState>>>& planner_nodes, const NomdpPlannerState& random_state) const;

        NomdpPlannerState SampleRandomTargetState();

        std::pair<std::vector<NomdpPlannerState>, std::pair<EigenHelpers::VectorVector3d, std::vector<std::pair<Eigen::Vector3d, bool>>>> ForwardSimulateParticles(const NomdpPlannerState& nearest, const NomdpPlannerState& random);

        std::vector<EigenHelpers::VectorVector3d> ClusterParticles(const EigenHelpers::VectorVector3d& particles) const;

#ifdef USE_ROS
        std::vector<NomdpPlannerState> PropagateForwardsAndDraw(const NomdpPlannerState& nearest, const NomdpPlannerState& random, ros::Publisher& display_pub);
#endif

        std::vector<NomdpPlannerState> PropagateForwards(const NomdpPlannerState& nearest, const NomdpPlannerState& random);

#ifdef USE_ROS
        visualization_msgs::MarkerArray DrawForwardPropagation(const EigenHelpers::VectorVector3d& start, const std::vector<std::pair<Eigen::Vector3d, bool>>& end, const bool is_split) const;
#endif

        bool GoalReached(const NomdpPlannerState& state, const Eigen::Vector3d& goal_position) const;

        void GoalReachedCallback(simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& new_goal, const Eigen::Vector3d& goal_position) const;

        void BlacklistGoalBranch(const int64_t goal_branch_root_index) const;

        bool CheckIfGoalBranchRoot(const simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& state) const;

        void UpdateNodeGoalReachedProbability(simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& current_node) const;

        double ComputeTransitionGoalProbability(const std::vector<int64_t>& child_node_indices) const;

        inline bool PlannerTerminationCheck(const std::chrono::time_point<std::chrono::high_resolution_clock>& start_time, const std::chrono::duration<double>& time_limit) const
        {
             if (((std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now() - start_time) > time_limit)
             {
                 return true;
             }
             else
             {
                 if (total_goal_reached_probability_ >= 0.99)
                 {
                     return true;
                 }
                 else
                 {
                     return false;
                 }
             }
        }
    };
}

namespace PrettyPrint
{
    template<>
    inline std::string PrettyPrint(const nomdp_contact_planning::NomdpPlannerState& state_to_print, const bool add_delimiters, const std::string& separator)
    {
        UNUSED(add_delimiters);
        UNUSED(separator);
        return "Nomdp Planner State - Expectation: " + PrettyPrint(state_to_print.GetExpectation()) + " Variance: " + std::to_string(state_to_print.GetVariance()) + " Space-independent Variance: " + std::to_string(state_to_print.GetSpaceIndependentVariance()) + " Pfeasibility(parent->this): " + std::to_string(state_to_print.GetEdgePfeasibility()) + " Pfeasibility(start->this): " + std::to_string(state_to_print.GetMotionPfeasibility());
    }

    template<typename Observation, typename Action>
    inline std::string PrettyPrint(const nomdp_contact_planning::ExecutionPolicy<Observation, Action>& policy_to_print, const bool add_delimiters=false, const std::string& separator="")
    {
        UNUSED(add_delimiters);
        UNUSED(separator);
        const std::vector<std::pair<std::pair<Observation, Action>, double>>& raw_policy = policy_to_print.GetRawPolicy();
        std::ostringstream strm;
        strm << "Execution Policy - Policy: ";
        for (size_t idx = 0; idx < raw_policy.size(); idx++)
        {
            const std::pair<Observation, Action>& observation_action_pair = raw_policy[idx].first;
            const double& pair_confidence = raw_policy[idx].second;
            strm << "\nObservation: " << PrettyPrint(observation_action_pair.first) << " | Action: " << PrettyPrint(observation_action_pair.second) << " | Confidence: " << pair_confidence;
        }
        return strm.str();
    }
}

#endif // NOMDP_CONTACT_PLANNING_HPP
