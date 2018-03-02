#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <random>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/simple_robot_model_interface.hpp>

#ifndef UNCERTAINTY_PLANNER_STATE_HPP
#define UNCERTAINTY_PLANNER_STATE_HPP

namespace uncertainty_planning_tools
{
    template<typename Configuration, typename ConfigSerializer, typename ConfigAlloc=std::allocator<Configuration>>
    class UncertaintyPlannerState
    {
    protected:

        typedef simple_robot_model_interface::SimpleRobotModelInterface<Configuration, ConfigAlloc> Robot;

        Configuration expectation_;
        Configuration command_;
        Eigen::VectorXd variances_;
        Eigen::VectorXd space_independent_variances_;
        std::vector<Configuration, ConfigAlloc> particles_;
        double step_size_;
        double parent_motion_Pfeasibility_;
        double raw_edge_Pfeasibility_;
        double effective_edge_Pfeasibility_;
        double reverse_edge_Pfeasibility_;
        double motion_Pfeasibility_;
        double variance_;
        double space_independent_variance_;
        double goal_Pfeasibility_;
        uint64_t state_id_;
        uint64_t transition_id_;
        uint64_t reverse_transition_id_;
        uint64_t split_id_;
        uint32_t attempt_count_;
        uint32_t reached_count_;
        uint32_t reverse_attempt_count_;
        uint32_t reverse_reached_count_;
        bool initialized_;
        bool has_particles_;
        bool use_for_nearest_neighbors_;
        bool action_outcome_is_nominally_independent_;

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static inline uint64_t Serialize(const UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc>& state, std::vector<uint8_t>& buffer)
        {
            return state.SerializeSelf(buffer);
        }

        static inline std::string GetConfigurationType()
        {
            return ConfigSerializer::TypeName();
        }

        inline uint64_t SerializeSelf(std::vector<uint8_t>& buffer) const
        {
            // Takes a state to serialize and a buffer to serialize into
            // Return number of bytes written to buffer
            assert(initialized_);
            const uint64_t start_buffer_size = buffer.size();
            // First thing we save is the qualified type id
            arc_helpers::SerializeFixedSizePOD<uint64_t>(std::numeric_limits<uint64_t>::max(), buffer);
            arc_helpers::SerializeString<char>(GetConfigurationType(), buffer);
            arc_helpers::SerializeFixedSizePOD<uint8_t>((uint8_t)has_particles_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint8_t>((uint8_t)use_for_nearest_neighbors_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint8_t>((uint8_t)action_outcome_is_nominally_independent_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint32_t>(attempt_count_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint32_t>(reached_count_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint32_t>(reverse_attempt_count_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint32_t>(reverse_reached_count_, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(step_size_, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(parent_motion_Pfeasibility_, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(effective_edge_Pfeasibility_, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(motion_Pfeasibility_, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(variance_, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(space_independent_variance_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint64_t>(state_id_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint64_t>(transition_id_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint64_t>(reverse_transition_id_, buffer);
            arc_helpers::SerializeFixedSizePOD<uint64_t>(split_id_, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(goal_Pfeasibility_, buffer);
            ConfigSerializer::Serialize(expectation_, buffer);
            ConfigSerializer::Serialize(command_, buffer);
            EigenHelpers::Serialize(variances_, buffer);
            EigenHelpers::Serialize(space_independent_variances_, buffer);
            // Serialize the particles
            arc_helpers::SerializeVector<Configuration, ConfigAlloc>(particles_, buffer, &ConfigSerializer::Serialize);
            // Figure out how many bytes we wrote
            const uint64_t end_buffer_size = buffer.size();
            const uint64_t bytes_written = end_buffer_size - start_buffer_size;
            return bytes_written;
        }

        static inline std::pair<UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc>, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc> temp_state;
            const uint64_t bytes_read = temp_state.DeserializeSelf(buffer, current);
            return std::make_pair(temp_state, bytes_read);
        }

        inline uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            uint64_t current_position = current;
            // First thing we load and check is the qualified type ID so we know that we're loading our state properly
            // First thing we save is the qualified type id
            const uint64_t reference_qualified_type_id_hash = std::numeric_limits<uint64_t>::max();
            const std::string reference_configuration_type = GetConfigurationType();
            const std::pair<uint64_t, uint64_t> deserialized_qualified_type_id_hash = arc_helpers::DeserializeFixedSizePOD<uint64_t>(buffer, current_position);
            const uint64_t qualified_type_id_hash = deserialized_qualified_type_id_hash.first;
            current_position += deserialized_qualified_type_id_hash.second;
            // Check types
            // If the file used the legacy type ID, we can't safely check it (std::hash is not required to be consistent across program executions!) so we warn the user and continue
            if (qualified_type_id_hash == reference_qualified_type_id_hash)
            {
                const std::pair<std::string, uint64_t> deserialized_configuration_type = arc_helpers::DeserializeString<char>(buffer, current_position);
                const std::string& configuration_type = deserialized_configuration_type.first;
                current_position += deserialized_configuration_type.second;
                if (configuration_type != reference_configuration_type)
                {
                    std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nLoaded configuration type: [" + configuration_type + "] does not match expected [" + reference_configuration_type + "]\nPROCEED WITH CAUTION - THIS MAY CAUSE UNDEFINED BEHAVIOR IN LOADING\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                }
            }
            else
            {
                std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nLoaded file uses old TypeId hash and cannot be safely checked\nPROCEED WITH CAUTION - THIS MAY CAUSE UNDEFINED BEHAVIOR IN LOADING\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            }
            // Load fixed size members
            const std::pair<uint8_t, uint64_t> deserialized_has_particles = arc_helpers::DeserializeFixedSizePOD<uint8_t>(buffer, current_position);
            has_particles_ = (bool)deserialized_has_particles.first;
            current_position += deserialized_has_particles.second;
            const std::pair<uint8_t, uint64_t> deserialized_use_for_nearest_neighbors = arc_helpers::DeserializeFixedSizePOD<uint8_t>(buffer, current_position);
            use_for_nearest_neighbors_ = (bool)deserialized_use_for_nearest_neighbors.first;
            current_position += deserialized_use_for_nearest_neighbors.second;
            const std::pair<uint8_t, uint64_t> deserialized_action_outcome_is_nominally_independent = arc_helpers::DeserializeFixedSizePOD<uint8_t>(buffer, current_position);
            action_outcome_is_nominally_independent_ = (bool)deserialized_action_outcome_is_nominally_independent.first;
            current_position += deserialized_action_outcome_is_nominally_independent.second;
            const std::pair<uint32_t, uint64_t> deserialized_attempt_count = arc_helpers::DeserializeFixedSizePOD<uint32_t>(buffer, current_position);
            attempt_count_ = deserialized_attempt_count.first;
            current_position += deserialized_attempt_count.second;
            const std::pair<uint32_t, uint64_t> deserialized_reached_count = arc_helpers::DeserializeFixedSizePOD<uint32_t>(buffer, current_position);
            reached_count_ = deserialized_reached_count.first;
            current_position += deserialized_reached_count.second;
            raw_edge_Pfeasibility_ = (double)reached_count_ / (double)attempt_count_;
            const std::pair<uint32_t, uint64_t> deserialized_reverse_attempt_count = arc_helpers::DeserializeFixedSizePOD<uint32_t>(buffer, current_position);
            reverse_attempt_count_ = deserialized_reverse_attempt_count.first;
            current_position += deserialized_reverse_attempt_count.second;
            const std::pair<uint32_t, uint64_t> deserialized_reverse_reached_count = arc_helpers::DeserializeFixedSizePOD<uint32_t>(buffer, current_position);
            reverse_reached_count_ = deserialized_reverse_reached_count.first;
            current_position += deserialized_reverse_reached_count.second;
            reverse_edge_Pfeasibility_ = (double)reverse_reached_count_ / (double)reverse_attempt_count_;
            const std::pair<double, uint64_t> deserialized_step_size = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            step_size_ = deserialized_step_size.first;
            current_position += deserialized_step_size.second;
            const std::pair<double, uint64_t> deserialized_parent_motion_Pfeasibility = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            parent_motion_Pfeasibility_ = deserialized_parent_motion_Pfeasibility.first;
            current_position += deserialized_parent_motion_Pfeasibility.second;
            const std::pair<double, uint64_t> deserialized_effective_edge_Pfeasibility = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            effective_edge_Pfeasibility_ = deserialized_effective_edge_Pfeasibility.first;
            current_position += deserialized_effective_edge_Pfeasibility.second;
            const std::pair<double, uint64_t> deserialized_motion_Pfeasibility = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            motion_Pfeasibility_ = deserialized_motion_Pfeasibility.first;
            current_position += deserialized_motion_Pfeasibility.second;
            const std::pair<double, uint64_t> deserialized_variance = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            variance_ = deserialized_variance.first;
            current_position += deserialized_variance.second;
            const std::pair<double, uint64_t> deserialized_space_independent_variance = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            space_independent_variance_ = deserialized_space_independent_variance.first;
            current_position += deserialized_space_independent_variance.second;
            const std::pair<uint64_t, uint64_t> deserialized_state_id = arc_helpers::DeserializeFixedSizePOD<uint64_t>(buffer, current_position);
            state_id_ = deserialized_state_id.first;
            current_position += deserialized_state_id.second;
            const std::pair<uint64_t, uint64_t> deserialized_transition_id = arc_helpers::DeserializeFixedSizePOD<uint64_t>(buffer, current_position);
            transition_id_ = deserialized_transition_id.first;
            current_position += deserialized_transition_id.second;
            const std::pair<uint64_t, uint64_t> deserialized_reverse_transition_id = arc_helpers::DeserializeFixedSizePOD<uint64_t>(buffer, current_position);
            reverse_transition_id_ = deserialized_reverse_transition_id.first;
            current_position += deserialized_reverse_transition_id.second;
            const std::pair<uint64_t, uint64_t> deserialized_split_id = arc_helpers::DeserializeFixedSizePOD<uint64_t>(buffer, current_position);
            split_id_ = deserialized_split_id.first;
            current_position += deserialized_split_id.second;
            const std::pair<double, uint64_t> deserialized_goal_Pfeasibility = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            goal_Pfeasibility_ = deserialized_goal_Pfeasibility.first;
            current_position += deserialized_goal_Pfeasibility.second;
            // Load the variable-sized components
            const std::pair<Configuration, uint64_t> deserialized_expectation = ConfigSerializer::Deserialize(buffer, current_position);
            expectation_ = deserialized_expectation.first;
            current_position += deserialized_expectation.second;
            const std::pair<Configuration, uint64_t> deserialized_command = ConfigSerializer::Deserialize(buffer, current_position);
            command_ = deserialized_command.first;
            current_position += deserialized_command.second;
            const std::pair<Eigen::VectorXd, uint64_t> deserialized_variances = EigenHelpers::Deserialize<Eigen::VectorXd>(buffer, current_position);
            variances_ = deserialized_variances.first;
            current_position += deserialized_variances.second;
            const std::pair<Eigen::VectorXd, uint64_t> deserialized_space_independent_variances = EigenHelpers::Deserialize<Eigen::VectorXd>(buffer, current_position);
            space_independent_variances_ = deserialized_space_independent_variances.first;
            current_position += deserialized_space_independent_variances.second;
            // Load the particles
            const std::pair<std::vector<Configuration, ConfigAlloc>, uint64_t> deserialized_particles = arc_helpers::DeserializeVector<Configuration, ConfigAlloc>(buffer, current_position, &ConfigSerializer::Deserialize);
            particles_ = deserialized_particles.first;
            current_position += deserialized_particles.second;
            // Initialize the state
            initialized_ = true;
            // Return how many bytes we read from the buffer
            const uint64_t bytes_read = current_position - current;
            return bytes_read;
        }

        inline UncertaintyPlannerState(const Configuration& expectation)
        {
            state_id_ = 0u;
            step_size_ = 0.0;
            expectation_ = expectation;
            particles_.clear();
            particles_.push_back(expectation_);
            variance_ = 0.0;
            variances_ = Eigen::VectorXd();
            space_independent_variance_ = 0.0;
            space_independent_variances_ = Eigen::VectorXd();
            attempt_count_ = 1u;
            reached_count_ = 1u;
            reverse_attempt_count_ = 1u;
            reverse_reached_count_ = 1u;
            parent_motion_Pfeasibility_ = 1.0;
            raw_edge_Pfeasibility_ = 1.0;
            effective_edge_Pfeasibility_ = 1.0;
            reverse_edge_Pfeasibility_ = 1.0;
            motion_Pfeasibility_ = 1.0;
            initialized_ = true;
            has_particles_ = true;
            use_for_nearest_neighbors_ = true;
            action_outcome_is_nominally_independent_ = true;
            command_ = expectation_;
            split_id_ = 0u;
            transition_id_ = 0;
            reverse_transition_id_ = 0;
            goal_Pfeasibility_ = 0.0;
        }

        inline UncertaintyPlannerState(const uint64_t state_id, const Configuration& particle, const uint32_t attempt_count, const uint32_t reached_count, const double effective_edge_Pfeasibility, const uint32_t reverse_attempt_count, const uint32_t reverse_reached_count, const double parent_motion_Pfeasibility, const double step_size, const Configuration& command, const uint64_t transition_id, const uint64_t reverse_transition_id, const uint64_t split_id, const bool action_outcome_is_nominally_independent)
        {
            state_id_ = state_id;
            step_size_ = step_size;
            expectation_ = particle;
            particles_.clear();
            particles_.push_back(expectation_);
            variance_ = 0.0;
            variances_ = Eigen::VectorXd();
            space_independent_variance_ = 0.0;
            space_independent_variances_ = Eigen::VectorXd();
            attempt_count_ = attempt_count;
            reached_count_ = reached_count;
            reverse_attempt_count_ = reverse_attempt_count;
            reverse_reached_count_ = reverse_reached_count;
            parent_motion_Pfeasibility_ = parent_motion_Pfeasibility;
            raw_edge_Pfeasibility_ = (double)reached_count_ / (double)attempt_count_;
            effective_edge_Pfeasibility_ = effective_edge_Pfeasibility;
            reverse_edge_Pfeasibility_ = (double)reverse_reached_count_ / (double)reverse_attempt_count_;
            motion_Pfeasibility_ = effective_edge_Pfeasibility_ * parent_motion_Pfeasibility_;
            initialized_ = true;
            has_particles_ = true;
            use_for_nearest_neighbors_ = true;
            action_outcome_is_nominally_independent_ = action_outcome_is_nominally_independent;
            command_ = command;
            transition_id_ = transition_id;
            reverse_transition_id_ = reverse_transition_id;
            split_id_ = split_id;
            goal_Pfeasibility_ = 0.0;
        }

        inline UncertaintyPlannerState(const uint64_t state_id, const std::vector<Configuration, ConfigAlloc>& particles, const uint32_t attempt_count, const uint32_t reached_count, const double effective_edge_Pfeasibility, const uint32_t reverse_attempt_count, const uint32_t reverse_reached_count, const double parent_motion_Pfeasibility, const double step_size, const Configuration& command, const uint64_t transition_id, const uint64_t reverse_transition_id, const uint64_t split_id, const bool action_outcome_is_nominally_independent)
        {
            state_id_ = state_id;
            step_size_ = step_size;
            particles_ = particles;
            attempt_count_ = attempt_count;
            reached_count_ = reached_count;
            reverse_attempt_count_ = reverse_attempt_count;
            reverse_reached_count_ = reverse_reached_count;
            parent_motion_Pfeasibility_ = parent_motion_Pfeasibility;
            raw_edge_Pfeasibility_ = (double)reached_count_ / (double)attempt_count_;
            effective_edge_Pfeasibility_ = effective_edge_Pfeasibility;
            reverse_edge_Pfeasibility_ = (double)reverse_reached_count_ / (double)reverse_attempt_count_;
            motion_Pfeasibility_ = effective_edge_Pfeasibility_ * parent_motion_Pfeasibility_;
            initialized_ = true;
            has_particles_ = true;
            use_for_nearest_neighbors_ = true;
            action_outcome_is_nominally_independent_ = action_outcome_is_nominally_independent;
            command_ = command;
            transition_id_ = transition_id;
            reverse_transition_id_ = reverse_transition_id;
            split_id_ = split_id;
            goal_Pfeasibility_ = 0.0;
        }

        inline std::pair<Configuration, std::pair<std::pair<double, Eigen::VectorXd>, std::pair<double, Eigen::VectorXd>>> UpdateStatistics(const std::shared_ptr<Robot>& robot_ptr)
        {
            std::function<Configuration(const std::vector<Configuration, ConfigAlloc>&)> average_fn = [&] (const std::vector<Configuration, ConfigAlloc>& particles) { return robot_ptr->AverageConfigurations(particles); };
            std::function<double(const Configuration&, const Configuration&)> distance_fn = [&] (const Configuration& config1, const Configuration& config2) { return robot_ptr->ComputeConfigurationDistance(config1, config2); };
            std::function<Eigen::VectorXd(const Configuration&, const Configuration&)> dim_distance_fn = [&] (const Configuration& config1, const Configuration& config2) { return robot_ptr->ComputePerDimensionConfigurationDistance(config1, config2); };
            expectation_ = ComputeExpectation(average_fn);
            variance_ = ComputeVariance(expectation_, distance_fn);
            variances_ = ComputeDirectionalVariance(expectation_, dim_distance_fn);
            space_independent_variance_ = ComputeSpaceIndependentVariance(expectation_, distance_fn, step_size_);
            space_independent_variances_ = ComputeSpaceIndependentDirectionalVariance(expectation_, dim_distance_fn, step_size_);
            return std::pair<Configuration, std::pair<std::pair<double, Eigen::VectorXd>, std::pair<double, Eigen::VectorXd>>>(expectation_, std::pair<std::pair<double, Eigen::VectorXd>, std::pair<double, Eigen::VectorXd>>(std::pair<double, Eigen::VectorXd>(variance_, variances_), std::pair<double, Eigen::VectorXd>(space_independent_variance_, space_independent_variances_)));
        }

        inline UncertaintyPlannerState() : goal_Pfeasibility_(0.0), state_id_(0), transition_id_(0), reverse_transition_id_(0), split_id_(0u), initialized_(false), has_particles_(false), use_for_nearest_neighbors_(false), action_outcome_is_nominally_independent_(false) {}

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

        inline bool IsActionOutcomeNominallyIndependent() const
        {
            return action_outcome_is_nominally_independent_;
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

        inline double GetRawEdgePfeasibility() const
        {
            return raw_edge_Pfeasibility_;
        }

        inline double GetEffectiveEdgePfeasibility() const
        {
            return effective_edge_Pfeasibility_;
        }

        inline double GetReverseEdgePfeasibility() const
        {
            return reverse_edge_Pfeasibility_;
        }

        inline double GetMotionPfeasibility() const
        {
            return motion_Pfeasibility_;
        }

        inline Configuration GetExpectation() const
        {
            return expectation_;
        }

        inline double GetGoalPfeasibility() const
        {
            return goal_Pfeasibility_;
        }

        inline std::pair<uint32_t, uint32_t> GetAttemptAndReachedCounts() const
        {
            return std::make_pair(attempt_count_, reached_count_);
        }

        inline std::pair<uint32_t, uint32_t> GetReverseAttemptAndReachedCounts() const
        {
            return std::make_pair(reverse_attempt_count_, reverse_reached_count_);
        }

        inline void UpdateAttemptAndReachedCounts(const uint32_t attempt_count, const uint32_t reached_count)
        {
            attempt_count_ = attempt_count;
            reached_count_ = reached_count;
            raw_edge_Pfeasibility_ = (double)reached_count_ / (double)attempt_count_;
            //std::cout << "Updated attempted/reached counts to " << attempt_count_ << "/" << reached_count_ << " with new edge P(feasibility) " << raw_edge_Pfeasibility_ << std::endl;
        }

        inline void UpdateReverseAttemptAndReachedCounts(const uint32_t reverse_attempt_count, const uint32_t reverse_reached_count)
        {
            reverse_attempt_count_ = reverse_attempt_count;
            reverse_reached_count_ = reverse_reached_count;
            reverse_edge_Pfeasibility_ = (double)reverse_reached_count_ / (double)reverse_attempt_count_;
            //std::cout << "Updated reverse attempted/reached counts to " << reverse_attempt_count_ << "/" << reverse_reached_count_ << " with new edge P(feasibility) " << reverse_edge_Pfeasibility_ << std::endl;
        }

        inline double SetEffectiveEdgePfeasibility(const double effective_edge_Pfeasibility)
        {
            effective_edge_Pfeasibility_ = effective_edge_Pfeasibility;
            motion_Pfeasibility_ = effective_edge_Pfeasibility_ * parent_motion_Pfeasibility_;
            return motion_Pfeasibility_;
        }

        inline void SetGoalPfeasibility(const double goal_Pfeasibility)
        {
            assert(goal_Pfeasibility >= -1.0); // We allow negative values to signifiy reverse edges!
            assert(goal_Pfeasibility <= 1.0);
            goal_Pfeasibility_ = goal_Pfeasibility;
        }

        inline void SetReverseEdgePfeasibility(const double reverse_edge_Pfeasibility)
        {
            assert(reverse_edge_Pfeasibility >= 0.0);
            assert(reverse_edge_Pfeasibility <= 1.0);
            reverse_edge_Pfeasibility_ = reverse_edge_Pfeasibility;
        }

        inline uint64_t GetStateId() const
        {
            return state_id_;
        }

        inline uint64_t GetTransitionId() const
        {
            return transition_id_;
        }

        inline uint64_t GetReverseTransitionId() const
        {
            return reverse_transition_id_;
        }

        inline uint64_t GetSplitId() const
        {
            return split_id_;
        }

        inline Configuration GetCommand() const
        {
            return command_;
        }

        inline void SetCommand(const Configuration& command)
        {
            command_ = command;
        }

        inline size_t GetNumParticles() const
        {
            return particles_.size();
        }

        inline std::pair<const std::vector<Configuration, ConfigAlloc>&, bool> GetParticlePositionsImmutable() const
        {
            if (has_particles_)
            {
                return std::pair<const std::vector<Configuration, ConfigAlloc>&, bool>(particles_, true);
            }
            else
            {
                return std::pair<const std::vector<Configuration, ConfigAlloc>&, bool>(particles_, false);
            }
        }

        inline std::pair<std::vector<Configuration, ConfigAlloc>&, bool> GetParticlePositionsMutable()
        {
            if (has_particles_)
            {
                return std::pair<std::vector<Configuration, ConfigAlloc>&, bool>(particles_, true);
            }
            else
            {
                return std::pair<std::vector<Configuration, ConfigAlloc>&, bool>(particles_, false);
            }
        }

        inline std::vector<Configuration, ConfigAlloc> CollectParticles(const size_t num_particles) const
        {
            if (particles_.size() == 0)
            {
                return std::vector<Configuration, ConfigAlloc>(num_particles, expectation_);
            }
            else if (particles_.size() == 1)
            {
                return std::vector<Configuration, ConfigAlloc>(num_particles, particles_[0]);
            }
            else
            {
                assert(num_particles == particles_.size());
                std::vector<Configuration, ConfigAlloc> resampled_particles = particles_;
                return resampled_particles;
            }
        }

        template<typename RNG>
        inline std::vector<Configuration, ConfigAlloc> ResampleParticles(const size_t num_particles, RNG& rng) const
        {
            if (particles_.size() == 0)
            {
                return std::vector<Configuration, ConfigAlloc>(num_particles, expectation_);
            }
            else if (particles_.size() == 1)
            {
                return std::vector<Configuration, ConfigAlloc>(num_particles, particles_[0]);
            }
            else
            {
                std::vector<Configuration, ConfigAlloc> resampled_particles(num_particles);
                double particle_probability = 1.0 / (double)particles_.size();
                std::uniform_int_distribution<size_t> resampling_distribution(0, particles_.size() - 1);
                std::uniform_real_distribution<double> importance_sampling_distribution(0.0, 1.0);
                size_t resampled = 0;
                while (resampled < num_particles)
                {
                    size_t random_index = resampling_distribution(rng);
                    const Configuration& random_particle = particles_[random_index];
                    if (importance_sampling_distribution(rng) < particle_probability)
                    {
                        resampled_particles[resampled] = random_particle;
                        resampled++;
                    }
                }
                return resampled_particles;
            }
        }

        inline double GetVariance() const
        {
            return variance_;
        }

        inline Eigen::VectorXd GetVariances() const
        {
            return variances_;
        }

        inline double GetSpaceIndependentVariance() const
        {
            return space_independent_variance_;
        }

        inline Eigen::VectorXd GetSpaceIndependentVariances() const
        {
            return space_independent_variances_;
        }

        inline Configuration ComputeExpectation(const std::function<Configuration(const std::vector<Configuration, ConfigAlloc>&)>& average_fn) const
        {
            if (particles_.size() == 0)
            {
                assert(!has_particles_);
                return expectation_;
            }
            else if (particles_.size() == 1)
            {
                return particles_[0];
            }
            else
            {
                return average_fn(particles_);
            }
        }

        inline double ComputeVariance(const Configuration& expectation, const std::function<double(const Configuration&, const Configuration&)>& distance_fn) const
        {
            if (particles_.size() == 0)
            {
                return 0.0;
            }
            else if (particles_.size() == 1)
            {
                return 0.0;
            }
            else
            {
                const double weight = 1.0 / (double)particles_.size();
                double var_sum = 0.0;
                for (size_t idx = 0; idx < particles_.size(); idx++)
                {
                    const double raw_distance = distance_fn(expectation, particles_[idx]);
                    const double squared_distance = pow(raw_distance, 2.0);
                    var_sum += (squared_distance * weight);
                }
                return var_sum;
            }
        }

        inline double ComputeSpaceIndependentVariance(const Configuration& expectation, const std::function<double(const Configuration&, const Configuration&)>& distance_fn, const double step_size) const
        {
            if (particles_.size() == 0)
            {
                return 0.0;
            }
            else if (particles_.size() == 1)
            {
                return 0.0;
            }
            else
            {
                const double weight = 1.0 / (double)particles_.size();
                double var_sum = 0.0;
                for (size_t idx = 0; idx < particles_.size(); idx++)
                {
                    const double raw_distance = distance_fn(expectation, particles_[idx]);
                    const double space_independent_distance = raw_distance / step_size;
                    const double squared_distance = pow(space_independent_distance, 2.0);
                    var_sum += (squared_distance * weight);
                }
                return var_sum;
            }
        }

        inline Eigen::VectorXd ComputeDirectionalVariance(const Configuration& expectation, const std::function<Eigen::VectorXd(const Configuration&, const Configuration&)>& dim_distance_fn) const
        {
            if (particles_.size() == 0)
            {
                return dim_distance_fn(expectation, expectation);
            }
            else if (particles_.size() == 1)
            {
                return dim_distance_fn(particles_[0], particles_[0]);
            }
            else
            {
                const double weight = 1.0 / (double)particles_.size();
                Eigen::VectorXd variances;
                for (size_t idx = 0; idx < particles_.size(); idx++)
                {
                    const Eigen::VectorXd error = dim_distance_fn(expectation, particles_[idx]);
                    const Eigen::VectorXd squared_error = error.cwiseProduct(error);
                    const Eigen::VectorXd weighted_squared_error = squared_error * weight;
                    if (variances.size() != weighted_squared_error.size())
                    {
                        variances.setZero(weighted_squared_error.size());
                    }
                    variances += weighted_squared_error;
                }
                return variances;
            }
        }

        inline Eigen::VectorXd ComputeSpaceIndependentDirectionalVariance(const Configuration& expectation, const std::function<Eigen::VectorXd(const Configuration&, const Configuration&)>& dim_distance_fn, const double step_size) const
        {
            if (particles_.size() == 0)
            {
                return dim_distance_fn(expectation, expectation);
            }
            else if (particles_.size() == 1)
            {
                return dim_distance_fn(particles_[0], particles_[0]);
            }
            else
            {
                const double weight = 1.0 / (double)particles_.size();
                Eigen::VectorXd variances;
                for (size_t idx = 0; idx < particles_.size(); idx++)
                {
                    const Eigen::VectorXd error = dim_distance_fn(expectation, particles_[idx]);
                    const Eigen::VectorXd space_independent_error = error / step_size;
                    const Eigen::VectorXd squared_error = space_independent_error.cwiseProduct(space_independent_error);
                    const Eigen::VectorXd weighted_squared_error = squared_error * weight;
                    if (variances.size() != weighted_squared_error.size())
                    {
                        variances.setZero(weighted_squared_error.size());
                    }
                    variances += weighted_squared_error;
                }
                return variances;
            }
        }

        inline std::string Print() const
        {
            std::ostringstream strm;
            strm << "Uncertainty Planner State (Configuration Type: " << GetConfigurationType() << ") - Expectation: " << PrettyPrint::PrettyPrint(GetExpectation()) << " Command: " << PrettyPrint::PrettyPrint(GetCommand()) << " Action outcome is nominally independent [" << PrettyPrint::PrettyPrint(action_outcome_is_nominally_independent_) << "] Variance: " << GetVariance() << " Space-independent Variance: " << GetSpaceIndependentVariance() << " Raw Pfeasibility(parent->this): " << GetRawEdgePfeasibility() << " Effective Pfeasibility(parent->this): " << GetEffectiveEdgePfeasibility() << " Raw Pfeasibility(this->parent): " << GetReverseEdgePfeasibility() << " Pfeasibility(start->this): " << GetMotionPfeasibility();
            return strm.str();
        }
    };
}

template<typename Robot, typename Configuration, typename ConfigSerializer, typename ConfigAlloc=std::allocator<Configuration>>
std::ostream& operator<<(std::ostream& strm, const uncertainty_planning_tools::UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc>& state)
{
    strm << state.Print();
    return strm;
}

#endif // UNCERTAINTY_PLANNER_STATE_HPP
