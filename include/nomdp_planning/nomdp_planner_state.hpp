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

#ifndef NOMDP_PLANNER_STATE_HPP
#define NOMDP_PLANNER_STATE_HPP

namespace nomdp_planning_tools
{
    template<typename Configuration, typename ConfigSerializer, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
    class NomdpPlannerState
    {
    protected:

        bool initialized_;
        bool has_particles_;
        bool use_for_nearest_neighbors_;
        double step_size_;
        double parent_motion_Pfeasibility_;
        double raw_edge_Pfeasibility_;
        double effective_edge_Pfeasibility_;
        double reverse_edge_Pfeasibility_;
        double motion_Pfeasibility_;
        double variance_;
        double space_independent_variance_;
        uint64_t state_id_;
        uint64_t transition_id_;
        uint64_t split_id_;
        double goal_Pfeasibility_;
        Configuration expectation_;
        Configuration command_;
        Eigen::VectorXd variances_;
        Eigen::VectorXd space_independent_variances_;
        std::vector<Configuration, ConfigAlloc> particles_;
        AverageFn average_fn_;
        DistanceFn distance_fn_;
        DimDistanceFn dim_distance_fn_;

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static inline std::string GetQualifiedTypeID()
        {
            const std::string qualified_type_id = "NomdpPlannerState<" + ConfigSerializer::TypeName() + "_" + AverageFn::TypeName() + "_" + DistanceFn::TypeName() + "_" + DimDistanceFn::TypeName() + ">";
            return qualified_type_id;
        }

        static inline uint64_t GetQualifiedTypeIDHash()
        {
            return (uint64_t)std::hash<std::string>()(GetQualifiedTypeID());
        }

        static inline uint64_t Serialize(const NomdpPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>& state, std::vector<uint8_t>& buffer)
        {
            return state.SerializeSelf(buffer);
        }

        inline uint64_t SerializeSelf(std::vector<uint8_t>& buffer) const
        {
            // Takes a state to serialize and a buffer to serialize into
            // Return number of bytes written to buffer
            assert(initialized_);
            const uint64_t start_buffer_size = buffer.size();
            // First thing we save is the qualified type id
            //std::cout << "Serializing self with QualifiedTypeID: [" << GetQualifiedTypeID() << "] and QualifiedTypeID hash: [" << GetQualifiedTypeIDHash() << "]" << std::endl;
            const uint64_t qualified_type_id_hash = GetQualifiedTypeIDHash();
            // Now build the fixed-size serialized parts
            const uint8_t has_particles_serialized = (uint8_t)(has_particles_);
            const uint8_t use_for_nearest_neighbors_serialized = (uint8_t)(use_for_nearest_neighbors_);
            const uint64_t fixed_size_buffer_size = sizeof(qualified_type_id_hash)
                                                    + sizeof(has_particles_serialized)
                                                    + sizeof(use_for_nearest_neighbors_serialized)
                                                    + sizeof(step_size_)
                                                    + sizeof(parent_motion_Pfeasibility_)
                                                    + sizeof(raw_edge_Pfeasibility_)
                                                    + sizeof(effective_edge_Pfeasibility_)
                                                    + sizeof(reverse_edge_Pfeasibility_)
                                                    + sizeof(motion_Pfeasibility_)
                                                    + sizeof(variance_)
                                                    + sizeof(space_independent_variance_)
                                                    + sizeof(state_id_)
                                                    + sizeof(transition_id_)
                                                    + sizeof(split_id_)
                                                    + sizeof(goal_Pfeasibility_);
            std::vector<uint8_t> temp_buffer(fixed_size_buffer_size, 0x00);
            // Copy everything over
            uint64_t current_position = 0u;
            memcpy(&temp_buffer[current_position], &qualified_type_id_hash, sizeof(qualified_type_id_hash));
            current_position += sizeof(qualified_type_id_hash);
            memcpy(&temp_buffer[current_position], &has_particles_serialized, sizeof(has_particles_serialized));
            current_position += sizeof(has_particles_serialized);
            memcpy(&temp_buffer[current_position], &use_for_nearest_neighbors_serialized, sizeof(use_for_nearest_neighbors_serialized));
            current_position += sizeof(use_for_nearest_neighbors_serialized);
            memcpy(&temp_buffer[current_position], &step_size_, sizeof(step_size_));
            current_position += sizeof(step_size_);
            memcpy(&temp_buffer[current_position], &parent_motion_Pfeasibility_, sizeof(parent_motion_Pfeasibility_));
            current_position += sizeof(parent_motion_Pfeasibility_);
            memcpy(&temp_buffer[current_position], &raw_edge_Pfeasibility_, sizeof(raw_edge_Pfeasibility_));
            current_position += sizeof(raw_edge_Pfeasibility_);
            memcpy(&temp_buffer[current_position], &effective_edge_Pfeasibility_, sizeof(effective_edge_Pfeasibility_));
            current_position += sizeof(effective_edge_Pfeasibility_);
            memcpy(&temp_buffer[current_position], &reverse_edge_Pfeasibility_, sizeof(reverse_edge_Pfeasibility_));
            current_position += sizeof(reverse_edge_Pfeasibility_);
            memcpy(&temp_buffer[current_position], &motion_Pfeasibility_, sizeof(motion_Pfeasibility_));
            current_position += sizeof(motion_Pfeasibility_);
            memcpy(&temp_buffer[current_position], &variance_, sizeof(variance_));
            current_position += sizeof(variance_);
            memcpy(&temp_buffer[current_position], &space_independent_variance_, sizeof(space_independent_variance_));
            current_position += sizeof(space_independent_variance_);
            memcpy(&temp_buffer[current_position], &state_id_, sizeof(state_id_));
            current_position += sizeof(state_id_);
            memcpy(&temp_buffer[current_position], &transition_id_, sizeof(transition_id_));
            current_position += sizeof(transition_id_);
            memcpy(&temp_buffer[current_position], &split_id_, sizeof(split_id_));
            current_position += sizeof(split_id_);
            memcpy(&temp_buffer[current_position], &goal_Pfeasibility_, sizeof(goal_Pfeasibility_));
            current_position += sizeof(goal_Pfeasibility_);
            // Serialize the variable-sized parts
            ConfigSerializer::Serialize(expectation_, temp_buffer);
            ConfigSerializer::Serialize(command_, temp_buffer);
            EigenHelpers::Serialize(variances_, temp_buffer);
            EigenHelpers::Serialize(space_independent_variances_, temp_buffer);
            // Serialize the particles
            arc_helpers::SerializeVector<Configuration, ConfigAlloc>(particles_, temp_buffer, &ConfigSerializer::Serialize);
            // Move into the buffer
            buffer.insert(buffer.end(), temp_buffer.begin(), temp_buffer.end());
            // Figure out how many bytes we wrote
            const uint64_t end_buffer_size = buffer.size();
            const uint64_t bytes_written = end_buffer_size - start_buffer_size;
            return bytes_written;
        }

        static inline std::pair<NomdpPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            NomdpPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc> temp_state;
            const uint64_t bytes_read = temp_state.DeserializeSelf(buffer, current);
            return std::make_pair(temp_state, bytes_read);
        }

        inline uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            // First thing we load and check is the qualified type ID so we know that we're loading our state properly
            // First thing we save is the qualified type id
            const uint64_t reference_qualified_type_id_hash = GetQualifiedTypeIDHash();
            // Now build the fixed-size serialized parts
            uint8_t has_particles_serialized = 0x00;
            uint8_t use_for_nearest_neighbors_serialized = 0x00;
            const uint64_t fixed_size_buffer_size = sizeof(reference_qualified_type_id_hash)
                                                    + sizeof(has_particles_serialized)
                                                    + sizeof(use_for_nearest_neighbors_serialized)
                                                    + sizeof(step_size_)
                                                    + sizeof(parent_motion_Pfeasibility_)
                                                    + sizeof(raw_edge_Pfeasibility_)
                                                    + sizeof(effective_edge_Pfeasibility_)
                                                    + sizeof(reverse_edge_Pfeasibility_)
                                                    + sizeof(motion_Pfeasibility_)
                                                    + sizeof(variance_)
                                                    + sizeof(space_independent_variance_)
                                                    + sizeof(state_id_)
                                                    + sizeof(transition_id_)
                                                    + sizeof(split_id_)
                                                    + sizeof(goal_Pfeasibility_);
            assert(current < buffer.size());
            assert((current + fixed_size_buffer_size) <= buffer.size());
            // Load the stored type ID header
            uint64_t current_position = current;
            uint64_t qualified_type_id_hash = 0u;
            memcpy(&qualified_type_id_hash, &buffer[current_position], sizeof(qualified_type_id_hash));
            current_position += sizeof(qualified_type_id_hash);
            // Check types
            //std::cout << "Reference TypeID hash: " << reference_qualified_type_id_hash << std::endl;
            //std::cout << "Loaded TypeID hash: " << qualified_type_id_hash << std::endl;
            assert(qualified_type_id_hash == reference_qualified_type_id_hash);
            // Copy the remaining fixed-size elements
            memcpy(&has_particles_serialized, &buffer[current_position], sizeof(has_particles_serialized));
            current_position += sizeof(has_particles_serialized);
            has_particles_ = (bool)has_particles_serialized;
            memcpy(&use_for_nearest_neighbors_serialized, &buffer[current_position], sizeof(use_for_nearest_neighbors_serialized));
            current_position += sizeof(use_for_nearest_neighbors_serialized);
            use_for_nearest_neighbors_ = (bool)use_for_nearest_neighbors_serialized;
            memcpy(&step_size_, &buffer[current_position], sizeof(step_size_));
            current_position += sizeof(step_size_);
            memcpy(&parent_motion_Pfeasibility_, &buffer[current_position], sizeof(parent_motion_Pfeasibility_));
            current_position += sizeof(parent_motion_Pfeasibility_);
            memcpy(&raw_edge_Pfeasibility_, &buffer[current_position], sizeof(raw_edge_Pfeasibility_));
            current_position += sizeof(raw_edge_Pfeasibility_);
            memcpy(&effective_edge_Pfeasibility_, &buffer[current_position], sizeof(effective_edge_Pfeasibility_));
            current_position += sizeof(effective_edge_Pfeasibility_);
            memcpy(&reverse_edge_Pfeasibility_, &buffer[current_position], sizeof(reverse_edge_Pfeasibility_));
            current_position += sizeof(reverse_edge_Pfeasibility_);
            memcpy(&motion_Pfeasibility_, &buffer[current_position], sizeof(motion_Pfeasibility_));
            current_position += sizeof(motion_Pfeasibility_);
            memcpy(&variance_, &buffer[current_position], sizeof(variance_));
            current_position += sizeof(variance_);
            memcpy(&space_independent_variance_, &buffer[current_position], sizeof(space_independent_variance_));
            current_position += sizeof(space_independent_variance_);
            memcpy(&state_id_, &buffer[current_position], sizeof(state_id_));
            current_position += sizeof(state_id_);
            memcpy(&transition_id_, &buffer[current_position], sizeof(transition_id_));
            current_position += sizeof(transition_id_);
            memcpy(&split_id_, &buffer[current_position], sizeof(split_id_));
            current_position += sizeof(split_id_);
            memcpy(&goal_Pfeasibility_, &buffer[current_position], sizeof(goal_Pfeasibility_));
            current_position += sizeof(goal_Pfeasibility_);
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

        inline NomdpPlannerState(const Configuration& expectation)
        {
            state_id_ = 0u;
            step_size_ = 0.0;
            expectation_ = expectation;
            particles_ = std::vector<Configuration, ConfigAlloc>{expectation_};
            UpdateStatistics();
            parent_motion_Pfeasibility_ = 1.0;
            raw_edge_Pfeasibility_ = 1.0;
            effective_edge_Pfeasibility_ = 1.0;
            reverse_edge_Pfeasibility_ = 1.0;
            motion_Pfeasibility_ = 1.0;
            initialized_ = true;
            has_particles_ = false;
            use_for_nearest_neighbors_ = true;
            split_id_ = 0u;
            transition_id_ = 0;
            goal_Pfeasibility_ = 0.0;
        }

        inline NomdpPlannerState(const uint64_t state_id, const std::vector<Configuration, ConfigAlloc>& particles, const double raw_edge_Pfeasibility, const double effective_edge_Pfeasibility, const double reverse_edge_Pfeasibility, const double parent_motion_Pfeasibility, const double step_size, const Configuration& command, const uint64_t transition_id, const uint64_t split_id)
        {
            state_id_ = state_id;
            step_size_ = step_size;
            particles_ = particles;
            UpdateStatistics();
            parent_motion_Pfeasibility_ = parent_motion_Pfeasibility;
            raw_edge_Pfeasibility_ = raw_edge_Pfeasibility;
            effective_edge_Pfeasibility_ = effective_edge_Pfeasibility;
            reverse_edge_Pfeasibility_ = reverse_edge_Pfeasibility;
            motion_Pfeasibility_ = effective_edge_Pfeasibility_ * parent_motion_Pfeasibility_;
            initialized_ = true;
            has_particles_ = true;
            use_for_nearest_neighbors_ = true;
            command_ = command;
            transition_id_ = transition_id;
            split_id_ = split_id;
            goal_Pfeasibility_ = 0.0;
        }

        inline NomdpPlannerState() : initialized_(false), has_particles_(false), use_for_nearest_neighbors_(false), state_id_(0), transition_id_(0), split_id_(0u), goal_Pfeasibility_(0.0) {}

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

        inline Configuration ComputeExpectation() const
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
                return average_fn_(particles_);
            }
        }

        inline double ComputeVariance(const Configuration& expectation) const
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
                double weight = 1.0 / (double)particles_.size();
                double var_sum = 0.0;
                for (size_t idx = 0; idx < particles_.size(); idx++)
                {
                    double raw_distance = distance_fn_(expectation, particles_[idx]);
                    double squared_distance = pow(raw_distance, 2.0);
                    var_sum += (squared_distance * weight);
                }
                return var_sum;
            }
        }

        inline double ComputeSpaceIndependentVariance(const Configuration& expectation, const double step_size) const
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
                double weight = 1.0 / (double)particles_.size();
                double var_sum = 0.0;
                for (size_t idx = 0; idx < particles_.size(); idx++)
                {
                    double raw_distance = distance_fn_(expectation, particles_[idx]);
                    double space_independent_distance = raw_distance / step_size;
                    double squared_distance = pow(space_independent_distance, 2.0);
                    var_sum += (squared_distance * weight);
                }
                return var_sum;
            }
        }

        inline Eigen::VectorXd ComputeDirectionalVariance(const Configuration& expectation) const
        {
            if (particles_.size() == 0)
            {
                return dim_distance_fn_(expectation, expectation);
            }
            else if (particles_.size() == 1)
            {
                return dim_distance_fn_(particles_[0], particles_[0]);
            }
            else
            {
                double weight = 1.0 / (double)particles_.size();
                Eigen::VectorXd variances;
                for (size_t idx = 0; idx < particles_.size(); idx++)
                {
                    Eigen::VectorXd error = dim_distance_fn_(expectation, particles_[idx]);
                    Eigen::VectorXd squared_error = error.cwiseProduct(error);
                    Eigen::VectorXd weighted_squared_error = squared_error * weight;
                    if (variances.size() != weighted_squared_error.size())
                    {
                        variances.setZero(weighted_squared_error.size());
                    }
                    variances += weighted_squared_error;
                }
                return variances;
            }
        }

        inline Eigen::VectorXd ComputeSpaceIndependentDirectionalVariance(const Configuration& expectation, const double step_size) const
        {
            if (particles_.size() == 0)
            {
                return dim_distance_fn_(expectation, expectation);
            }
            else if (particles_.size() == 1)
            {
                return dim_distance_fn_(particles_[0], particles_[0]);
            }
            else
            {
                double weight = 1.0 / (double)particles_.size();
                Eigen::VectorXd variances;
                for (size_t idx = 0; idx < particles_.size(); idx++)
                {
                    Eigen::VectorXd error = dim_distance_fn_(expectation, particles_[idx]);
                    Eigen::VectorXd space_independent_error = error / step_size;
                    Eigen::VectorXd squared_error = space_independent_error.cwiseProduct(space_independent_error);
                    Eigen::VectorXd weighted_squared_error = squared_error * weight;
                    if (variances.size() != weighted_squared_error.size())
                    {
                        variances.setZero(weighted_squared_error.size());
                    }
                    variances += weighted_squared_error;
                }
                return variances;
            }
        }

        inline std::pair<Configuration, std::pair<std::pair<double, Eigen::VectorXd>, std::pair<double, Eigen::VectorXd>>> UpdateStatistics()
        {
            expectation_ = ComputeExpectation();
            variance_ = ComputeVariance(expectation_);
            variances_ = ComputeDirectionalVariance(expectation_);
            space_independent_variance_ = ComputeSpaceIndependentVariance(expectation_, step_size_);
            space_independent_variances_ = ComputeSpaceIndependentDirectionalVariance(expectation_, step_size_);
            return std::pair<Configuration, std::pair<std::pair<double, Eigen::VectorXd>, std::pair<double, Eigen::VectorXd>>>(expectation_, std::pair<std::pair<double, Eigen::VectorXd>, std::pair<double, Eigen::VectorXd>>(std::pair<double, Eigen::VectorXd>(variance_, variances_), std::pair<double, Eigen::VectorXd>(space_independent_variance_, space_independent_variances_)));
        }
    };
}

template<typename Configuration, typename ConfigSerializer, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
std::ostream& operator<<(std::ostream& strm, const nomdp_planning_tools::NomdpPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>& state)
{
    strm << "Nomdp Planner State (QualifiedTypeID: " << state.GetQualifiedTypeID() << ") - Expectation: " << PrettyPrint::PrettyPrint(state.GetExpectation()) << " Command: " << PrettyPrint::PrettyPrint(state.GetCommand()) << " Variance: " << state.GetVariance() << " Space-independent Variance: " << state.GetSpaceIndependentVariance() << " Raw Pfeasibility(parent->this): " << state.GetRawEdgePfeasibility() << " Effective Pfeasibility(parent->this): " << state.GetEffectiveEdgePfeasibility() << " Raw Pfeasibility(this->parent): " << state.GetReverseEdgePfeasibility() << " Pfeasibility(start->this): " << state.GetMotionPfeasibility();
    return strm;
}

#endif // NOMDP_PLANNER_STATE_HPP
