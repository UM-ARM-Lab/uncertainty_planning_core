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

#ifndef NOMDP_PLANNER_STATE_HPP
#define NOMDP_PLANNER_STATE_HPP

namespace nomdp_planning_tools
{
    template<typename Configuration, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
    class NomdpPlannerState
    {
    protected:

        bool initialized_;
        bool has_particles_;
        bool use_for_nearest_neighbors_;
        double step_size_;
        double edge_Pfeasibility_;
        double reverse_edge_Pfeasibility_;
        double motion_Pfeasibility_;
        double variance_;
        double space_independent_variance_;
        u_int64_t state_id_;
        u_int64_t transition_id_;
        u_int64_t split_id_;
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

        inline NomdpPlannerState(const Configuration& expectation)
        {
            state_id_ = 0u;
            step_size_ = 0.0;
            expectation_ = expectation;
            particles_ = std::vector<Configuration, ConfigAlloc>{expectation_};
            UpdateStatistics();
            edge_Pfeasibility_ = 1.0;
            reverse_edge_Pfeasibility_ = 1.0;
            motion_Pfeasibility_ = 1.0;
            initialized_ = true;
            has_particles_ = false;
            use_for_nearest_neighbors_ = true;
            split_id_ = 0u;
            transition_id_ = 0;
            goal_Pfeasibility_ = 0.0;
        }

        inline NomdpPlannerState(const u_int64_t state_id, const std::vector<Configuration, ConfigAlloc>& particles, const double edge_Pfeasibility, const double reverse_edge_Pfeasibility, const double parent_motion_Pfeasibility, const double step_size, const Configuration& command, const u_int64_t transition_id, const u_int64_t split_id)
        {
            state_id_ = state_id;
            step_size_ = step_size;
            particles_ = particles;
            UpdateStatistics();
            edge_Pfeasibility_ = edge_Pfeasibility;
            reverse_edge_Pfeasibility_ = reverse_edge_Pfeasibility;
            motion_Pfeasibility_ = edge_Pfeasibility_ * parent_motion_Pfeasibility;
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

        inline double GetEdgePfeasibility() const
        {
            return edge_Pfeasibility_;
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

        inline u_int64_t GetStateId() const
        {
            return state_id_;
        }

        inline u_int64_t GetTransitionId() const
        {
            return transition_id_;
        }

        inline u_int64_t GetSplitId() const
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

template<typename Configuration, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
std::ostream& operator<<(std::ostream& strm, const nomdp_planning_tools::NomdpPlannerState<Configuration, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>& state)
{
    strm << "Nomdp Planner State - Expectation: " << PrettyPrint::PrettyPrint(state.GetExpectation()) << " Command: " << PrettyPrint::PrettyPrint(state.GetCommand()) << " Variance: " << state.GetVariance() << " Space-independent Variance: " << state.GetSpaceIndependentVariance() << " Pfeasibility(parent->this): " << state.GetEdgePfeasibility() << " Pfeasibility(start->this): " << state.GetMotionPfeasibility();
    return strm;
}

#endif // NOMDP_PLANNER_STATE_HPP
