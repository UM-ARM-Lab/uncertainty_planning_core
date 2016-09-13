#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <functional>
#include <random>
#include <Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include "arc_utilities/eigen_helpers.hpp"
#include "arc_utilities/eigen_helpers_conversions.hpp"
#include "arc_utilities/pretty_print.hpp"
#include "arc_utilities/voxel_grid.hpp"
#include "arc_utilities/simple_rrt_planner.hpp"
#include "uncertainty_planning_core/simple_pid_controller.hpp"
#include "uncertainty_planning_core/simple_uncertainty_models.hpp"
#include "uncertainty_planning_core/uncertainty_contact_planning.hpp"
#include "uncertainty_planning_core/simplelinked_robot_helpers.hpp"

#ifndef BAXTER_ACTUATOR_HELPERS_HPP
#define BAXTER_ACTUATOR_HELPERS_HPP

namespace baxter_actuator_helpers
{
    class SimpleLinkedDimDistancer
    {
    public:

        Eigen::VectorXd operator()(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2) const
        {
            return Distance(q1, q2);
        }

        static Eigen::VectorXd Distance(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2)
        {
            return RawDistance(q1, q2).cwiseAbs();
        }

        static double RawJointDistance(const simplelinked_robot_helpers::SimpleJointModel& j1, const simplelinked_robot_helpers::SimpleJointModel& j2)
        {
            assert(j1.IsContinuous() == j2.IsContinuous());
            return j1.SignedDistance(j1.GetValue(), j2.GetValue());
        }

        static Eigen::VectorXd RawDistance(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2)
        {
            const std::vector<double> max_velocities = {0.27, 0.27, 0.27, 0.27, 0.3, 0.3, 0.5};
            std::vector<double> distance_weights(max_velocities.size(), 0.0);
            for (size_t idx = 0; idx < max_velocities.size(); idx++)
            {
                distance_weights[idx] = 1.0 / max_velocities[idx];
            }
            assert(q1.size() == q2.size());
            assert(q1.size() == 7);
            Eigen::VectorXd distances = Eigen::VectorXd::Zero((ssize_t)(q1.size()));
            for (size_t idx = 0; idx < q1.size(); idx++)
            {
                const simplelinked_robot_helpers::SimpleJointModel& j1 = q1[idx];
                const simplelinked_robot_helpers::SimpleJointModel& j2 = q2[idx];
                const double joint_weight = distance_weights[idx];
                distances((int64_t)idx) = RawJointDistance(j1, j2) * joint_weight;
            }
            return distances;
        }

        static inline std::string TypeName()
        {
            return std::string("BaxterSimpleLinkedDimDistancer");
        }
    };

    class SimpleLinkedDistancer
    {
    public:

        double operator()(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2) const
        {
            return Distance(q1, q2);
        }

        static double Distance(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2)
        {
            const Eigen::VectorXd dim_distances = SimpleLinkedDimDistancer::Distance(q1, q2);
            return dim_distances.norm();
        }

        static inline std::string TypeName()
        {
            return std::string("BaxterSimpleLinkedDistancer");
        }
    };

    typedef std::vector<std::pair<std::pair<double, double>, std::vector<double>>> JointUncertaintySampleModel;

    inline std::vector<double> DownsampleBin(const std::vector<double>& raw_bin, const uint32_t downsampled_size)
    {
        std::vector<double> downsampled_bin(downsampled_size, 0.0);
        for (uint32_t idx = 0; idx < downsampled_size; idx++)
        {
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<size_t> pick_dist(0, raw_bin.size() - 1);
            const size_t pick_idx = pick_dist(rng);
            const double bin_item = raw_bin[pick_idx];
            downsampled_bin[idx] = bin_item;
        }
        return downsampled_bin;
    }

    inline size_t GetMatchingBin(const JointUncertaintySampleModel& bins, const double commanded_velocity)
    {
        for (size_t idx = 0; idx < bins.size(); idx++)
        {
            const std::pair<std::pair<double, double>, std::vector<double>>& bin = bins[idx];
            const std::pair<double, double>& bin_bounds = bin.first;
            //std::cout << PrettyPrint::PrettyPrint(bin_bounds) << std::endl;
            if (commanded_velocity >= bin_bounds.first && commanded_velocity <= bin_bounds.second)
            {
                return idx;
            }
        }
        std::cerr << "Value " << commanded_velocity << " is not in any bin" << std::endl;
        assert(false);
    }

    inline std::shared_ptr<JointUncertaintySampleModel> LoadModel(const std::string& model_file, const double actuator_limit, const uint32_t num_bins, const uint32_t bin_elements)
    {
        // Read the CSV file
        std::ifstream indata;
        indata.open(model_file);
        std::string line;
        std::vector<std::pair<double, double>> raw_data;
        while (getline(indata, line))
        {
            std::stringstream lineStream(line);
            std::string cell;
            std::vector<double> line_data;
            while (std::getline(lineStream, cell, ','))
            {
                //Process cell
                const double val = std::stod(cell);
                line_data.push_back(val);
            }
            assert(line_data.size() == 2);
            const std::pair<double, double> line_pair(line_data[0], line_data[1]);
            raw_data.push_back(line_pair);
        }
        indata.close();
        std::cout << "Loaded " << raw_data.size() << " data points from " << model_file << std::endl;
        // Make the empty bins
        std::shared_ptr<JointUncertaintySampleModel> bins(new JointUncertaintySampleModel());
        const double bin_size = (actuator_limit * 2.0) / (double)num_bins;
        double previous_bin_upper = -actuator_limit;
        for (size_t idx = 0; idx < num_bins; idx++)
        {
            double bin_lower = previous_bin_upper;
            if (idx == 0)
            {
                bin_lower = -INFINITY;
            }
            // Make sure the last bin's bounds are right
            double bin_upper = previous_bin_upper + bin_size;
            if (idx >= (num_bins - 1))
            {
                bin_upper = INFINITY;
            }
            previous_bin_upper = bin_upper;
            const std::pair<double, double> bin_bounds(bin_lower, bin_upper);
            //std::cout << "Bin bounds " << PrettyPrint::PrettyPrint(bin_bounds) << std::endl;
            bins->push_back(std::make_pair(bin_bounds, std::vector<double>()));
        }
        //std::cout << "Made " << bins.size() << " empty bins" << std::endl;
        // Put data in bins
        for (size_t idx = 0; idx < raw_data.size(); idx++)
        {
            const std::pair<double, double>& data_pair = raw_data[idx];
            const double commanded_velocity = data_pair.first;
            const double velocity_error = data_pair.second;
            const size_t bin_idx = GetMatchingBin(*bins, commanded_velocity);
            (*bins)[bin_idx].second.push_back(velocity_error);
        }
        std::cout << "Loaded " << raw_data.size() << " data points into " << bins->size() << " bins" << std::endl;
        for (size_t idx = 0; idx < bins->size(); idx++)
        {
            std::pair<std::pair<double, double>, std::vector<double>>& bin_contents = (*bins)[idx];
            std::vector<double>& bin_items = bin_contents.second;
            bin_contents.second = DownsampleBin(bin_items, bin_elements);
        }
        std::cout << "Downsampled each bin to " << bin_elements << " examples" << std::endl;
        // Return the model
        return bins;
    }

    class BaxterUncertaintySamplesJointActuatorModel
    {
        bool initialized_;
        double actuator_limit_;
        std::shared_ptr<JointUncertaintySampleModel> model_ptr_;

        template<typename RNG>
        inline double GetNoiseValue(const double commanded_velocity, RNG rng) const
        {
            if (model_ptr_)
            {
                const size_t bin_idx = GetMatchingBin(*model_ptr_, commanded_velocity);
                const std::vector<double>& best_match_bin = (*model_ptr_)[bin_idx].second;
                std::uniform_int_distribution<size_t> pick_dist(0, best_match_bin.size() - 1);
                const size_t pick_idx = pick_dist(rng);
                const double noise = best_match_bin[pick_idx];
                return noise;
            }
            else
            {
                UNUSED(rng);
                return 0.0;
            }
        }

    public:

        BaxterUncertaintySamplesJointActuatorModel(const std::shared_ptr<JointUncertaintySampleModel> model_ptr, const double max_velocity) : initialized_(true), actuator_limit_(fabs(max_velocity)), model_ptr_(model_ptr) {}

        BaxterUncertaintySamplesJointActuatorModel(const double max_velocity) : initialized_(true), actuator_limit_(fabs(max_velocity)) {}

        BaxterUncertaintySamplesJointActuatorModel() : initialized_(true), actuator_limit_(0.0) {}

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline double GetControlValue(const double control_input) const
        {
            assert(std::isnan(control_input) == false);
            assert(std::isinf(control_input) == false);
            double real_control_input = std::min(actuator_limit_, control_input);
            real_control_input = std::max(-actuator_limit_, real_control_input);
            return real_control_input;
        }

        template<typename RNG>
        inline double GetControlValue(const double control_input, RNG& rng) const
        {
            assert(std::isnan(control_input) == false);
            assert(std::isinf(control_input) == false);
            double real_control_input = GetControlValue(control_input);
            const double noise = GetNoiseValue(real_control_input, rng);
            const double noisy_control_input = real_control_input + noise;
            return noisy_control_input;
        }
    };

    typedef simple_uncertainty_models::SimpleUncertainVelocityActuator BaxterJointActuatorModel;
}

#endif // BAXTER_ACTUATOR_HELPERS_HPP
