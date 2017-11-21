#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <random>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>

#ifndef SIMPLE_UNCERTAINTY_MODELS_HPP
#define SIMPLE_UNCERTAINTY_MODELS_HPP

namespace simple_uncertainty_models
{
    class TruncatedNormalUncertainSensor
    {
    protected:

        bool initialized_;
        mutable arc_helpers::TruncatedNormalDistribution noise_distribution_;

    public:

        TruncatedNormalUncertainSensor(const double noise_lower_bound, const double noise_upper_bound) : initialized_(true), noise_distribution_(0.0, std::max((std::abs(noise_lower_bound) * 0.5), (std::abs(noise_upper_bound) * 0.5)), noise_lower_bound, noise_upper_bound) {}

        TruncatedNormalUncertainSensor() : initialized_(false), noise_distribution_(0.0, 1.0, 0.0, 0.0) {}

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        template<typename RNG>
        inline double GetSensorValue(const double process_value, RNG& rng) const
        {
            assert(std::isnan(process_value) == false);
            assert(std::isinf(process_value) == false);
            double noise = noise_distribution_(rng);
            return process_value + noise;
        }
    };

    class TruncatedNormalUncertainVelocityActuator
    {
    protected:

        bool initialized_;
        mutable arc_helpers::TruncatedNormalDistribution noise_distribution_;
        double velocity_limit_;
        double acceleration_limit_;
        double proportional_noise_bound_;
        double minimum_noise_bound_;

    public:

        TruncatedNormalUncertainVelocityActuator(const double velocity_limit, const double acceleration_limit, const double proportional_noise_bound, const double minimum_noise_bound, const double percent_variance) : initialized_(true), noise_distribution_(0.0, arc_helpers::ClampValue(std::abs(percent_variance), 0.0, 1.0), -1.0, 1.0), velocity_limit_(std::abs(velocity_limit)), acceleration_limit_(std::abs(acceleration_limit)), proportional_noise_bound_(std::abs(proportional_noise_bound)), minimum_noise_bound_(std::abs(minimum_noise_bound)) {}

        TruncatedNormalUncertainVelocityActuator() : initialized_(false), noise_distribution_(0.0, 1.0, 0.0, 0.0), velocity_limit_(0.0), acceleration_limit_(0.0), proportional_noise_bound_(0.0), minimum_noise_bound_(0.0) {}

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline double GetControlValue(const double control_input) const
        {
            assert(std::isnan(control_input) == false);
            assert(std::isinf(control_input) == false);
            return arc_helpers::ClampValue(control_input, -velocity_limit_, velocity_limit_);
        }

        template<typename RNG>
        inline double GetControlValue(const double control_input, RNG& rng) const
        {
            const double real_control_input = GetControlValue(control_input);
            // This is the version trialled in thesis, where noise is proportional to velocity with a "floor"
            // This can emulate proportional-only and percent-max noise with different parameters
            const double real_proportional_noise_bound = proportional_noise_bound_ * std::abs(real_control_input);
            const double real_minimum_noise_bound = minimum_noise_bound_ * velocity_limit_;
            const double real_noise_bound = std::max(real_proportional_noise_bound, real_minimum_noise_bound);
            const double real_noise = noise_distribution_(rng) * real_noise_bound;
            // Combine noise with control input
            const double noisy_control_input = real_control_input + real_noise;
            return noisy_control_input;
        }

        inline double GetMaxVelocity() const
        {
            return velocity_limit_;
        }

        inline double GetMaxAcceleration() const
        {
            return acceleration_limit_;
        }

        inline double GetMaxVelocityNoise(const double velocity) const
        {
            const double real_control_input = GetControlValue(velocity);
            if (real_control_input >= 0.0)
            {
                const double max_noise = std::max((proportional_noise_bound_ * real_control_input), (minimum_noise_bound_ * velocity_limit_));
                return max_noise;
            }
            else
            {
                const double max_noise = std::min((proportional_noise_bound_ * real_control_input), (minimum_noise_bound_ * -velocity_limit_));
                return max_noise;
            }
        }

        inline double GetMaxVelocityNoise() const
        {
            return GetMaxVelocityNoise(GetMaxVelocity());
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

    class SampledUncertainVelocityActuator
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

        SampledUncertainVelocityActuator(const std::shared_ptr<JointUncertaintySampleModel> model_ptr, const double max_velocity) : initialized_(true), actuator_limit_(std::abs(max_velocity)), model_ptr_(model_ptr) {}

        SampledUncertainVelocityActuator(const double max_velocity) : initialized_(true), actuator_limit_(std::abs(max_velocity)) {}

        SampledUncertainVelocityActuator() : initialized_(true), actuator_limit_(0.0) {}

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
}

#endif // SIMPLE_UNCERTAINTY_MODELS_HPP
