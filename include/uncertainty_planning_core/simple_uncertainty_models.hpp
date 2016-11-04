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
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>

#ifndef SIMPLE_UNCERTAINTY_MODELS_HPP
#define SIMPLE_UNCERTAINTY_MODELS_HPP

namespace simple_uncertainty_models
{
    class SimpleUncertainSensor
    {
    protected:

        bool initialized_;
        mutable arc_helpers::TruncatedNormalDistribution noise_distribution_;

    public:

        SimpleUncertainSensor(const double noise_lower_bound, const double noise_upper_bound) : initialized_(true), noise_distribution_(0.0, std::max((fabs(noise_lower_bound) * 0.5), (fabs(noise_upper_bound) * 0.5)), noise_lower_bound, noise_upper_bound) {}

        SimpleUncertainSensor() : initialized_(false), noise_distribution_(0.0, 1.0, 0.0, 0.0) {}

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

    class SimpleUncertainVelocityActuator
    {
    protected:

        bool initialized_;
        mutable arc_helpers::TruncatedNormalDistribution noise_distribution_;
        double actuator_limit_;
        double noise_bound_;

    public:

        SimpleUncertainVelocityActuator(const double noise_bound, const double actuator_limit) : initialized_(true), noise_distribution_(0.0, 0.5 , -1.0, 1.0), actuator_limit_(std::abs(actuator_limit)), noise_bound_(std::abs(noise_bound)) {}

        SimpleUncertainVelocityActuator() : initialized_(false), noise_distribution_(0.0, 1.0, 0.0, 0.0), actuator_limit_(0.0), noise_bound_(0.0) {}

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
            const double real_control_input = GetControlValue(control_input);
            // This is the version used in WAFR SE(2) and SE(3)
            //const double noise = noise_distribution_(rng) * noise_bound_;
            // This is the version trialled in WAFR R(7), where noise is proportional to velocity
            const double noise = noise_distribution_(rng) * (noise_bound_ * real_control_input);
            // Combine noise with control input
            const double noisy_control_input = real_control_input + noise;
            return noisy_control_input;
        }
    };
}

#endif // SIMPLE_UNCERTAINTY_MODELS_HPP
