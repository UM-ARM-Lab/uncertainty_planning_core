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
            assert(isnan(process_value) == false);
            assert(isinf(process_value) == false);
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

    public:

        SimpleUncertainVelocityActuator(const double noise_lower_bound, const double noise_upper_bound, const double actuator_limit) : initialized_(true), noise_distribution_(0.0, std::max((fabs(noise_lower_bound) * 0.5), (fabs(noise_upper_bound) * 0.5)), noise_lower_bound, noise_upper_bound), actuator_limit_(fabs(actuator_limit)) {}

        SimpleUncertainVelocityActuator() : initialized_(false), noise_distribution_(0.0, 1.0, 0.0, 0.0), actuator_limit_(0.0) {}

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline double GetControlValue(const double control_input) const
        {
            assert(isnan(control_input) == false);
            assert(isinf(control_input) == false);
            double real_control_input = std::min(actuator_limit_, control_input);
            real_control_input = std::max(-actuator_limit_, real_control_input);
            return real_control_input;
        }

        template<typename RNG>
        inline double GetControlValue(const double control_input, RNG& rng) const
        {
            assert(isnan(control_input) == false);
            assert(isinf(control_input) == false);
            double real_control_input = GetControlValue(control_input);
            const double noise = noise_distribution_(rng);
            const double noisy_control_input = real_control_input + noise;
            return noisy_control_input;
        }
    };
}

#endif // SIMPLE_UNCERTAINTY_MODELS_HPP
