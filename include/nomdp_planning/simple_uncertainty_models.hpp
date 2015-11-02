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
#include "nomdp_planning/simple_pid_controller.hpp"

#ifndef SIMPLE_UNCERTAINTY_MODELS_HPP
#define SIMPLE_UNCERTAINTY_MODELS_HPP

namespace simple_uncertainty_models
{
    class SimpleUncertainSensor
    {
    protected:

        bool initialized_;
        mutable std::uniform_real_distribution<double> noise_distribution_;

    public:

        SimpleUncertainSensor(const double noise_lower_bound, const double noise_upper_bound) : initialized_(true), noise_distribution_(noise_lower_bound, noise_upper_bound) {}

        SimpleUncertainSensor() : initialized_(false), noise_distribution_(0.0, 0.0) {}

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline double GetSensorValue(const double process_value, std::mt19937_64& rng) const
        {
            double noise = noise_distribution_(rng);
            return process_value + noise;
        }
    };

    class SimpleUncertainVelocityActuator
    {
    protected:

        bool initialized_;
        mutable std::uniform_real_distribution<double> noise_distribution_;
        double actuator_limit_;

    public:

        SimpleUncertainVelocityActuator(const double noise_lower_bound, const double noise_upper_bound, const double actuator_limit) : initialized_(true), noise_distribution_(noise_lower_bound, noise_upper_bound), actuator_limit_(fabs(actuator_limit)) {}

        SimpleUncertainVelocityActuator() : initialized_(false), noise_distribution_(0.0, 0.0), actuator_limit_(0.0) {}

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline double GetControlValue(const double control_input, std::mt19937_64& rng) const
        {
            double real_control_input = std::min(actuator_limit_, control_input);
            real_control_input = std::max(-actuator_limit_, real_control_input);
            double noise = noise_distribution_(rng);
            double noisy_control_input = real_control_input + noise;
            return noisy_control_input;
        }
    };

    class Simple3dRobot
    {
    protected:

        bool initialized_;
        simple_pid_controller::SimplePIDController x_axis_controller_;
        simple_pid_controller::SimplePIDController y_axis_controller_;
        simple_pid_controller::SimplePIDController z_axis_controller_;
        simple_uncertainty_models::SimpleUncertainSensor x_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor y_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor z_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator x_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator y_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator z_axis_actuator_;
        Eigen::Vector3d position_;

    public:

        Simple3dRobot(const Eigen::Vector3d& initial_position, const double kp, const double ki, const double kd, const double integral_clamp, const double actuator_velocity_limit, const double sensor_noise_mag, const double actuator_noise_mag)
        {
            x_axis_controller_ = simple_pid_controller::SimplePIDController(kp, ki, kd, integral_clamp);
            y_axis_controller_ = simple_pid_controller::SimplePIDController(kp, ki, kd, integral_clamp);
            z_axis_controller_ = simple_pid_controller::SimplePIDController(kp, ki, kd, integral_clamp);
            x_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-sensor_noise_mag, sensor_noise_mag);
            y_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-sensor_noise_mag, sensor_noise_mag);
            z_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-sensor_noise_mag, sensor_noise_mag);
            x_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-actuator_noise_mag, actuator_noise_mag, actuator_velocity_limit);
            y_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-actuator_noise_mag, actuator_noise_mag, actuator_velocity_limit);
            z_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-actuator_noise_mag, actuator_noise_mag, actuator_velocity_limit);
            position_ = initial_position;
            initialized_ = true;
        }

        Simple3dRobot()
        {
            initialized_ = false;
        }

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline Eigen::Vector3d GetPosition() const
        {
            return position_;
        }

        inline void SetPosition(const Eigen::Vector3d& position)
        {
            position_ = position;
        }

        inline std::pair<Eigen::Vector3d, bool> MoveTowardsTarget(const Eigen::Vector3d& target, const double step_size, std::function<std::pair<Eigen::Vector3d, bool>(const Eigen::Vector3d&, const Eigen::Vector3d&)>& forward_simulation_fn, std::mt19937_64& rng)
        {
            Eigen::Vector3d start = GetPosition();
            double total_distance = (target - start).norm();
            int32_t num_steps = (int32_t)ceil(total_distance / step_size);
            num_steps = std::max(1, num_steps);
            bool collided = false;
            for (int32_t step_num = 1; step_num <= num_steps; step_num++)
            {
                double percent = (double)step_num / (double)num_steps;
                Eigen::Vector3d current_target = EigenHelpers::Interpolate(start, target, percent);
                Eigen::Vector3d current = GetPosition();
                double x_axis_control = x_axis_actuator_.GetControlValue(x_axis_controller_.ComputeFeedbackTerm(current_target.x(), x_axis_sensor_.GetSensorValue(current.x(), rng), 1.0), rng);
                double y_axis_control = y_axis_actuator_.GetControlValue(y_axis_controller_.ComputeFeedbackTerm(current_target.y(), y_axis_sensor_.GetSensorValue(current.y(), rng), 1.0), rng);
                double z_axis_control = z_axis_actuator_.GetControlValue(z_axis_controller_.ComputeFeedbackTerm(current_target.z(), z_axis_sensor_.GetSensorValue(current.z(), rng), 1.0), rng);
                Eigen::Vector3d control(x_axis_control, y_axis_control, z_axis_control);
                std::pair<Eigen::Vector3d, bool> result = forward_simulation_fn(current, control);
                assert(!isnan(control.x()));
                assert(!isnan(control.y()));
                assert(!isnan(control.z()));
                assert(!isnan(result.first.x()));
                assert(!isnan(result.first.y()));
                assert(!isnan(result.first.z()));
                SetPosition(result.first);
                if (result.second)
                {
                    collided = true;
                }
            }
            return std::pair<Eigen::Vector3d, bool>(GetPosition(), collided);
        }
    };

}

#endif // SIMPLE_UNCERTAINTY_MODELS_HPP
