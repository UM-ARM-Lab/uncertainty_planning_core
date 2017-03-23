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

#ifndef SIMPLE_PID_CONTROLLER_HPP
#define SIMPLE_PID_CONTROLLER_HPP

namespace simple_pid_controller
{
    class PIDParams
    {
    protected:

        double kp_;
        double ki_;
        double kd_;
        double i_clamp_;

    public:

        PIDParams(const double kp, const double ki, const double kd, const double i_clamp) : kp_(kp), ki_(ki), kd_(kd), i_clamp_(i_clamp) {}

        PIDParams() : kp_(0.0), ki_(0.0), kd_(0.0), i_clamp_(0.0) {}

        inline double Kp() const
        {
            return kp_;
        }

        inline double Ki() const
        {
            return ki_;
        }

        inline double Kd() const
        {
            return kd_;
        }

        inline double Iclamp() const
        {
            return i_clamp_;
        }
    };

    class SimplePIDController
    {
    protected:

        bool initialized_;
        double kp_;
        double ki_;
        double kd_;
        double integral_clamp_;
        double error_integral_;
        double last_error_;

    public:

        SimplePIDController(const double kp, const double ki, const double kd, const double integral_clamp)
        {
            Initialize(kp, ki, kd, integral_clamp);
        }

        SimplePIDController(const PIDParams& params)
        {
            Initialize(params.Kp(), params.Ki(), params.Kd(), params.Iclamp());
        }

        SimplePIDController()
        {
            kp_ = 0.0;
            ki_ = 0.0;
            kd_ = 0.0;
            integral_clamp_ = 0.0;
            error_integral_ = 0.0;
            last_error_ = 0.0;
            initialized_ = false;
        }

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline PIDParams GetParams() const
        {
            return PIDParams(kp_, ki_, kd_, integral_clamp_);
        }

        inline void Zero()
        {
            last_error_ = 0.0;
            error_integral_ = 0.0;
        }

        inline void Initialize(const double kp, const double ki, const double kd, const double integral_clamp)
        {
            kp_ = std::abs(kp);
            ki_ = std::abs(ki);
            kd_ = std::abs(kd);
            integral_clamp_ = std::abs(integral_clamp);
            error_integral_ = 0.0;
            last_error_ = 0.0;
            initialized_ = true;
        }

        inline double ComputeFeedbackTerm(const double target_value, const double process_value, const double timestep)
        {
            // Get the current error
            const double current_error = target_value - process_value;
            return ComputeFeedbackTerm(current_error, timestep);
        }

        inline double ComputeFeedbackTerm(const double current_error, const double timestep)
        {
            // Update the integral error
            const double timestep_error_integral = ((current_error * 0.5) + (last_error_ * 0.5)) * timestep; // Trapezoidal integration over the timestep
            const double new_error_integral = error_integral_ + timestep_error_integral;
            error_integral_ = std::max(-integral_clamp_, std::min(integral_clamp_, new_error_integral));
            // Update the derivative error
            const double error_derivative = (current_error - last_error_) / timestep;
            // Update the stored error
            last_error_ = current_error;
            // Compute the correction
            const double correction = (current_error * kp_) + (error_integral_ * ki_) + (error_derivative * kd_);
            return correction;
        }
    };
}

#endif // SIMPLE_PID_CONTROLLER_HPP
